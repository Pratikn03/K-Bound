"""tests.test_production_gateway -- Comprehensive tests for the KGA Autonomous Safety Gateway."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import Optional, get_type_hints

import numpy as np

from kga.gateway.interface import FallbackReason, GatewayDecision
from kga.gateway.modes import DeploymentMode
from kga.gateway.safe_gateway import SafeInferenceGateway
from kga.guards.circuit_breaker import CircuitBreaker, CircuitState
from kga.guards.evidence_support import EvidenceSupportGuard
from kga.guards.numerical_health import NumericalHealthGuard
from kga.guards.streaming_monitor import StreamingDriftMonitor
from kga.observability.audit_logger import AuditLogger
from kga.observability.metrics import METRICS, InMemoryMetricsCollector
from kga.policy import Decision
from kga.registry.sealed_protocol import ProtocolIntegrityError, SealedProtocolManifest


def test_weight_divergence_annotations_resolve() -> None:
    """Release tooling can resolve the public guard's postponed annotations."""
    hints = get_type_hints(NumericalHealthGuard.check_weight_divergence)
    assert hints["return"] == tuple[bool, float, Optional[str]]


def test_http_server_defaults_to_loopback_and_preserves_explicit_host(monkeypatch) -> None:
    from kga.server import service

    calls = []
    sentinel = object()

    def capture_server(address, handler):
        calls.append(address)
        return sentinel

    monkeypatch.setattr(service.http.server, "HTTPServer", capture_server)
    monkeypatch.setattr(service.GatewayRequestHandler, "gateway", None)
    assert service.create_server(None) is sentinel
    assert calls == [("127.0.0.1", 8080)]
    assert service.create_server(None, host="192.0.2.1", port=9000) is sentinel
    assert calls[-1] == ("192.0.2.1", 9000)


class TestProductionGateway(unittest.TestCase):
    """Unit and integration test suite for the KGA Autonomous Safety Gateway."""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.num_classes = 5
        self.dim_features = 4

        # Dummy base model f0 (outputs balanced probabilities)
        self.base_model = lambda x: np.full((len(x), self.num_classes), 1.0 / self.num_classes)

        # Dummy adapted model fa (outputs peaked valid probabilities)
        def adapted_model(x: np.ndarray) -> np.ndarray:
            probs = np.full((len(x), self.num_classes), 0.05)
            probs[:, 0] = 0.80
            return probs
        self.candidate_adapter = adapted_model

        # Dummy feature extractor (returns 4D vector)
        self.feature_extractor = lambda x, y0, ya: np.array([0.1, 0.2, 0.3, 0.4])

        # Calibration features for evidence guard
        self.calib_features = self.rng.normal(0.0, 1.0, size=(100, self.dim_features))

    def test_fallback_on_zero_containment(self) -> None:
        """When the compound interval contains zero, gateway MUST abstain and serve f0."""
        # delta_hat = 0.02, total radius will be > 0.02, so interval contains 0
        gateway = SafeInferenceGateway(
            base_model=self.base_model,
            candidate_adapter=self.candidate_adapter,
            benefit_estimator=lambda z: 0.02,
            feature_extractor=self.feature_extractor,
            calibration_epsilon=0.05,
            m_min=16,
        )

        test_x = np.ones((32, 10))
        preds, decision = gateway.predict(test_x)

        self.assertEqual(decision.action, Decision.ABSTAIN)
        self.assertEqual(decision.served_model, "frozen_base")
        self.assertEqual(decision.fallback_reason, FallbackReason.ZERO_CONTAINMENT)
        self.assertTrue(decision.lower_bound <= 0.0 <= decision.upper_bound)
        # Verify served predictions match base model
        np.testing.assert_allclose(preds, self.base_model(test_x))

    def test_strict_adapt_certification(self) -> None:
        """When lower bound is strictly positive with adequate sample size, gateway serves fa."""
        # delta_hat = 0.50, epsilon = 0.01, b(100, 0.05) approx 0.136 -> lower bound approx 0.354 > 0
        gateway = SafeInferenceGateway(
            base_model=self.base_model,
            candidate_adapter=self.candidate_adapter,
            benefit_estimator=lambda z: 0.50,
            feature_extractor=self.feature_extractor,
            calibration_epsilon=0.01,
            m_min=16,
        )

        test_x = np.ones((100, 10))
        preds, decision = gateway.predict(test_x)

        self.assertEqual(decision.action, Decision.ADAPT)
        self.assertEqual(decision.served_model, "adapted_candidate")
        self.assertEqual(decision.fallback_reason, FallbackReason.NONE)
        self.assertGreater(decision.lower_bound, 0.0)
        np.testing.assert_allclose(preds, self.candidate_adapter(test_x))

    def test_fallback_on_estimated_harmful(self) -> None:
        """When upper bound is strictly negative, gateway freezes and serves f0."""
        gateway = SafeInferenceGateway(
            base_model=self.base_model,
            candidate_adapter=self.candidate_adapter,
            benefit_estimator=lambda z: -0.50,
            feature_extractor=self.feature_extractor,
            calibration_epsilon=0.01,
            m_min=16,
        )

        test_x = np.ones((100, 10))
        preds, decision = gateway.predict(test_x)

        self.assertEqual(decision.action, Decision.FREEZE)
        self.assertEqual(decision.served_model, "frozen_base")
        self.assertEqual(decision.fallback_reason, FallbackReason.ESTIMATED_HARMFUL)
        self.assertLess(decision.upper_bound, 0.0)
        np.testing.assert_allclose(preds, self.base_model(test_x))

    def test_sample_size_starvation_guard(self) -> None:
        """When m < m_min, gateway falls back to f0 without invoking benefit estimator."""
        called = False

        def spy_estimator(z: np.ndarray) -> float:
            nonlocal called
            called = True
            return 1.0

        gateway = SafeInferenceGateway(
            base_model=self.base_model,
            candidate_adapter=self.candidate_adapter,
            benefit_estimator=spy_estimator,
            feature_extractor=self.feature_extractor,
            calibration_epsilon=0.01,
            m_min=32,
        )

        test_x = np.ones((10, 10))  # m=10 < 32
        preds, decision = gateway.predict(test_x)

        self.assertFalse(called)
        self.assertEqual(decision.action, Decision.ABSTAIN)
        self.assertEqual(decision.fallback_reason, FallbackReason.SAMPLE_SIZE_STARVATION)
        self.assertEqual(decision.served_model, "frozen_base")

    def test_numerical_health_guard_nan_injection(self) -> None:
        """When adapter produces NaN or degenerate entropy, gateway intercepts and serves f0."""
        def bad_adapter(x: np.ndarray) -> np.ndarray:
            arr = np.full((len(x), self.num_classes), 0.2)
            arr[0, 0] = np.nan
            return arr

        gateway = SafeInferenceGateway(
            base_model=self.base_model,
            candidate_adapter=bad_adapter,
            benefit_estimator=lambda z: 0.50,
            feature_extractor=self.feature_extractor,
            calibration_epsilon=0.01,
            m_min=16,
        )

        test_x = np.ones((20, 10))
        preds, decision = gateway.predict(test_x)

        self.assertEqual(decision.action, Decision.ABSTAIN)
        self.assertEqual(decision.served_model, "frozen_base")
        self.assertEqual(decision.fallback_reason, FallbackReason.NUMERICAL_INSTABILITY)
        np.testing.assert_allclose(preds, self.base_model(test_x))

    def test_evidence_support_guard_ood(self) -> None:
        """When evidence Z lies far outside calibration support, guard trips to ABSTAIN."""
        guard = EvidenceSupportGuard(self.calib_features, max_mahalanobis_distance=4.0)

        # Extractor returns an extreme outlier
        gateway = SafeInferenceGateway(
            base_model=self.base_model,
            candidate_adapter=self.candidate_adapter,
            benefit_estimator=lambda z: 0.50,
            feature_extractor=lambda x, y0, ya: np.array([50.0, 50.0, 50.0, 50.0]),
            calibration_epsilon=0.01,
            evidence_guard=guard,
            m_min=16,
        )

        test_x = np.ones((20, 10))
        preds, decision = gateway.predict(test_x)

        self.assertEqual(decision.action, Decision.ABSTAIN)
        self.assertEqual(decision.served_model, "frozen_base")
        self.assertEqual(decision.fallback_reason, FallbackReason.OOD_EVIDENCE)

    def test_latency_sla_watchdog(self) -> None:
        """When adaptation exceeds max_latency_ms, gateway intercepts and falls back to f0."""
        def slow_adapter(x: np.ndarray) -> np.ndarray:
            time.sleep(0.05)  # 50ms sleep
            return self.candidate_adapter(x)

        gateway = SafeInferenceGateway(
            base_model=self.base_model,
            candidate_adapter=slow_adapter,
            benefit_estimator=lambda z: 0.50,
            feature_extractor=self.feature_extractor,
            calibration_epsilon=0.01,
            max_latency_ms=10.0,  # SLA = 10ms
            m_min=16,
        )

        test_x = np.ones((20, 10))
        preds, decision = gateway.predict(test_x)

        self.assertEqual(decision.action, Decision.ABSTAIN)
        self.assertEqual(decision.served_model, "frozen_base")
        self.assertEqual(decision.fallback_reason, FallbackReason.LATENCY_TIMEOUT)

    def test_circuit_breaker_trip_and_latch(self) -> None:
        """Consecutive failures trip circuit breaker to OPEN, latching future requests to f0."""
        cb = CircuitBreaker(failure_threshold=2, recovery_cooldown_seconds=100.0)

        def crashing_adapter(x: np.ndarray) -> np.ndarray:
            raise RuntimeError("CUDA out of memory in test-time adapter")

        gateway = SafeInferenceGateway(
            base_model=self.base_model,
            candidate_adapter=crashing_adapter,
            benefit_estimator=lambda z: 0.50,
            feature_extractor=self.feature_extractor,
            calibration_epsilon=0.01,
            circuit_breaker=cb,
            m_min=16,
        )

        test_x = np.ones((20, 10))

        # Failure 1
        gateway.predict(test_x)
        self.assertEqual(cb.state, CircuitState.CLOSED)

        # Failure 2 -> trips to OPEN
        gateway.predict(test_x)
        self.assertEqual(cb.state, CircuitState.OPEN)

        # Request 3: Short-circuited immediately without calling adapter
        preds, decision = gateway.predict(test_x)
        self.assertEqual(decision.fallback_reason, FallbackReason.CIRCUIT_BREAKER_OPEN)
        self.assertEqual(decision.served_model, "frozen_base")

        # Manual reset restores CLOSED
        cb.reset()
        self.assertEqual(cb.state, CircuitState.CLOSED)

    def test_streaming_martingale_drift_monitor(self) -> None:
        """Sequential e-values crossing 1/alpha trip the drift monitor and circuit breaker."""
        monitor = StreamingDriftMonitor(alpha=0.10)  # threshold = 10.0
        cb = CircuitBreaker()

        # Update with e-value 2.0 repeatedly
        monitor.update(2.0)  # W=2
        self.assertFalse(monitor.is_tripped)
        monitor.update(3.0)  # W=6
        self.assertFalse(monitor.is_tripped)
        status = monitor.update(2.0)  # W=12 >= 10.0
        self.assertTrue(status.tripped)
        self.assertTrue(monitor.is_tripped)

    def test_shadow_canary_mode(self) -> None:
        """In SHADOW_CANARY mode, live traffic is 100% served by f0 even when ADAPT is certified."""
        gateway = SafeInferenceGateway(
            base_model=self.base_model,
            candidate_adapter=self.candidate_adapter,
            benefit_estimator=lambda z: 0.50,
            feature_extractor=self.feature_extractor,
            calibration_epsilon=0.01,
            mode=DeploymentMode.SHADOW_CANARY,
            m_min=16,
        )

        test_x = np.ones((50, 10))
        preds, decision = gateway.predict(test_x)

        self.assertEqual(decision.action, Decision.ADAPT)
        self.assertEqual(decision.served_model, "frozen_base")
        self.assertEqual(decision.fallback_reason, FallbackReason.SHADOW_MODE)
        np.testing.assert_allclose(preds, self.base_model(test_x))

    def test_sealed_protocol_manifest_tampering(self) -> None:
        """SealedProtocolManifest verifies digest integrity and catches tampered weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "protocol.json"

            manifest = SealedProtocolManifest(
                protocol_name="kbound-production-v1",
                protocol_version="1.0.0",
                model_name="resnet50",
                adapter_name="tent",
                weights_sha256="abcdef1234567890",
                calibration_data_sha256="123456abcdef7890",
                feature_schema=["ks_mean", "disagree", "entropy", "margin"],
                alpha=0.10,
                delta=0.05,
                m_min=16,
                epsilon_calibrated=0.025,
            )
            manifest.save(manifest_path)

            # Clean load
            loaded = SealedProtocolManifest.load(manifest_path, verify_seal=True)
            self.assertEqual(loaded.model_name, "resnet50")

            # Tamper verification
            self.assertTrue(loaded.verify_runtime_artifacts(actual_weights_hash="abcdef1234567890"))
            with self.assertRaises(ProtocolIntegrityError):
                loaded.verify_runtime_artifacts(actual_weights_hash="tampered_weights_hash")

    def test_observability_and_audit_logger(self) -> None:
        """AuditLogger records valid JSONL lines and Prometheus metrics accumulate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "audit.jsonl"
            logger = AuditLogger(log_file)

            gateway = SafeInferenceGateway(
                base_model=self.base_model,
                candidate_adapter=self.candidate_adapter,
                benefit_estimator=lambda z: 0.50,
                feature_extractor=self.feature_extractor,
                calibration_epsilon=0.01,
                audit_logger=logger,
                m_min=16,
            )

            test_x = np.ones((30, 10))
            gateway.predict(test_x, request_id="req-12345")

            self.assertTrue(log_file.exists())
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["request_id"], "req-12345")
            self.assertEqual(record["decision"]["action"], "ADAPT")

            # Verify Prometheus text generation
            prom_text = METRICS.generate_prometheus_text()
            self.assertIn("kga_requests_total", prom_text)


class TestGatewayHTTPServer(unittest.TestCase):
    """In-process mock test for the Gateway HTTP request handler."""

    def setUp(self) -> None:
        import io
        from kga.server.service import GatewayRequestHandler

        base_model = lambda x: np.full((len(x), 3), 1.0 / 3.0)
        candidate_adapter = lambda x: np.array([[0.8, 0.1, 0.1]] * len(x))
        gateway = SafeInferenceGateway(
            base_model=base_model,
            candidate_adapter=candidate_adapter,
            benefit_estimator=lambda z: 0.50,
            feature_extractor=lambda x, y0, ya: np.array([0.1, 0.2]),
            calibration_epsilon=0.01,
            m_min=2,
        )

        class MockHandler(GatewayRequestHandler):
            def __init__(self, path: str, method: str = "GET", body: bytes = b"") -> None:
                self.gateway = gateway
                self.path = path
                self.command = method
                self.rfile = io.BytesIO(body)
                self.wfile = io.BytesIO()
                self.headers = {"Content-Length": str(len(body)), "Content-Type": "application/json"}
                self._headers_buffer = []

            def send_response(self, code: int, message: Optional[str] = None) -> None:
                self.response_code = code

            def send_header(self, keyword: str, value: str) -> None:
                self._headers_buffer.append((keyword, value))

            def end_headers(self) -> None:
                pass

        self.mock_handler_cls = MockHandler

    def test_healthz_and_metrics_endpoints(self) -> None:
        # Generate at least one request to populate metrics
        payload = json.dumps({"inputs": [[1.0, 2.0]] * 20, "request_id": "setup-req"}).encode("utf-8")
        h_prep = self.mock_handler_cls(path="/v1/predict", method="POST", body=payload)
        h_prep.do_POST()

        # GET /healthz
        handler = self.mock_handler_cls(path="/healthz", method="GET")
        handler.do_GET()
        self.assertEqual(handler.response_code, 200)
        data = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["circuit_state"], "closed")

        # GET /metrics
        handler = self.mock_handler_cls(path="/metrics", method="GET")
        handler.do_GET()
        self.assertEqual(handler.response_code, 200)
        metrics_body = handler.wfile.getvalue().decode("utf-8")
        self.assertIn("kga_requests_total", metrics_body)

    def test_predict_and_reset_endpoints(self) -> None:
        # POST /v1/predict with m=20 (sufficient for Hoeffding concentration to certify ADAPT)
        payload = json.dumps({"inputs": [[1.0, 2.0]] * 20, "request_id": "test-req"}).encode("utf-8")
        handler = self.mock_handler_cls(path="/v1/predict", method="POST", body=payload)
        handler.do_POST()
        self.assertEqual(handler.response_code, 200)
        data = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertIn("predictions", data)
        self.assertEqual(data["decision"]["action"], "ADAPT")

        # POST /circuit-breaker/reset
        handler = self.mock_handler_cls(path="/circuit-breaker/reset", method="POST", body=b"{}")
        handler.do_POST()
        self.assertEqual(handler.response_code, 200)
        data = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(data["circuit_state"], "closed")

    def test_sensitivity_frontier_and_break_even_beta(self) -> None:
        from kga.certificate import Certificate
        from kga.sensitivity import (
            SensitivityFrontier,
            compute_break_even_beta,
            compute_sensitivity_frontier,
            compute_sensitivity_interval,
        )

        # Case 1: ADAPT certificate (lower > 0)
        cert_adapt = Certificate(delta_hat=0.10, epsilon=0.04, method="ebern", alpha=0.05, n=100)
        self.assertAlmostEqual(cert_adapt.lower, 0.06)
        self.assertAlmostEqual(cert_adapt.upper, 0.14)

        frontier_obj = SensitivityFrontier(cert_adapt)
        self.assertAlmostEqual(frontier_obj.break_even_beta, 0.03)
        self.assertAlmostEqual(compute_break_even_beta(cert_adapt.lower, cert_adapt.upper), 0.03)

        # Sensitivity interval at beta=0.01: [0.06 - 0.02, 0.14 + 0.02] = [0.04, 0.16] -> ADAPT
        low, high = frontier_obj.interval(0.01)
        self.assertAlmostEqual(low, 0.04)
        self.assertAlmostEqual(high, 0.16)

        # Sensitivity frontier
        frontier = frontier_obj.frontier(max_beta=0.05, steps=6)
        self.assertEqual(len(frontier), 6)
        # At beta=0.0, action is ADAPT
        self.assertEqual(frontier[0]["action"], "ADAPT")
        # At beta=0.04 > break_even_beta (0.03), action becomes ABSTAIN
        self.assertEqual(frontier[-1]["action"], "ABSTAIN")

        # Case 2: FREEZE certificate (upper < 0)
        cert_freeze = Certificate(delta_hat=-0.10, epsilon=0.04, method="ebern", alpha=0.05, n=100)
        freeze_frontier = SensitivityFrontier(cert_freeze)
        self.assertAlmostEqual(freeze_frontier.break_even_beta, 0.03)

        # Case 3: ABSTAIN certificate (lower <= 0 <= upper)
        cert_abstain = Certificate(delta_hat=0.02, epsilon=0.05, method="ebern", alpha=0.05, n=100)
        abstain_frontier = SensitivityFrontier(cert_abstain)
        self.assertEqual(abstain_frontier.break_even_beta, 0.0)
        self.assertFalse(abstain_frontier.summary()["is_robust_to_1pct_bias"])

    def test_multi_architecture_and_partial_adaptation_probes(self) -> None:
        from experiments.kbound.benchmarks.multi_architecture_eval import run_multi_architecture_probe
        from experiments.kbound.cct20.partial_adaptation_gate import run_cct20_partial_adaptation_probe

        arch_report = run_multi_architecture_probe(seed=42, n_samples=200)
        self.assertIn("resnet18", arch_report["architectures"])
        self.assertIn("vit_b16", arch_report["architectures"])
        self.assertIn("convnext_tiny", arch_report["architectures"])
        self.assertEqual(arch_report["summary"]["safeguard_success_rate"], 1.0)

        cct_report = run_cct20_partial_adaptation_probe(seed=42, n_per_loc=100)
        self.assertIn("full_tta", cct_report["regimes"])
        self.assertIn("entropy_gated_partial", cct_report["regimes"])
        # Full TTA should exhibit zero adaptations due to high drift risk (conservative retention)
        self.assertEqual(cct_report["regimes"]["full_tta"]["adapt_count"], 0)
        self.assertEqual(cct_report["regimes"]["full_tta"]["false_adapt_count"], 0)
        # All regimes strictly maintain 0 false-adaptations (Theorem 3 guarantee)
        for r_name, r_data in cct_report["regimes"].items():
            self.assertEqual(r_data["false_adapt_count"], 0)
            self.assertEqual(r_data["adapt_count"] + r_data["freeze_count"] + r_data["abstain_count"], 5)


if __name__ == "__main__":
    unittest.main()
