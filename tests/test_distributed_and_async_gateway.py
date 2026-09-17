"""tests.test_distributed_and_async_gateway -- Tests for distributed backends and async gateway."""

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from kga.gateway.async_gateway import AsyncSafeInferenceGateway
from kga.gateway.interface import FallbackReason
from kga.guards.circuit_breaker import CircuitBreaker, CircuitState
from kga.guards.state_backend import DistributedCircuitState, FileStateBackend, InMemoryStateBackend
from kga.integrations.torch_adapter import TorchModelAdapter
from kga.policy import Decision


class TestDistributedAndAsyncGateway(unittest.TestCase):
    """Test suite for distributed state synchronization and async gateway."""

    def setUp(self) -> None:
        self.num_classes = 4
        self.base_model = lambda x: np.full((len(x), self.num_classes), 0.25)
        self.candidate_adapter = lambda x: np.array([[0.7, 0.1, 0.1, 0.1]] * len(x))
        self.feature_extractor = lambda x, y0, ya: np.array([0.2, 0.3, 0.4, 0.5])

    def test_file_state_backend_cross_worker_sync(self) -> None:
        """Two independent circuit breakers share and synchronize state through FileStateBackend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "circuit_state.json"
            backend1 = FileStateBackend(state_file)
            backend2 = FileStateBackend(state_file)

            cb1 = CircuitBreaker(failure_threshold=2, backend=backend1)
            cb2 = CircuitBreaker(failure_threshold=2, backend=backend2)

            self.assertEqual(cb1.state, CircuitState.CLOSED)
            self.assertEqual(cb2.state, CircuitState.CLOSED)

            # Worker 1 experiences consecutive failures and trips
            cb1.record_failure("Failure 1")
            cb1.record_failure("Failure 2")
            self.assertEqual(cb1.state, CircuitState.OPEN)

            # Worker 2 automatically observes OPEN state on next query
            self.assertEqual(cb2.state, CircuitState.OPEN)
            self.assertFalse(cb2.allow_adaptation_attempt())

            # Worker 2 resets breaker
            cb2.reset()
            self.assertEqual(cb2.state, CircuitState.CLOSED)

            # Worker 1 observes CLOSED state
            self.assertEqual(cb1.state, CircuitState.CLOSED)
            self.assertTrue(cb1.allow_adaptation_attempt())

    def test_async_safe_inference_gateway_predict(self) -> None:
        """AsyncSafeInferenceGateway executes non-blocking predictions and certifies ADAPT."""
        async def run_test() -> None:
            gateway = AsyncSafeInferenceGateway(
                base_model=self.base_model,
                candidate_adapter=self.candidate_adapter,
                benefit_estimator=lambda z: 0.50,
                feature_extractor=self.feature_extractor,
                calibration_epsilon=0.01,
                m_min=16,
            )

            test_x = np.ones((32, 10))
            preds, decision = await gateway.predict(test_x, request_id="async-req-1")

            self.assertEqual(decision.action, Decision.ADAPT)
            self.assertEqual(decision.served_model, "adapted_candidate")
            self.assertEqual(decision.fallback_reason, FallbackReason.NONE)
            np.testing.assert_allclose(preds, self.candidate_adapter(test_x))

        asyncio.run(run_test())

    def test_async_safe_inference_gateway_timeout(self) -> None:
        """AsyncSafeInferenceGateway intercepts slow adapters and safely falls back to f0."""
        def slow_adapter(x: np.ndarray) -> np.ndarray:
            time.sleep(0.08)  # 80ms sleep
            return self.candidate_adapter(x)

        async def run_test() -> None:
            gateway = AsyncSafeInferenceGateway(
                base_model=self.base_model,
                candidate_adapter=slow_adapter,
                benefit_estimator=lambda z: 0.50,
                feature_extractor=self.feature_extractor,
                calibration_epsilon=0.01,
                max_latency_ms=10.0,  # 10ms SLA
                m_min=16,
            )

            test_x = np.ones((32, 10))
            preds, decision = await gateway.predict(test_x, request_id="async-req-timeout")

            self.assertEqual(decision.action, Decision.ABSTAIN)
            self.assertEqual(decision.served_model, "frozen_base")
            self.assertEqual(decision.fallback_reason, FallbackReason.LATENCY_TIMEOUT)
            np.testing.assert_allclose(preds, self.base_model(test_x))

        asyncio.run(run_test())

    def test_torch_model_adapter_evidence_extraction(self) -> None:
        """TorchModelAdapter extracts standardized label-free evidence vector Z."""
        base_probs = np.array([[0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]])
        adapted_probs = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.2, 0.3, 0.4]])

        adapter = TorchModelAdapter(model=self.base_model)
        z = TorchModelAdapter.extract_standard_evidence_features(np.zeros((2, 10)), base_probs, adapted_probs)

        self.assertEqual(len(z), 4)
        # Check disagreement: sample 0 has argmax 0 vs 0 (no disagree if base argmax=0), check rate in [0, 1]
        self.assertTrue(0.0 <= z[0] <= 1.0)
        # Check confidence difference
        self.assertIsInstance(z[1], float)
        # Check sharpness
        self.assertTrue(0.0 <= z[2] <= 1.0)
        # Check margin
        self.assertTrue(0.0 <= z[3] <= 1.0)


if __name__ == "__main__":
    unittest.main()
