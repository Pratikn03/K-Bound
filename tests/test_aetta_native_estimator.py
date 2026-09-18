"""Lightweight contracts plus an explicitly requested real-native CPU stage.

Run native conformance with the AETTA interpreter and ``--native-stage``.
Default discovery must not import PyTorch or run a model implicitly.
"""
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import resource
import signal
import sys
import tempfile
import textwrap
import time
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/research/kbound/scripts/aetta_native_estimator.py"
SOURCE = ROOT / "external/aetta_official"


def bridge_module():
    if not SCRIPT.is_file():
        raise AssertionError("The authenticated native AETTA estimator bridge is missing")
    spec = importlib.util.spec_from_file_location("tested_aetta_bridge", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BridgeContracts(unittest.TestCase):
    def test_bridge_import_has_no_native_runtime_side_effect(self):
        before = set(sys.modules)
        bridge_module()
        self.assertNotIn("torch", set(sys.modules) - before)

    def test_modified_source_is_rejected_before_execution(self):
        bridge = bridge_module()
        with tempfile.TemporaryDirectory(prefix="aetta_changed_source_") as directory:
            root = Path(directory)
            (root / "learner").mkdir()
            (root / "learner/dnn.py").write_text("raise RuntimeError('must not execute')\n")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                bridge.authenticate_source(root)

    def test_symlink_source_is_rejected(self):
        bridge = bridge_module()
        with tempfile.TemporaryDirectory(prefix="aetta_link_source_") as directory:
            root = Path(directory)
            (root / "learner").mkdir()
            payload = root / "payload.py"
            payload.write_text("not authenticated\n")
            (root / "learner/dnn.py").symlink_to(payload)
            with self.assertRaisesRegex(ValueError, "regular.*non-symlink"):
                bridge.authenticate_source(root)

    def test_missing_source_fails_closed(self):
        bridge = bridge_module()
        with tempfile.TemporaryDirectory(prefix="aetta_missing_source_") as directory:
            with self.assertRaisesRegex(ValueError, "missing|regular"):
                bridge.authenticate_source(Path(directory))

    def test_explicit_receipt_checks_fixed_origin_pin_url_and_hash_not_user_hash(self):
        bridge = bridge_module()
        self.assertIn("auth_receipt", inspect.signature(bridge.authenticate_source).parameters,
                      "The explicit pinned-raw-source authentication route is missing")
        with tempfile.TemporaryDirectory(prefix="aetta_receipt_contract_") as directory:
            root = Path(directory).resolve()
            payload = b"# synthetic source contract, never executed\n"
            source_file = root / "fixture.py"
            source_file.write_bytes(payload)
            fixed_hash = hashlib.sha256(payload).hexdigest()
            receipt = {
                "schema": "kbound_pinned_raw_source_auth_v1",
                "origin": "https://github.com/taeckyung/AETTA",
                "commit": bridge.SOURCE_COMMIT,
                "method": "https_commit_pinned_raw_files",
                "files": {"fixture.py": {
                    "url": f"https://raw.githubusercontent.com/taeckyung/AETTA/{bridge.SOURCE_COMMIT}/fixture.py",
                    "sha256": fixed_hash,
                }},
            }
            auth_path = root / "authentication.json"
            auth_path.write_text(json.dumps(receipt))
            # This is the external Git boundary only. Exact native estimator
            # conformance below executes real upstream functions without mocks.
            with mock.patch.object(bridge, "SOURCE_SHA256", {"fixture.py": fixed_hash}), \
                    mock.patch.object(bridge.subprocess, "check_output", side_effect=AssertionError("No Git or lazy fetch allowed")):
                self.assertEqual(bridge.authenticate_source(root, auth_receipt=auth_path), {"fixture.py": payload})
                for field in ("origin", "commit"):
                    changed = json.loads(json.dumps(receipt))
                    changed[field] = "untrusted"
                    auth_path.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):
                        bridge.authenticate_source(root, auth_receipt=auth_path)
                for field in ("url", "sha256"):
                    changed = json.loads(json.dumps(receipt))
                    changed["files"]["fixture.py"][field] = "untrusted"
                    auth_path.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):
                        bridge.authenticate_source(root, auth_receipt=auth_path)
                changed_payload = b"# different source\n"
                source_file.write_bytes(changed_payload)
                receipt["files"]["fixture.py"]["sha256"] = hashlib.sha256(changed_payload).hexdigest()
                auth_path.write_text(json.dumps(receipt))
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    bridge.authenticate_source(root, auth_receipt=auth_path)

    def test_git_authentication_disables_implicit_lazy_fetch(self):
        bridge = bridge_module()
        with tempfile.TemporaryDirectory(prefix="aetta_git_boundary_") as directory:
            root = Path(directory).resolve()
            payload = b"# synthetic source contract, never executed\n"
            root.joinpath("fixture.py").write_bytes(payload)
            calls = []

            def git_boundary(argv, **kwargs):
                self.assertEqual(argv[:3], ["git", "-c", "protocol.allow=never"],
                                 "Even older Git must reject any implicit network transport")
                self.assertEqual(kwargs.get("env", {}).get("GIT_NO_LAZY_FETCH"), "1",
                                 "Source verification may not implicitly download or write Git objects")
                calls.append(argv)
                return bridge.SOURCE_COMMIT if "rev-parse" in argv else payload

            with mock.patch.object(bridge, "SOURCE_SHA256", {"fixture.py": hashlib.sha256(payload).hexdigest()}), \
                    mock.patch.object(bridge.subprocess, "check_output", side_effect=git_boundary):
                self.assertEqual(bridge.authenticate_source(root), {"fixture.py": payload})
            self.assertEqual(len(calls), 2)


def native_oracle(source, model):
    """Independent line-slice extraction: never calls bridge extraction helpers."""
    import torch
    import torch.nn.functional as F
    import torchvision

    namespace = dict(__name__="independently_extracted_native_aetta_oracle",
                     torch=torch, F=F, torchvision=torchvision)
    namespace["conf"] = types.SimpleNamespace(args=types.SimpleNamespace(
        opt={"indices_in_1k": None}, dataset="imagenetoutdist", dropout_rate=0.5))
    losses = source.joinpath("utils/loss_functions.py").read_text()
    tree = ast.parse(losses)
    for name in ("calc_energy", "softmax_entropy"):
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        # Preserve native TorchScript decorators and original source locations.
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        lines = losses.splitlines(keepends=True)
        snippet = "\n" * (start - 1) + "".join(lines[start - 1:node.end_lineno])
        exec(compile(snippet, str(source / "utils/loss_functions.py"), "exec"), namespace)
    normalization = {}
    exec(compile(source.joinpath("utils/normalize_layer.py").read_bytes(),
                 str(source / "utils/normalize_layer.py"), "exec"), normalization)
    dnn = source.joinpath("learner/dnn.py").read_text()
    cls = next(n for n in ast.parse(dnn).body if isinstance(n, ast.ClassDef) and n.name == "DNN")
    methods = {}
    for name in ("model_inference", "evaluate_dropout", "aetta"):
        node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
        snippet = textwrap.dedent("".join(dnn.splitlines(keepends=True)[node.lineno - 1:node.end_lineno]))
        exec(compile("\n" * (node.lineno - 1) + snippet,
                     str(source / "learner/dnn.py"), "exec"), namespace)
        methods[name] = namespace[name]
    oracle = type("IndependentNativeOracle", (), methods)()
    oracle.net = torch.nn.Sequential(normalization["get_normalize_layer"]("imagenetoutdist"), model)
    oracle.est_ema_dropout = None
    oracle.acc_est_json = {key: [] for key in (
        "est_dropout", "est_dropout_avg_entropy", "est_dropout_softmax_mean",
        "est_dropout_softmax_std", "aetta")}
    return oracle


def run_native_stage(auth_receipt=None):
    """Explicit model-stage tests; not loaded as default unittest/pytest tests."""
    import torch
    import torchvision

    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("90-second native fixture limit")))
    signal.alarm(90)
    bridge = bridge_module()
    authenticated = bridge.NativeAETTASource(SOURCE, auth_receipt=auth_receipt)
    torch.manual_seed(71)
    model = authenticated.create_model()
    # Synthetic branch fixture only: scale random head logits so the entropy
    # correction is not always saturated. This is not a pretrained model.
    with torch.no_grad():
        model.fc.weight.mul_(0.001)
        model.fc.bias.zero_()
        model.fc.bias[0] = 0.2
    model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    episode = authenticated.new_episode(model)
    oracle = native_oracle(SOURCE, model)
    generator = torch.Generator().manual_seed(1729)
    images = torch.rand((1, 3, 224, 224), generator=generator)
    checks = {}
    checks["full_native_resnet50_1000_classes"] = (
        model.fc.out_features == 1000 and [len(model.layer1), len(model.layer2),
                                         len(model.layer3), len(model.layer4)] == [3, 4, 6, 3])
    checks["no_labels_interface"] = list(inspect.signature(episode.estimate).parameters) == ["images"]
    bridge_results = []
    forward_dropout = []
    hook = model.register_forward_pre_hook(
        lambda module, args, kwargs: forward_dropout.append(kwargs.get("dropout", 0.0)), with_kwargs=True)
    first_seed = 819
    first = None
    for index in range(2):
        torch.manual_seed(first_seed + index)
        before = len(forward_dropout)
        result = episode.estimate(images if index == 0 else 1.0 - images)
        bridge_results.append(result)
        checks[f"native_ten_stage_dropout_draws_batch_{index}"] = (
            forward_dropout[before:] == [0.0] + [0.5] * 10)
        torch.manual_seed(first_seed + index)
        with torch.no_grad():
            oracle.aetta(images if index == 0 else 1.0 - images, None)
        checks[f"exact_native_estimator_state_batch_{index}"] = (
            episode.estimator_state == oracle.acc_est_json
            and episode.ema_error == oracle.est_ema_dropout
            and result["estimated_accuracy_percent"] == oracle.acc_est_json["aetta"][-1])
        if index == 0:
            first = result
    hook.remove()
    checks["fixture_exercises_unclipped_entropy_correction"] = any(
        0.0 < (1.0 - result["dropout_agreement"]) / (result["dropout_mean_entropy"] / 6.9078) ** 3 < 0.999
        for result in bridge_results)
    checks["native_ema_retained_across_batches"] = len(episode.estimator_state["aetta"]) == 2
    fresh = authenticated.new_episode(model)
    checks["fresh_episode_starts_without_ema"] = fresh.ema_error is None and not fresh.estimator_state["aetta"]
    torch.manual_seed(first_seed)
    checks["fresh_episode_reproduces_first_state"] = fresh.estimate(images) == first
    checks["previous_episode_not_reset_by_new_episode"] = len(episode.estimator_state["aetta"]) == 2
    checks["no_model_update_by_estimator"] = all(torch.equal(model_state[k], v) for k, v in model.state_dict().items())
    before = episode.estimator_state
    try:
        episode.estimate(images, labels=torch.zeros(1))
    except TypeError:
        checks["labels_argument_rejected"] = episode.estimator_state == before
    for name, invalid in (("already_normalized_input", images - 2.0),
                          ("wrong_spatial_size", images[:, :, :32, :32]),
                          ("empty_batch", images[:0]),
                          ("nonfinite_input", images * float("nan"))):
        try:
            episode.estimate(invalid)
            checks[name] = False
        except ValueError:
            checks[name] = episode.estimator_state == before
    try:
        authenticated.new_episode(torchvision.models.resnet50(weights=None))
        checks["non_native_head_dropout_or_bn_model_rejected"] = False
    except ValueError:
        checks["non_native_head_dropout_or_bn_model_rejected"] = True
    checks["source_bytes_still_authenticated"] = bridge.authenticate_source(SOURCE, auth_receipt=auth_receipt) == authenticated.source_bytes
    checks["receipt_discloses_extraction_not_native_main"] = (
        authenticated.provenance["native_main_executed"] is False
        and authenticated.provenance["function_bodies_changed"] is False
        and authenticated.provenance["cuda_parity_verified"] is False)
    checks["bridge_and_runtime_identity_recorded"] = (
        authenticated.provenance.get("bridge_sha256") == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
        and authenticated.provenance.get("runtime") == {
            "python": sys.version.split()[0], "torch": torch.__version__, "torchvision": torchvision.__version__})
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    checks["cpu_fixture_within_2GiB"] = peak <= 2 * 1024**3
    signal.alarm(0)
    result = dict(status="PASS_ENGINEERING_DIAGNOSTIC_ONLY" if all(checks.values()) else "FAIL",
                  checks=checks, elapsed_seconds=time.monotonic() - started, peak_rss_bytes=peak,
                  implementation_sha256=hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
                  test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_authentication_receipt_sha256=(hashlib.sha256(Path(auth_receipt).read_bytes()).hexdigest()
                                                       if auth_receipt is not None else None),
                  torch=torch.__version__, torchvision=torchvision.__version__,
                  fixture="random full ResNet-50 seed 71; fc.weight *= 0.001, fc.bias=0 except bias[0]=0.2; 1000 classes, batch 1, RGB 224x224, CPU; no checkpoint or labels",
                  results=bridge_results, provenance=authenticated.provenance,
                  benchmark_executed=False, accuracy_measured=False)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not all(checks.values()):
        raise AssertionError("Native conformance failed: " + str([k for k, v in checks.items() if not v]))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--native-stage":
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--native-stage", action="store_true", required=True)
        parser.add_argument("--auth-receipt", type=Path)
        args = parser.parse_args()
        run_native_stage(auth_receipt=args.auth_receipt)
    else:
        unittest.main()
