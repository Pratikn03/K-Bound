"""EXPLICIT native runtime stage; excluded from default source-only discovery.

Run this file with --native-stage and declared POEM_PYTHON / POEM_SOURCE.
Missing prerequisites fail the required stage; they never count as a skip/pass.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class NativeCPUSeam(unittest.TestCase):
    __test__ = False  # pytest source-only discovery must not launch native runtime.
    def test_temperature_source_cdf_update_and_partial_reset_are_bound(self):
        # Catches a CUDA-only constructor, fabricated CDF, missed warmup/update,
        # or a misleading claim that native POEM.reset resets streaming state.
        root = Path(__file__).resolve().parents[1]
        script = root / "docs/research/kbound/scripts/poem_native_cpu_seam.py"
        source = Path(os.environ["POEM_SOURCE"])
        names = ["main.py", "poem.py", "protector.py", "cdf.py", "temperature_scaling.py", "sar.py", "sam.py"]
        before = {n: hashlib.sha256((source / n).read_bytes()).hexdigest() for n in names}
        with tempfile.TemporaryDirectory(prefix="poem_cpu_seam_test_") as temporary:
            receipt = Path(os.environ.get("POEM_SEAM_RECEIPT", str(Path(temporary) / "receipt.json")))
            result = subprocess.run(
                [os.environ["POEM_PYTHON"], str(script), "--native-stage",
                 "--poem-source", str(source), "--receipt", str(receipt)],
                cwd=root, capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stderr[-4000:])
            record = json.loads(receipt.read_text())
            self.assertEqual(record["status"], "PASS_ENGINEERING_DIAGNOSTIC_ONLY")
            self.assertEqual(record["source_sha256"], before)
            self.assertEqual(record["execution_stage"], "explicit_poem_native_cpu_conformance_required_for_task3")
            self.assertEqual(record["driver_semantics_binding"]["sha256"], before["main.py"])
            helper = root / "docs/research/kbound/scripts/poem_dependency_bootstrap.py"
            self.assertEqual(record["local_dependencies"], [{
                "path": "docs/research/kbound/scripts/poem_dependency_bootstrap.py",
                "sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
                "source_snapshot_required": True,
            }])
            self.assertEqual(record["samples"], 104)
            self.assertEqual(record["device"], "cpu")
            self.assertFalse(record["pretrained"])
            self.assertFalse(record["native_main_executed"])
            self.assertEqual(record["cdf_source"], "native_ecdf_of_fixture_model_source_entropies")
            self.assertTrue(all(record["checks"].values()), record["checks"])
            self.assertIn("observed_batch_sha256", record, "Actual upstream input batches must be observed")
            self.assertEqual(record["observed_batch_sha256"], record["expected_batch_sha256"])
            self.assertTrue(record["native_run_eval_mode"])
            self.assertEqual(record["native_reset"]["sample_count_after"], 104)
            self.assertEqual(record["native_reset"]["protector_samples_after"], 104)
            self.assertGreater(record["parameter_tensors_changed"], 0)
            self.assertLess(record["elapsed_seconds"], 120)
        self.assertEqual(before, {n: hashlib.sha256((source / n).read_bytes()).hexdigest() for n in names})


def load_tests(loader, tests, pattern):
    """Default unittest discovery excludes the separately required native stage."""
    return unittest.TestSuite()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-stage", action="store_true", required=True)
    parser.parse_args()
    for required in ("POEM_PYTHON", "POEM_SOURCE"):
        if not os.environ.get(required):
            parser.error(f"{required} must be explicitly declared for the required native stage")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(NativeCPUSeam)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
