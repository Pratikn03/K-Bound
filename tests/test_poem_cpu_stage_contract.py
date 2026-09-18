"""Source-only CLI contracts: stdlib, no PyTorch or external checkout required."""
import os
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class NativeStageDeclaration(unittest.TestCase):
    def test_default_discovery_does_not_execute_native_runtime(self):
        path = Path(__file__).with_name("test_poem_native_cpu_seam.py")
        spec = importlib.util.spec_from_file_location("poem_native_stage_discovery_contract", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(unittest.defaultTestLoader.loadTestsFromModule(module).countTestCases(), 0)

    def test_changed_driver_reference_is_rejected_before_receipt_or_runtime_import(self):
        root = Path(__file__).resolve().parents[1]
        script = root / "docs/research/kbound/scripts/poem_native_cpu_seam.py"
        with tempfile.TemporaryDirectory(prefix="poem_stage_contract_") as directory:
            source = Path(directory)
            (source / ".git").mkdir()  # Synthetic existence fixture, not a Git repository.
            (source / "main.py").write_text("raise RuntimeError('must not execute')\n")
            receipt = source / "receipt.json"
            result = subprocess.run([sys.executable, str(script), "--native-stage", "--poem-source", directory,
                                     "--receipt", str(receipt)], env=dict(os.environ, POEM_PYTHON=sys.executable),
                                    text=True, capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("SHA-256 mismatch", result.stderr)
            self.assertFalse(receipt.exists())

    def test_missing_declared_runtime_fails_before_creating_receipt(self):
        root = Path(__file__).resolve().parents[1]
        script = root / "docs/research/kbound/scripts/poem_native_cpu_seam.py"
        with tempfile.TemporaryDirectory(prefix="poem_stage_contract_") as directory:
            receipt = Path(directory) / "receipt.json"
            env = dict(os.environ)
            env.pop("POEM_PYTHON", None)
            result = subprocess.run([sys.executable, str(script), "--native-stage",
                                     "--poem-source", directory, "--receipt", str(receipt)],
                                    env=env, text=True, capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("POEM_PYTHON must explicitly name", result.stderr)
            self.assertFalse(receipt.exists())

    def test_declared_runtime_mismatch_fails_before_creating_receipt(self):
        root = Path(__file__).resolve().parents[1]
        script = root / "docs/research/kbound/scripts/poem_native_cpu_seam.py"
        with tempfile.TemporaryDirectory(prefix="poem_stage_contract_") as directory:
            receipt = Path(directory) / "receipt.json"
            env = dict(os.environ, POEM_PYTHON=str(Path(directory) / "not-the-running-interpreter"))
            result = subprocess.run([sys.executable, str(script), "--native-stage",
                                     "--poem-source", directory, "--receipt", str(receipt)],
                                    env=env, text=True, capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("POEM_PYTHON does not match", result.stderr)
            self.assertFalse(receipt.exists())


if __name__ == "__main__":
    unittest.main()
