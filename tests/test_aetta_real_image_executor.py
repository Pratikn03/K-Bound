"""Read-only input/output contracts for the separately declared AETTA smoke."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/research/kbound/scripts/run_aetta_real_image_smoke.py"


def executor():
    if not SCRIPT.is_file():
        raise AssertionError("The bounded native AETTA real-image executor is missing")
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        spec = importlib.util.spec_from_file_location("tested_aetta_image_runner", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPT.parent))


def synthetic_panel():
    def row(i, source):
        name = f"n00000001/ILSVRC2012_val_{i:08d}.JPEG"
        return dict(relative_path=name if source else "gaussian_noise/5/" + name,
                    sample_id=f"ILSVRC2012_val_{i:08d}", sha256="a" * 64,
                    size_bytes=100, width=224, height=224, decoded_rgb=True)
    return dict(schema="kbound-real-image-smoke-panel-v1", scope="ENGINEERING_SMOKE_NOT_BENCHMARK",
                outcomes_read=False, upstream_full_dataset_protocol=False, official_image_bytes_authenticated=False,
                seed=0, batch_size=4, condition="gaussian_noise/5", clean_root="/clean", corruption_root="/corrupt",
                source=[row(i, True) for i in range(1, 33)], target=[row(i, False) for i in range(33, 137)])


class AETTAExecutorContracts(unittest.TestCase):
    def test_import_does_not_start_native_runtime_or_import_poem(self):
        before = set(sys.modules)
        executor()
        imported = set(sys.modules) - before
        self.assertNotIn("torch", imported)
        self.assertFalse(any("poem" in name for name in imported))

    def test_missing_or_changed_panel_cannot_select_new_images(self):
        module = executor()
        with tempfile.TemporaryDirectory(prefix="aetta_image_contract_") as tmp:
            path = Path(tmp).resolve() / "manifest.json"
            with self.assertRaises((FileNotFoundError, ValueError)):
                module.read_panel(path)
            path.write_text(json.dumps(synthetic_panel()))
            with self.assertRaisesRegex(ValueError, "digest"):
                module.read_panel(path)

    def test_panel_scope_count_seed_and_ids_are_not_tunable(self):
        module = executor()
        panel = synthetic_panel()
        module.validate_panel(panel)
        for field, value in (("seed", 1), ("batch_size", 1), ("scope", "BENCHMARK"),
                             ("outcomes_read", True), ("condition", "shot_noise/5")):
            changed = copy.deepcopy(panel)
            changed[field] = value
            with self.assertRaises(ValueError):
                module.validate_panel(changed)
        changed = copy.deepcopy(panel)
        changed["target"].pop()
        with self.assertRaises(ValueError):
            module.validate_panel(changed)
        changed = copy.deepcopy(panel)
        changed["target"][0] = copy.deepcopy(changed["target"][1])
        with self.assertRaises(ValueError):
            module.validate_panel(changed)

    def test_label_or_outcome_fields_cannot_enter_runtime_rows(self):
        module = executor()
        for field in ("label", "gt", "accuracy", "benefit"):
            panel = synthetic_panel()
            panel["target"][0][field] = 4
            with self.assertRaisesRegex(ValueError, "fields"):
                module.validate_panel(panel)

    def test_checkpoint_missing_wrong_size_wrong_hash_and_symlink_fail(self):
        module = executor()
        with tempfile.TemporaryDirectory(prefix="aetta_weight_contract_") as tmp:
            root = Path(tmp).resolve()
            path = root / "wrong.pth"
            with self.assertRaises((ValueError, FileNotFoundError)):
                module.read_checkpoint(path)
            path.write_bytes(b"not a checkpoint")
            with self.assertRaisesRegex(ValueError, "size"):
                module.read_checkpoint(path)
            with mock.patch.object(module, "CHECKPOINT_SIZE", path.stat().st_size):
                with self.assertRaisesRegex(ValueError, "digest"):
                    module.read_checkpoint(path)
            link = root / "link.pth"
            link.symlink_to(path)
            with self.assertRaises((ValueError, OSError)):
                module.read_checkpoint(link)

    def test_fresh_artifact_writer_cannot_overwrite_or_escape(self):
        module = executor()
        with tempfile.TemporaryDirectory(prefix="aetta_output_contract_") as tmp:
            root = Path(tmp).resolve()
            fd = module.panel_io._open_directory_chain(root)
            try:
                self.assertEqual(module.write_artifact(fd, "receipt.json", b"first"), hashlib.sha256(b"first").hexdigest())
                with self.assertRaises(FileExistsError):
                    module.write_artifact(fd, "receipt.json", b"replacement")
                with self.assertRaises(ValueError):
                    module.write_artifact(fd, "../escaped.json", b"bad")
                self.assertEqual(root.joinpath("receipt.json").read_bytes(), b"first")
            finally:
                os.close(fd)

    def test_outputs_cannot_enter_datasets_or_upstream_source(self):
        module = executor()
        panel = synthetic_panel()
        for output in ("/clean/out", "/corrupt/out", "/source/out"):
            with self.assertRaisesRegex(ValueError, "outside"):
                module.validate_output(Path(output), panel, Path("/source"))

    def test_frozen_model_change_missing_predictions_and_reordered_ids_fail(self):
        module = executor()
        panel = synthetic_panel()
        expected = [row["sample_id"] for row in panel["target"]]
        module.verify_execution(expected, expected, 26, "a" * 64, "a" * 64, True)
        for seen, count, state_after, forward_ok in ((expected[:-1], 26, "a" * 64, True),
                                                   (list(reversed(expected)), 26, "a" * 64, True),
                                                   (expected, 25, "a" * 64, True),
                                                   (expected, 26, "b" * 64, True),
                                                   (expected, 26, "a" * 64, False)):
            with self.assertRaises(ValueError):
                module.verify_execution(expected, seen, count, "a" * 64, state_after, forward_ok)

    def test_replaced_output_path_does_not_redirect_receipt_to_dataset(self):
        module = executor()
        with tempfile.TemporaryDirectory(prefix="aetta_output_swap_") as tmp:
            root = Path(tmp).resolve()
            clean, corrupt = root / "clean", root / "corrupt"
            clean.mkdir()
            corrupt.mkdir()
            panel = synthetic_panel()
            panel.update(clean_root=str(clean), corruption_root=str(corrupt))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(panel))
            output, retained = root / "output", root / "retained"

            def only_swap_output(args, locked, record):
                output.rename(retained)
                output.symlink_to(clean, target_is_directory=True)
                record["status"] = "PASS_REAL_IMAGE_ENGINEERING_SMOKE_ONLY"

            argv = [str(SCRIPT), "--manifest", str(manifest), "--checkpoint", str(root / "unused.pth"),
                    "--aetta-source", str(root / "source"), "--auth-receipt", str(root / "auth.json"),
                    "--output", str(output)]
            with mock.patch.object(module, "PANEL_SHA256", hashlib.sha256(manifest.read_bytes()).hexdigest()), \
                    mock.patch.object(module, "run", only_swap_output), \
                    mock.patch.dict(os.environ, AETTA_PYTHON=sys.executable), mock.patch.object(sys, "argv", argv):
                self.assertEqual(module.main(), 2)
            self.assertFalse(clean.joinpath("receipt.json").exists())
            self.assertEqual(json.loads(retained.joinpath("receipt.json").read_text())["status"], "FAILED_REAL_IMAGE_SMOKE")


if __name__ == "__main__":
    unittest.main()
