from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments/kbound/wilds/run_imagenetr_kbound.py"
LOCK = ROOT / "research_lock/IMAGENETR_DIVERSE_PANEL_PROTOCOL_D_v1.yaml"
KBTRAIN = ROOT / "docs/research/kbound/scripts/kbtrain.sh"
WILDS_PYTHON = Path.home() / ".venv_wilds/bin/python"
SIZE_DIVERSE_BACKBONES = (
    "resnet101",
    "resnet152",
    "resnext101_32x8d",
    "efficientnet_b0",
    "efficientnet_b3",
    "convnext_tiny",
    "convnext_base",
    "vit_b_16",
    "swin_t",
    "swin_b",
)


def test_protocol_d_lock_registers_independent_backbone_panel() -> None:
    text = LOCK.read_text()

    assert "PROTOCOL D" in text
    assert "ImageNet-R" in text
    assert "diverse_backbones" in text
    assert "f0_backbone: resnet50" in text
    assert "candidate_backbones:" in text
    assert "size_diverse" in text
    assert "expected_candidate_records: 480" in text
    assert "imagenetr_protocol_d_size_diverse_panel_v2" in text
    for name in SIZE_DIVERSE_BACKBONES:
        assert f"- {name}" in text
    assert "success_criteria_stated_in_advance" in text
    assert "forbidden" in text


def test_imagenetr_runner_dry_run_exposes_protocol_d_without_loading_models() -> None:
    proc = subprocess.run(
        [
            str(WILDS_PYTHON),
            str(RUNNER),
            "--panel",
            "diverse_backbones",
            "--dry-run",
            "--seeds",
            "0",
            "--compositions",
            "iid",
            "--batch-regimes",
            "tiny",
            "--aggressiveness",
            "mild",
            "--n-eval",
            "32",
            "--run-name",
            "imagenetr_protocol_d_test",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "DRY RUN" in proc.stdout
    assert "--panel diverse_backbones" in proc.stdout
    assert f"candidate_backbones={','.join(SIZE_DIVERSE_BACKBONES)}" in proc.stdout
    assert "conditions=1" in proc.stdout
    assert "records=10" in proc.stdout
    assert "frozen_eval_batch=32" in proc.stdout
    assert "load_models=False" in proc.stdout


def test_protocol_d_runner_uses_mps_tractable_lazy_candidate_loading() -> None:
    text = RUNNER.read_text()

    assert "frozen_eval_batch" in text
    assert "candidate_specs.append((name, cand, desc))" not in text
    assert 'make_masked_backbone(name, sel, torch.device("cpu"))' not in text
    assert "load candidate lazily" in text


def test_kbtrain_exposes_imagenetr_protocol_d_commands() -> None:
    text = KBTRAIN.read_text()

    assert "imagenetr-d)" in text
    assert "imagenetr-d-dry-run)" in text
    assert "--panel diverse_backbones" in text
    assert "imagenetr_protocol_d_size_diverse_panel_v2" in text
