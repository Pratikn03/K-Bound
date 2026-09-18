from __future__ import annotations

import ast
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BASELINE_ADAPTER = ROOT / "docs/research/kbound/scripts/baseline_decisions_adapter.py"
OFFICIAL_HEADTOHEAD = ROOT / "docs/research/kbound/scripts/official_baselines_headtohead.py"
MULTISEED_AGGREGATE = ROOT / "docs/research/kbound/scripts/multiseed_aggregate.py"
TABLE_GENERATOR = ROOT / "docs/research/kbound/scripts/make_tables.py"
RELEASE_CANDIDATE = ROOT / "docs/research/kbound/runbooks/release_candidate.sh"


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"{path.stem}_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("method", "row"),
    [
        ("aetta", {"condition": "c", "est_acc_adapted": "nan", "est_acc_frozen": 0.2}),
        ("aetta", {"condition": "c", "est_acc_adapted": float("inf"), "est_acc_frozen": 0.2}),
        ("poem", {"condition": "c", "updated": "definitely"}),
    ],
)
def test_baseline_adapter_never_maps_malformed_evidence_to_an_action(method: str, row: dict) -> None:
    module = _load_module(BASELINE_ADAPTER)
    assert module.to_decision(method, row) == ("c", None)


def test_baseline_adapter_rejects_partial_output_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module(BASELINE_ADAPTER)
    source = tmp_path / "input.json"
    conditions = tmp_path / "conditions.json"
    output = tmp_path / "decisions.json"
    source.write_text('{"records":[{"condition":"a","updated":true}]}')
    conditions.write_text('{"records":[{"condition":"a"},{"condition":"b"}]}')
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            str(BASELINE_ADAPTER),
            "--method",
            "poem",
            "--input",
            str(source),
            "--conditions-from",
            str(conditions),
            "--out",
            str(output),
        ],
    )
    with pytest.raises(SystemExit):
        module.main()
    assert not output.exists()


def test_baseline_adapter_uses_repository_canonical_conditions() -> None:
    module = _load_module(BASELINE_ADAPTER)
    assert Path(module.CANON).resolve() == (ROOT / "experiments/kbound/results/per_condition_cifar10c_tent_seed0.json")


def test_official_headtohead_rejects_non_boolean_official_flag(tmp_path: Path) -> None:
    module = _load_module(OFFICIAL_HEADTOHEAD)
    payload = tmp_path / "decisions.json"
    payload.write_text('{"decisions":{"cell":"adapt"},"official_label_allowed":"false"}')
    with pytest.raises(SystemExit, match="boolean"):
        module.load_external(str(payload), ["cell"])


def test_official_headtohead_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    module = _load_module(OFFICIAL_HEADTOHEAD)
    payload = tmp_path / "decisions.json"
    payload.write_text('{"decisions":{"cell":"adapt","cell":"freeze"}}')
    with pytest.raises(SystemExit, match="duplicate"):
        module.load_external(str(payload), ["cell"])


def test_official_headtohead_rejects_self_attested_official_wrapper(tmp_path: Path) -> None:
    module = _load_module(OFFICIAL_HEADTOHEAD)
    payload = tmp_path / "decisions.json"
    payload.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "decisions": {"cell": "adapt"},
                "official_label_allowed": True,
                "label": "official_implementation_under_protocol_adapter",
            }
        )
    )
    with pytest.raises(SystemExit, match="audit|provenance"):
        module.load_external(str(payload), ["cell"])


@pytest.mark.parametrize("unsafe", [None, float("nan"), float("inf"), "0.0"])
def test_multiseed_summary_rejects_missing_or_malformed_false_adapt(tmp_path: Path, unsafe: object) -> None:
    module = _load_module(MULTISEED_AGGREGATE)
    payload = {
        "regret_kga": 0.01,
        "regret_adapt": 0.03,
        "regret_freeze": 0.04,
    }
    if unsafe is not None:
        payload["false_adapt"] = unsafe
    path = tmp_path / "seed.json"
    import json

    path.write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match="false[_-]adapt|finite|numeric"):
        module.load_seed(str(path))


def test_multiseed_overlap_does_not_certify_no_harm():
    module = _load_module(MULTISEED_AGGREGATE)
    seeds = [dict(kga=value, adapt=0.3, freeze=0.1, fau=0.0, src=f"synthetic-{i}")
             for i, value in enumerate([0.05, 0.08, 0.10, 0.12, 0.15])]
    result = module.summarize("synthetic", seeds)
    low, high = result["gap_vs_better"]["ci95"]
    assert low < 0 < high
    assert result["verdict"] == "inconclusive vs better; nominal improvement vs worse"


@pytest.mark.parametrize(
    ("container", "field"),
    [
        ("policy_mean_regret", "always_adapt"),
        ("policy_false_adapt_rate", "kga"),
        ("policy_decisive_rate", "kga"),
    ],
)
def test_paper_table_fallback_rejects_missing_headtohead_metrics(container: str, field: str) -> None:
    tree = ast.parse(TABLE_GENERATOR.read_text(), filename=str(TABLE_GENERATOR))
    wanted = {"_headtohead"}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace = {
        "SRC": "manifest",
        "H2H_DEFAULT": "fallback",
        "math": math,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(TABLE_GENERATOR), "exec"), namespace)
    fallback = {
        "headtohead": {"VERDICT": "WIN"},
        "policy_mean_regret": {
            "kga": 0.01,
            "always_adapt": 0.02,
            "always_freeze": 0.03,
            "poem": 0.04,
            "aetta": 0.05,
        },
        "policy_false_adapt_rate": {"kga": 0.0},
        "policy_decisive_rate": {"kga": 0.5},
    }
    del fallback[container][field]
    namespace["_load_json"] = lambda path: {} if path == "manifest" else fallback
    with pytest.raises(ValueError, match="missing|invalid"):
        namespace["_headtohead"]()


def _promotion_summary() -> dict[str, object]:
    return {
        "mode": "label_free",
        "alpha": 0.1,
        "regret_kga": 0.01,
        "regret_always_adapt": 0.02,
        "regret_always_freeze": 0.03,
        "integrity_failures": [],
        "frozen_estimator_verified": True,
        "held_out_natural_datasets": 2,
        "frozen_before_scoring": True,
        "independent_splits": 3,
        "false_adapt_rate": 0.0,
        "coverage": 0.5,
        "confidence_intervals_complete": True,
        "strong_baselines_complete": True,
        "required_tracks_complete": True,
    }


@pytest.mark.parametrize(
    ("field", "malformed"),
    [
        ("alpha", float("inf")),
        ("false_adapt_rate", -0.1),
        ("coverage", 1.1),
        ("regret_kga", "0.01"),
        ("frozen_estimator_verified", "false"),
        ("frozen_before_scoring", 1),
        ("held_out_natural_datasets", "2"),
        ("independent_splits", "3"),
        ("integrity_failures", "none"),
        ("confidence_intervals_complete", "yes"),
        ("strong_baselines_complete", 1),
        ("required_tracks_complete", "true"),
    ],
)
def test_promotion_assessor_never_promotes_malformed_metadata(field: str, malformed: object) -> None:
    from kga.integrations.claims import assess_promotion

    summary = _promotion_summary()
    assert assess_promotion(summary)["eligible"] is True
    summary[field] = malformed
    assert assess_promotion(summary)["eligible"] is False


def _elara_estimator(*, residual_count: int = 20):
    from kga.benefit import FrozenLinearBenefitEstimator

    names = (
        "ks_mean",
        "ks_max",
        "disagree",
        "entropy_shift",
        "conf_shift",
        "ess_frac",
        "best_val_auc",
        "val_gap",
        "val_disagreement",
        "n_experts",
    )
    return FrozenLinearBenefitEstimator(
        feature_names=names,
        weights=np.zeros(len(names)),
        intercept=0.2,
        feature_center=np.zeros(len(names)),
        feature_scale=np.ones(len(names)),
        residuals=np.zeros(residual_count),
        evidence_schema_version="elara-evidence/1",
        protocol_sha256="a" * 64,
        fit_unit="fit",
        calibration_unit="calibration",
    )


def _install_fake_elara_router(monkeypatch: pytest.MonkeyPatch, module) -> None:
    class Policy:
        pass

    def reliability_features(_scores, _labels):
        return {"val_auc": [0.8, 0.7], "best_auc": 0.8, "gap": 0.1, "disagreement": 0.2}

    def route(_s_val, _y_val, s_test, _policy, *, action):
        return np.asarray(s_test[:, 1]), action

    monkeypatch.setattr(module, "_load_router_api", lambda: (Policy, reliability_features, route))


def _elara_inputs() -> dict[str, object]:
    return {
        "s_val": np.array([[0.1, 0.2], [0.8, 0.7], [0.2, 0.4], [0.9, 0.6]]),
        "y_val": np.array([0, 1, 0, 1]),
        "s_test": np.array([[0.2, 0.3], [0.7, 0.6], [0.4, 0.5], [0.8, 0.7]]),
        "mode": "label_free",
    }


def test_elara_label_free_requires_externally_authorized_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kga.integrations import elara

    _install_fake_elara_router(monkeypatch, elara)
    with pytest.raises(ValueError, match="protocol"):
        elara.ELARAKGAGuard().decide(**_elara_inputs(), estimator=_elara_estimator())


def test_elara_abstain_record_is_strict_json_and_retains_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kga.integrations import elara
    from kga.policy import Decision

    _install_fake_elara_router(monkeypatch, elara)
    with pytest.warns(UserWarning, match="no finite radius"):
        result = elara.ELARAKGAGuard().decide(
            **_elara_inputs(),
            estimator=_elara_estimator(residual_count=1),
            protocol_sha256="a" * 64,
        )
    assert result.decision is Decision.ABSTAIN
    assert result.deployed_action == "retain_frozen"
    assert np.array_equal(result.deployed_scores, result.frozen_scores)
    json.dumps(result.to_record(), allow_nan=False)


def test_release_deep_local_preflight_fails_when_required_datasets_are_absent(tmp_path: Path) -> None:
    canonical_receipts = (
        ROOT / "docs/research/kbound/audits/python_environment_2026_09_02.json",
        ROOT / "docs/research/kbound/audits/release_toolchain_2026_09_02.json",
    )
    # This isolated fixture tests failure on missing datasets, not whether this
    # checkout has historical machine receipts. Preserve absence as well as bytes.
    receipt_bytes_before = {
        path: path.read_bytes() if path.exists() else None for path in canonical_receipts
    }
    repo = tmp_path / "repo"
    runbook = repo / "docs/research/kbound/runbooks/release_candidate.sh"
    runbook.parent.mkdir(parents=True)
    runbook.write_text(RELEASE_CANDIDATE.read_text(encoding="utf-8"), encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname = 'preflight-fixture'\n", encoding="utf-8")
    package = repo / "docs/research/kbound/kbound_repro"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "runtime.py").write_text(
        "def describe_runtime():\n"
        "    return {'python': '3.12.13', 'platform': 'fixture', 'numpy': '2.4.4', "
        "'torch': '2.5.1', 'torchvision': '0.20.1', 'sklearn': '1.8.0'}\n",
        encoding="utf-8",
    )
    (package / "paths.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "def imagenetr_root():\n    return Path(os.environ['KBOUND_IMAGENETR_ROOT'])\n"
        "def pacs_root():\n    return Path(os.environ['KBOUND_PACS_ROOT'])\n",
        encoding="utf-8",
    )
    missing_imagenetr = tmp_path / "missing-imagenetr"
    missing_pacs = tmp_path / "missing-pacs"
    verified_python = tmp_path / "verified-python"
    verified_python.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "from pathlib import Path\n"
        "script = sys.argv[1] if len(sys.argv) > 1 else ''\n"
        "if script.endswith(('verify_python_environment.py', 'verify_release_toolchain.py')):\n"
        "    if '--output' in sys.argv:\n"
        "        output = Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "        output.parent.mkdir(parents=True, exist_ok=True)\n"
        "        output.write_text('{}\\n', encoding='utf-8')\n"
        "    if '--resolved-tools-output' in sys.argv:\n"
        "        resolved = Path(sys.argv[sys.argv.index('--resolved-tools-output') + 1])\n"
        "        names = ('LATEXMK','LATEXPAND','PANDOC','PDFLATEX','PDFINFO','PDFTOPPM','PDFTOTEXT','PDFDETACH','PERL','SOFFICE')\n"
        "        resolved.write_text(''.join(f'KBOUND_TOOL_{name}\\t{Path(sys.argv[0]).resolve()}\\n' for name in names), encoding='utf-8')\n"
        "    raise SystemExit(0)\n"
        "os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n",
        encoding="utf-8",
    )
    verified_python.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "KBOUND_PYTHON": str(verified_python),
            "KBOUND_IMAGENETR_ROOT": str(missing_imagenetr),
            "KBOUND_PACS_ROOT": str(missing_pacs),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )

    proc = subprocess.run(
        ["bash", str(runbook), "deep-local-preflight"],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "KBOUND_IMAGENETR_ROOT" in proc.stdout + proc.stderr
    assert "KBOUND_PACS_ROOT" in proc.stdout + proc.stderr
    assert {
        path: path.read_bytes() if path.exists() else None for path in canonical_receipts
    } == receipt_bytes_before
