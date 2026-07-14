import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/research/kbound/paper/generated/kbound_result_manifest.json"
PAPER = ROOT / "docs/research/kbound/kbound_short.tex"


def load(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_every_canonical_local_source_exists():
    manifest = load(MANIFEST)
    paths = []
    for track in manifest["tracks"].values():
        for key in ("source", "multiseed_source"):
            if track.get(key):
                paths.append(track[key])
    paths.extend(
        [
            manifest["headtohead"]["source"],
            manifest["sensitivity_ablations"]["artifact"],
            manifest["sensitivity_ablations"]["script"],
            manifest["decision_gate_baselines"]["archived_artifact"],
            manifest["decision_gate_baselines"]["exact_rank_artifact"],
            manifest["decision_gate_baselines"]["script"],
            manifest["decision_gate_baselines"]["input"],
            manifest["runtime_profile"]["artifact"],
            manifest["runtime_profile"]["script"],
        ]
    )
    missing = [path for path in paths if not (ROOT / path).exists()]
    assert not missing, missing

    for track in manifest["tracks"].values():
        if track.get("source_sha256"):
            assert sha256(ROOT / track["source"]) == track["source_sha256"]
    gates = manifest["decision_gate_baselines"]
    assert sha256(ROOT / gates["archived_artifact"]) == gates["archived_artifact_sha256"]
    assert sha256(ROOT / gates["exact_rank_artifact"]) == gates["exact_rank_artifact_sha256"]
    runtime = manifest["runtime_profile"]
    assert sha256(ROOT / runtime["artifact"]) == runtime["artifact_sha256"]


def test_camelyon_multiseed_artifacts_match_paper_rows():
    result_dir = ROOT / "experiments/kbound/results/camelyon17_multiseed_v1"
    expected = {
        "tent": ([0.0201, 0.023], [0.138, 0.0192], [0.0201, 0.023], 0.0),
        "eata": ([0.0393, 0.0252], [0.0417, 0.0248], [0.0424, 0.024], 0.0),
        "sar": ([0.041, 0.0165], [0.0002, 0.0002], [0.0654, 0.0265], 0.1111),
    }
    for candidate, values in expected.items():
        artifact = load(result_dir / f"multiseed_camelyon17_{candidate}.json")
        assert artifact["seeds"] == [0, 1, 2, 3]
        assert artifact["conditions_per_seed"] == 9
        assert (
            artifact["regret_kga"],
            artifact["regret_adapt"],
            artifact["regret_freeze"],
            artifact["FA_u_max"],
        ) == values


def test_promoted_evidence_and_runtime_wording_is_source_backed():
    paper = PAPER.read_text()
    assert "exploratory $16$-feature" in paper
    assert "natural-shift protocols use a $16$-dim panel" not in paper
    assert "controller_cost_v1/cost_profile.json" in paper
    assert "0.343" in paper
    profile = load(ROOT / "experiments/kbound/results/controller_cost_v1/cost_profile.json")
    assert profile["scope"].startswith("controller-only")
    assert profile["rollback_checkpoint"]["sha256"]


def test_natural_point_win_locks_are_explicitly_superseded():
    office = (ROOT / "research_lock/OFFICEHOME_PROTOCOL_M_RECONCILED_v3.yaml").read_text()
    iwild = (ROOT / "research_lock/IWILDCAM_PROTOCOL_H_RECONCILED_v3.yaml").read_text()
    assert "supersedes: OFFICEHOME_PROTOCOL_M_v2" in office
    assert "verdict: no_harm_not_ci_robust_beats_both" in office
    assert "supersedes: IWILDCAM_PROTOCOL_H_v2" in iwild
    assert "verdict: no_harm_ties_freeze" in iwild
