"""Execute the supervisor without opening research data or starting training."""
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "docs/research/kbound/scripts/supervise_rxrx1_9plus.sh"


def test_missing_checkpoint_rejects_stale_overall_success(tmp_path):
    results = tmp_path / "experiments/kbound/results"
    results.mkdir(parents=True)
    marker = results / "rxrx1_protocol_c_9plus_ALL.done"
    marker.write_text("historical marker\n")
    result = subprocess.run(
        ["zsh", str(SCRIPT)], env={**os.environ,
            "KBOUND_REPO_ROOT": str(tmp_path),
            "KBOUND_EXTERNAL_ROOT": str(tmp_path / "external"),
            "RXRX1_MODEL_SEEDS": "0"},
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 1, result.stderr
    assert not marker.exists()
    assert [p.read_text() for p in results.glob("*.done.stale.*")] == ["historical marker\n"]
    log = (results / "rxrx1_protocol_c_9plus_supervisor.log").read_text()
    assert "all_done=0" in log
    assert "launching runner" not in log


def test_invalid_receipt_is_preserved_then_launch_and_verify_share_context(tmp_path):
    runner = tmp_path / "experiments/kbound/wilds/run_rxrx1_kbound.py"
    runner.parent.mkdir(parents=True)
    # Replace only the expensive training/receipt service; execute the real supervisor.
    runner.write_text('''import json, sys
from pathlib import Path
args = sys.argv[1:]
verify = "--verify-completion" in args
if verify:
    receipt = Path(args[1])
    args = args[2:]
else:
    args.remove("--resume")
    receipt = Path(args[args.index("--results-root")+1]) / args[args.index("--run-name")+1] / ".done"
with (receipt.parent / "calls.jsonl").open("a") as f:
    f.write(json.dumps({"verify": verify, "args": args}) + "\\n")
if verify:
    sys.exit(0 if receipt.read_text() == json.dumps(args) else 1)
receipt.write_text(json.dumps(args))
''')
    ckpt = tmp_path / "external/kbound_rxrx1_ckpt/rxrx1_seed:2_epoch:best_model.pth"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_text("synthetic checkpoint")
    results = tmp_path / "experiments/kbound/results"
    out = results / "rxrx1_protocol_c_9plus_modelseed2"
    out.mkdir(parents=True)
    (out / ".done").write_text("invalid historical receipt")
    result = subprocess.run(["zsh", str(SCRIPT)], env={**os.environ,
        "KBOUND_REPO_ROOT": str(tmp_path), "KBOUND_EXTERNAL_ROOT": str(tmp_path / "external"),
        "KBOUND_PYTHON": sys.executable, "RXRX1_MODEL_SEEDS": "2"},
        capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert (results / "rxrx1_protocol_c_9plus_ALL.done").exists()
    assert [p.read_text() for p in out.glob(".done.stale.*")] == ["invalid historical receipt"]
    calls = [json.loads(line) for line in (out / "calls.jsonl").read_text().splitlines()]
    assert [call["verify"] for call in calls] == [True, False, True, True]
    assert all(call["args"] == calls[0]["args"] for call in calls)
    args = calls[0]["args"]
    assert args[args.index("--model-seed")+1] == "2"
