"""The ImageNet-C POEM runbook must select the pinned official checkout."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def test_poem_imagenetc_runbook_uses_pinned_official_source() -> None:
    repo = Path(__file__).resolve().parents[1]
    runbook = (repo / "docs/research/kbound/runbooks/run_poem_imagenetc.sh").read_text(encoding="utf-8")
    task3_runbook = (repo / "docs/research/kbound/runbooks/run_task3_natural.sh").read_text(encoding="utf-8")
    native_runner = (repo / "docs/research/kbound/scripts/run_official_native.py").read_text(encoding="utf-8")

    assert 'P="${POEM_SOURCE:-$R/external/poem_official}"' in runbook
    assert 'P="$R/external/poem"' not in runbook
    assert "external/poem/main.py line 40 import guarded" not in runbook
    assert 'local poem_source="${POEM_SOURCE:-$ROOT/external/poem_official}"' in task3_runbook
    assert 'Path("external/poem_official")' in native_runner
    assert 'Path("external/poem")' not in native_runner
    assert 'POEM_OUTPUT' in runbook
    assert 'poem_imagenetc_native' in runbook
    assert 'output directory must be fresh' in runbook


def test_poem_imagenetc_runbook_binds_clean_and_corrupted_roots_separately() -> None:
    repo = Path(__file__).resolve().parents[1]
    runbook = (repo / "docs/research/kbound/runbooks/run_poem_imagenetc.sh").read_text(encoding="utf-8")

    # POEM's loader derives its holdout split from clean ImageNet and indexes
    # the corresponding corrupted ImageNet-C images.  Passing ImageNet-C as
    # both roots silently fails at <imagenetc>/val and is not a valid run.
    assert 'IMAGENET_ROOT' in runbook
    assert 'CLEAN="${IMAGENET_ROOT:-' in runbook
    assert '--data "$CLEAN"' in runbook
    assert '--data_corruption "$IC"' in runbook
    assert 'clean ImageNet root' in runbook


def test_poem_imagenetc_runbook_rejects_untracked_model_shims() -> None:
    repo = Path(__file__).resolve().parents[1]
    runbook = (repo / "docs/research/kbound/runbooks/run_poem_imagenetc.sh").read_text(encoding="utf-8")

    # The upstream POEM commit imports models.Res unconditionally, but does
    # not track that package.  An ignored local shim is useful for --help, yet
    # it must never be silently promoted as a native official receipt.
    assert 'poem_dependency_bootstrap.py' in runbook
    assert '--poem-source "$P" --sar-source "$SAR" --' in runbook
    assert 'ignored local shim' in runbook.lower()


def test_poem_imagenetc_runbook_is_not_mac_only() -> None:
    repo = Path(__file__).resolve().parents[1]
    runbook = (repo / "docs/research/kbound/runbooks/run_poem_imagenetc.sh").read_text(encoding="utf-8")

    # Native evidence is expected on a Linux CUDA host as well as on macOS.
    # The optional sleep-prevention wrapper must therefore not make caffeinate
    # a hard dependency of the benchmark command.
    assert 'command -v caffeinate' in runbook
    assert 'run_native_cmd' in runbook
    assert 'caffeinate -is "$PYBIN"' not in runbook


@pytest.fixture
def synthetic_runbook(tmp_path):
    """Run the real shell/bootstrap with small authenticated stdlib-only sources.

    The driver retains upstream's normal-mode corruption loop. Only model and
    runtime preflights and ImageNet class counts are replaced at their external
    command boundaries; no real model dependency is imported by this fixture.
    """
    repo = Path(__file__).resolve().parents[1]
    poem = tmp_path / "poem"
    sar = tmp_path / "sar"
    poem.mkdir()
    (sar / "models").mkdir(parents=True)
    (poem / "main.py").write_text(textwrap.dedent("""\
        import argparse
        import json
        import os
        from pathlib import Path
        import models.Res

        def get_args():
            parser = argparse.ArgumentParser()
            parser.add_argument('--method')
            parser.add_argument('--model')
            parser.add_argument('--exp_type', default='normal')
            parser.add_argument('--data')
            parser.add_argument('--data_corruption')
            parser.add_argument('--corruption', default='gaussian_noise')
            parser.add_argument('--level')
            parser.add_argument('--seed')
            parser.add_argument('--test_batch_size')
            parser.add_argument('--workers')
            parser.add_argument('--output')
            return parser.parse_args()

        if __name__ == '__main__':
            args = get_args()
            common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
            if args.exp_type == 'normal':
                Path(args.output).mkdir(parents=True, exist_ok=True)
                for corruption in common_corruptions:
                    args.corruption = corruption
                    with open(os.environ['SYNTHETIC_EVENTS'], 'a') as stream:
                        stream.write(json.dumps({'method': args.method,
                            'corruption': args.corruption, 'output': args.output}) + '\\n')
                    if (os.environ.get('SYNTHETIC_FAILURE') == 'run'
                            and args.method == 'poem' and corruption == 'shot_noise'):
                        raise SystemExit(23)
        """), encoding="utf-8")
    (sar / "models/Res.py").write_text(textwrap.dedent("""\
        import os
        if os.environ.get('SYNTHETIC_FAILURE') == 'dependency':
            raise RuntimeError('synthetic dependency import failed')
        MODEL_ID = 'synthetic-no-model'
        """), encoding="utf-8")

    pins = {}
    for name, source, filename in (("POEM", poem, "main.py"), ("SAR", sar, "models/Res.py")):
        for args in (
            ["init", "-q"], ["add", filename],
            ["-c", "user.name=Synthetic Test", "-c", "user.email=synthetic@example.invalid",
             "-c", "commit.gpgsign=false", "commit", "-qm", "synthetic fixture"],
        ):
            subprocess.run(
                ["git", "-C", str(source), "-c", "core.hooksPath=/dev/null", *args],
                check=True, capture_output=True,
            )
        pins[f"SYNTHETIC_{name}_COMMIT"] = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        pins[f"SYNTHETIC_{name}_HASH"] = hashlib.sha256((source / filename).read_bytes()).hexdigest()

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    def executable(name, body):
        path = bin_dir / name
        path.write_text(f"#!{sys.executable}\n" + textwrap.dedent(body), encoding="utf-8")
        path.chmod(0o755)
        return path

    python = executable("synthetic-python", """\
        import importlib.util
        import json
        import os
        import sys

        args = sys.argv[1:]
        if args[:1] == ['-c']:
            command = args[1]
            if command.startswith('import torch; print('):
                print('cpu' if os.environ.get('SYNTHETIC_FAILURE') == 'backend' else 'cuda')
            elif command == "import timm; timm.create_model('resnet50_gn', pretrained=True)":
                if os.environ.get('SYNTHETIC_FAILURE') == 'model':
                    raise SystemExit(19)
            else:
                raise AssertionError('unexpected runtime command: ' + command)
            raise SystemExit(0)
        if args[:1] == ['-u']:
            args = args[1:]
        assert args[0] == os.environ['SYNTHETIC_BOOTSTRAP']
        with open(os.environ['SYNTHETIC_CALLS'], 'a') as stream:
            stream.write(json.dumps(args[1:]) + '\\n')
        spec = importlib.util.spec_from_file_location('synthetic_bootstrap', args[0])
        bootstrap = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = bootstrap
        spec.loader.exec_module(bootstrap)
        bootstrap.POEM_COMMIT = os.environ['SYNTHETIC_POEM_COMMIT']
        bootstrap.SAR_COMMIT = os.environ['SYNTHETIC_SAR_COMMIT']
        bootstrap.MAIN_SHA256 = os.environ['SYNTHETIC_POEM_HASH']
        bootstrap.MODEL_SHA256 = os.environ['SYNTHETIC_SAR_HASH']
        sys.argv = args
        bootstrap.main()
        """)
    executable("find", """\
        import json
        import os
        import sys
        args = sys.argv[1:]
        if args[0] in json.loads(os.environ['SYNTHETIC_CLASS_ROOTS']):
            assert args[1:] == ['-mindepth', '1', '-maxdepth', '1', '-type', 'd', '-print']
            for i in range(1000):
                print(args[0] + '/synthetic-class-' + str(i))
        else:
            os.execv(os.environ['SYNTHETIC_REAL_FIND'], ['find', *args])
        """)
    executable("caffeinate", """\
        import os
        import sys
        assert sys.argv[1] == '-is'
        os.execv(sys.argv[2], sys.argv[2:])
        """)
    clean = tmp_path / "clean"
    corrupted = tmp_path / "corrupted"
    class_roots = [clean / "val"] + [
        corrupted / corruption / "1"
        for corruption in ("gaussian_noise", "shot_noise", "impulse_noise")
    ]
    for root in class_roots:
        root.mkdir(parents=True)
    output = tmp_path / "output"
    calls_file = tmp_path / "calls.jsonl"
    events_file = tmp_path / "events.jsonl"
    env = {
        **os.environ, **pins, "PYTHONDONTWRITEBYTECODE": "1",
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "KBOUND_REPO_ROOT": str(repo), "POEM_PYTHON": str(python),
        "POEM_SOURCE": str(poem), "SAR_SOURCE": str(sar), "POEM_OUTPUT": str(output),
        "IMAGENET_ROOT": str(clean), "IMAGENETC_ROOT": str(corrupted),
        "SEEDS": "0", "SEVERITIES": "1", "TEST_BATCH_SIZE": "1",
        "CORRUPTIONS": "gaussian_noise shot_noise impulse_noise",
        "SYNTHETIC_BOOTSTRAP": str(repo / "docs/research/kbound/scripts/poem_dependency_bootstrap.py"),
        "SYNTHETIC_CALLS": str(calls_file), "SYNTHETIC_EVENTS": str(events_file),
        "SYNTHETIC_CLASS_ROOTS": json.dumps([str(root) for root in class_roots]),
        "SYNTHETIC_REAL_FIND": shutil.which("find"), "SYNTHETIC_FAILURE": "",
    }

    def run(**overrides):
        result = subprocess.run(
            ["bash", str(repo / "docs/research/kbound/runbooks/run_poem_imagenetc.sh")],
            env={**env, **overrides}, capture_output=True, text=True, timeout=20,
        )
        calls = [json.loads(line) for line in calls_file.read_text().splitlines()] if calls_file.exists() else []
        events = [json.loads(line) for line in events_file.read_text().splitlines()] if events_file.exists() else []
        return result, calls, events, output

    return run


def test_runbook_executes_each_selected_noise_once_per_method(synthetic_runbook):
    result, calls, events, output = synthetic_runbook()
    assert result.returncode == 0, result.stdout + result.stderr + (output / "poem_imagenetc.log").read_text()
    assert [(event["method"], event["corruption"]) for event in events] == [
        ("poem", "gaussian_noise"), ("poem", "shot_noise"), ("poem", "impulse_noise"),
        ("no_adapt", "gaussian_noise"), ("no_adapt", "shot_noise"), ("no_adapt", "impulse_noise"),
    ]
    assert len(calls) == 7
    assert calls[0][-2:] == ["--", "--help"]
    assert "--single-corruption" not in calls[0]
    for call, event in zip(calls[1:], events):
        delimiter = call.index("--")
        assert call.index("--single-corruption") < delimiter
        assert call[call.index("--single-corruption") + 1] == event["corruption"]
        assert call.index("--corruption") > delimiter
        assert call[call.index("--corruption") + 1] == event["corruption"]
        assert call[call.index("--output") + 1] == event["output"]
        assert (Path(event["output"]) / "POEM_PROTOCOL_DERIVATION_RECEIPT.json").is_file()


def test_runbook_stops_on_first_failed_selected_stream(synthetic_runbook):
    result, calls, events, _ = synthetic_runbook(SYNTHETIC_FAILURE="run")
    assert result.returncode == 23, result.stdout + result.stderr
    assert len(calls) == 3  # Help, gaussian noise, then the failing shot noise.
    assert [(event["method"], event["corruption"]) for event in events] == [
        ("poem", "gaussian_noise"), ("poem", "shot_noise"),
    ]
    assert "runs done" not in result.stdout


@pytest.mark.parametrize("failure, message", [
    ("dependency", "authenticated POEM dependency/entrypoint preflight failed"),
    ("backend", "native POEM requires CUDA"),
    ("model", "timm resnet50_gn did not load"),
])
def test_runbook_preflight_failure_never_dispatches(synthetic_runbook, failure, message):
    result, calls, events, output = synthetic_runbook(SYNTHETIC_FAILURE=failure)
    assert result.returncode == 2, result.stdout + result.stderr
    assert message in result.stdout
    assert len(calls) == 1
    assert "--single-corruption" not in calls[0]
    assert events == []
    assert not list(output.glob("exps_*"))


def test_authenticated_dependency_reaches_clean_data_guard_without_running(synthetic_runbook):
    result, calls, events, output = synthetic_runbook(IMAGENET_ROOT="")
    assert len(calls) == 1
    assert events == []
    assert result.returncode == 2, result.stderr
    assert "set IMAGENET_ROOT" in result.stdout, result.stdout + result.stderr
    assert "RUN method=" not in result.stdout
    assert not list(output.glob("exps_*"))
