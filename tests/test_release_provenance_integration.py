"""Synthetic-only release boundary regressions; no project outcomes are read."""
import importlib.util
import ast
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "docs/research/kbound/scripts"
CONVERTER = ROOT / "docs/research/kbound/runbooks/convert_official_logs_to_decisions.py"
AUDITOR = SCRIPTS / "audit_official_baselines.py"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


@pytest.fixture
def inputs(tmp_path):
    conditions = [f"synthetic-{i:03d}" for i in range(432)]
    return {
        "conditions": conditions,
        "stream": write_json(tmp_path / "per_condition_cifar10c_tent_seed0.json", {"records": [{"condition": c} for c in conditions]}),
        "logs": write_json(tmp_path / "logs.json", {c: "adapt" for c in conditions}),
        "environment": write_json(tmp_path / "environment.json", {"synthetic": "environment"}),
        "toolchain": write_json(tmp_path / "toolchain.json", {"synthetic": "toolchain"}),
        "out": tmp_path / "decisions.json",
    }


def convert(inputs, *extra):
    return subprocess.run(
        [sys.executable, "-B", str(CONVERTER), "--method", "aetta", "--logs", str(inputs["logs"]),
         "--stream", str(inputs["stream"]), "--out", str(inputs["out"]), *map(str, extra)],
        text=True, capture_output=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def test_unverified_conversion_remains_available(inputs):
    result = convert(inputs)
    assert result.returncode == 0, result.stderr
    output = json.loads(inputs["out"].read_text())
    assert output["official_label_allowed"] is False
    assert len(output["decisions"]) == 432


@pytest.mark.parametrize("claim", [True, "true", 1])
def test_forged_legacy_audit_cannot_promote(inputs, claim):
    audit = write_json(inputs["out"].parent / "audit.json", {
        "schema_version": 2, "methods": {"aetta": {"official_label_allowed": claim}},
    })
    inputs["out"].write_text("previous-good-output\n")
    result = convert(inputs, "--provenance-audit", audit, "--require-official-label")
    assert result.returncode != 0, "legacy audit boolean must not confer official status"
    assert inputs["out"].read_text() == "previous-good-output\n"


def test_scorer_rejects_self_promoted_wrapper_without_evidence(inputs):
    path = write_json(inputs["out"], {
        "schema_version": 2, "method": "aetta", "official_label_allowed": True,
        "label": "official_implementation_under_protocol_adapter",
        "decisions": {c: "adapt" for c in inputs["conditions"]},
    })
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("headtohead_under_test", SCRIPTS / "official_baselines_headtohead.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises((SystemExit, ValueError), match="provenance|official"):
        module.load_external(path, inputs["conditions"])


def test_release_flags_are_bound_in_nonpromotable_schema3_audit(inputs):
    repo = inputs["out"].parent / "synthetic_repo"
    repo.mkdir()
    out = repo / "audit.json"
    result = subprocess.run([
        sys.executable, "-B", str(AUDITOR), "--repo", str(repo), "--output", str(out),
        "--locked-stream", str(inputs["stream"]),
        "--environment-receipt", str(inputs["environment"]),
        "--toolchain-receipt", str(inputs["toolchain"]),
    ], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    audit = json.loads(out.read_text())
    assert audit["schema_version"] == 3
    import hashlib
    for field, source in [("locked_stream_sha256", "stream"), ("environment_receipt_sha256", "environment"),
                          ("toolchain_receipt_sha256", "toolchain")]:
        assert audit["promotion_binding"][field] == hashlib.sha256(inputs[source].read_bytes()).hexdigest()
    assert audit["promotion_binding"]["condition_count"] == 432
    assert audit["all_promotable"] is False


def load_module(name, path):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def signed_native_fixture(inputs, tmp_path, monkeypatch):
    """Invented native trace and ephemeral TEST witness; no command is executed.

    Only this test process's validator constant is monkeypatched. The production
    trust-root file/key never changes, and the generated trace is not evidence.
    """
    import base64
    import hashlib
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    auditor = load_module('auditor_positive_fixture', AUDITOR)
    import official_baseline_provenance as provenance
    key = Ed25519PrivateKey.generate()
    monkeypatch.setattr(provenance, 'NATIVE_EXECUTION_WITNESS_PUBLIC_KEY_B64',
                        base64.b64encode(key.public_key().public_bytes_raw()).decode())
    repo = tmp_path/'synthetic-native-repository'; repo.mkdir()
    source = repo/'AETTA'; source.mkdir()
    (source/'model.py').write_text('# synthetic source, never executed\n')
    (source/'LICENSE.txt').write_text('Synthetic fixture license\n')
    (source/'aetta.yml').write_text('name: synthetic-test\n')
    (source/'VENDOR.md').write_text('upstream_commit_sha: ' + 'a'*40 + '\n')
    producer = repo/'docs/research/kbound/scripts/run_official_native.py'
    producer.parent.mkdir(parents=True)
    producer.write_text('# synthetic runner identity, never executed\n')
    out = repo/'out'; native = out/'aetta_native'; native.mkdir(parents=True)
    inputs['logs'] = write_json(native/'decisions.json', {c:'adapt' for c in inputs['conditions']})
    inputs['out'] = out/'aetta_decisions.staged.json'
    result = convert(inputs, '--stage')
    assert result.returncode == 0, result.stderr

    def canonical_bytes(value):
        return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n').encode()

    def canonical_file(name, value):
        (native/name).write_bytes(canonical_bytes(value))

    invocation = dict(command=['synthetic-command-not-executed'], method='aetta',
        producer=dict(path=producer.relative_to(repo).as_posix(), sha256=provenance.sha256_file(producer)),
        schema='kbound-official-native-invocation-v1',
        source=dict(path='AETTA', tree_sha256=auditor.tree_hash(source)), working_directory='AETTA')
    canonical_file('native_invocation.json', invocation)
    artifact_hashes = {inputs['logs'].relative_to(repo).as_posix(): provenance.sha256_file(inputs['logs'])}
    runner = dict(artifacts_sha256=artifact_hashes,
        invocation_sha256=hashlib.sha256(canonical_bytes(invocation)).hexdigest(), method='aetta',
        producer=invocation['producer'], returncode=0,
        schema='kbound-official-native-runner-receipt-v2', status='zero-exit-recorded')
    canonical_file('native_runner_receipt.json', runner)
    runner_hash = hashlib.sha256(canonical_bytes(runner)).hexdigest()
    completion = dict(exit_status=0, log_sha256=artifact_hashes, method='aetta',
        runner_receipt_sha256=runner_hash, schema='kbound-official-native-completion-v3',
        status='runner-zero-exit-recorded')
    canonical_file('native_completion.json', completion)
    claim = dict(method='aetta', runner_receipt_sha256=runner_hash,
        schema=provenance.NATIVE_EXECUTION_ATTESTATION_SCHEMA,
        witness_key_id=provenance.NATIVE_EXECUTION_WITNESS_KEY_ID)
    attestation = dict(claim, signature_b64=base64.b64encode(key.sign(canonical_bytes(claim))).decode())
    canonical_file('native_execution_attestation.json', attestation)
    return dict(inputs=inputs, repo=repo, out=out, native=native, auditor=auditor,
                audit=out/'audit.json', provenance=provenance)


def run_audit(fixture, monkeypatch, *extra):
    inputs = fixture['inputs']
    monkeypatch.setattr(sys, 'argv', [str(AUDITOR), '--repo', str(fixture['repo']),
        '--out-dir', str(fixture['out']), '--output', str(fixture['audit']),
        '--locked-stream', str(inputs['stream']), '--environment-receipt', str(inputs['environment']),
        '--toolchain-receipt', str(inputs['toolchain']), *extra])
    return fixture['auditor'].main()


def promote_fixture(f, monkeypatch):
    inputs = f['inputs']
    assert run_audit(f, monkeypatch) == 0
    audit = json.loads(f['audit'].read_text())
    assert audit['schema_version'] == 3
    assert audit['methods']['aetta']['official_label_allowed'] is True
    converter = load_module('converter_positive_fixture', CONVERTER)
    final = f['out']/'aetta_decisions.json'
    monkeypatch.setattr(sys, 'argv', [str(CONVERTER), '--method', 'aetta', '--logs', str(inputs['logs']),
        '--stream', str(inputs['stream']), '--out', str(final), '--provenance-audit', str(f['audit']),
        '--environment-receipt', str(inputs['environment']), '--toolchain-receipt', str(inputs['toolchain']),
        '--require-official-label'])
    converter.main()
    return final


def test_authentic_synthetic_stage_audit_converter_scorer_roundtrip(signed_native_fixture, monkeypatch):
    f = signed_native_fixture; inputs = f['inputs']
    final = promote_fixture(f, monkeypatch)
    wrapper = json.loads(final.read_text())
    assert wrapper['official_label_allowed'] is True
    scorer = load_module('scorer_positive_fixture', SCRIPTS/'official_baselines_headtohead.py')
    values, official, label = scorer.load_external(final, inputs['conditions'], method='aetta',
        provenance_audit=f['audit'], source_log=inputs['logs'], locked_stream=inputs['stream'],
        environment_receipt=inputs['environment'], toolchain_receipt=inputs['toolchain'])
    assert official is True
    assert len(values) == 432
    assert label == f['provenance'].OFFICIAL_LABEL


@pytest.mark.parametrize('mutation', ['missing_witness', 'wrong_decision', 'stale_stream', 'changed_native', 'changed_source'])
def test_auditor_does_not_promote_inconsistent_staged_evidence(signed_native_fixture, monkeypatch, mutation):
    f = signed_native_fixture; inputs = f['inputs']
    if mutation == 'missing_witness':
        # Only the test-owned synthetic attestation is removed.
        (f['native']/'native_execution_attestation.json').unlink()
    elif mutation == 'wrong_decision':
        stage = json.loads(inputs['out'].read_text())
        stage['decisions'][inputs['conditions'][0]] = 'freeze'
        write_json(inputs['out'], stage)
    elif mutation == 'stale_stream':
        inputs['stream'].write_text(inputs['stream'].read_text() + ' ')
    elif mutation == 'changed_native':
        inputs['logs'].write_text(inputs['logs'].read_text() + ' ')
    elif mutation == 'changed_source':
        (f['repo']/'AETTA/model.py').write_text('# changed synthetic source\n')
    assert run_audit(f, monkeypatch) == 0
    audit = json.loads(f['audit'].read_text())
    assert audit['methods']['aetta']['official_label_allowed'] is False


def test_require_promotable_failure_preserves_previous_audit(signed_native_fixture, monkeypatch):
    f = signed_native_fixture
    f['audit'].write_text('previous-good-audit\n')
    assert run_audit(f, monkeypatch, '--require-promotable') != 0  # no POEM fixture
    assert f['audit'].read_text() == 'previous-good-audit\n'


def test_preliminary_method_record_cannot_self_promote(signed_native_fixture):
    f = signed_native_fixture
    write_json(f['out']/'aetta_decisions.json', {c:'adapt' for c in f['inputs']['conditions']})
    record = f['auditor'].audit_aetta(f['repo'], f['out'])
    assert record['official_label_allowed'] is False


def scoring_arguments(f, final, output):
    inputs = f['inputs']
    return ['scorer', '--candidate', 'tent', '--decisions', f'aetta={final}',
            '--provenance-audit', str(f['audit']), '--locked-stream', str(inputs['stream']),
            '--environment-receipt', str(inputs['environment']), '--toolchain-receipt', str(inputs['toolchain']),
            '--source-logs', f"aetta={inputs['logs']}", '--out', str(output)]


@pytest.mark.parametrize('changed', ['audit','logs','stream','environment','toolchain','wrapper'])
def test_scorer_cli_revalidates_before_outcome_loading(signed_native_fixture, monkeypatch, changed):
    f = signed_native_fixture; inputs = f['inputs']
    final = promote_fixture(f, monkeypatch)
    if changed == 'wrapper':
        raw = json.loads(final.read_text()); raw['decisions'][inputs['conditions'][0]] = 'freeze'
        write_json(final, raw)
    else:
        path = f['audit'] if changed == 'audit' else inputs[changed]
        path.write_text(path.read_text() + ' ')
    scorer = load_module('scorer_cli_negative', SCRIPTS/'official_baselines_headtohead.py')
    monkeypatch.setattr(scorer, 'STRESS', str(inputs['stream'].parent))
    monkeypatch.setattr(scorer, 'load', lambda *args: pytest.fail('outcomes loaded before provenance rejection'))
    output = f['out']/'scores.json'; output.write_text('previous-good-score\n')
    monkeypatch.setattr(sys, 'argv', scoring_arguments(f, final, output))
    with pytest.raises(ValueError, match='provenance|binding|payload'):
        scorer.main()
    assert output.read_text() == 'previous-good-score\n'


def test_scorer_cli_accepts_validated_current_bindings_before_outcome_boundary(signed_native_fixture, monkeypatch):
    f = signed_native_fixture; final = promote_fixture(f, monkeypatch)
    scorer = load_module('scorer_cli_positive', SCRIPTS/'official_baselines_headtohead.py')
    monkeypatch.setattr(scorer, 'STRESS', str(f['inputs']['stream'].parent))
    class OutcomeBoundaryReached(Exception): pass
    def stop_before_outcomes(*args, **kwargs):
        assert kwargs.get('expected_sha256') == f['provenance'].sha256_file(f['inputs']['stream'])
        raise OutcomeBoundaryReached
    monkeypatch.setattr(scorer, 'load', stop_before_outcomes)
    monkeypatch.setattr(sys, 'argv', scoring_arguments(f, final, f['out']/'scores.json'))
    with pytest.raises(OutcomeBoundaryReached):
        scorer.main()
    assert not (f['out']/'scores.json').exists()


def test_scoring_load_checks_exact_bytes_before_decoding(inputs, monkeypatch):
    scorer = load_module('scorer_exact_byte_fixture', SCRIPTS/'official_baselines_headtohead.py')
    monkeypatch.setattr(scorer, 'STRESS', str(inputs['stream'].parent))
    # The synthetic JSON has no scoring fields. A provenance mismatch must be
    # detected before JSON/scoring-field interpretation, never merely afterward.
    with pytest.raises(ValueError, match='provenance|binding'):
        scorer.load('tent', expected_sha256='a'*64)


@pytest.mark.parametrize('value', [True, 'NaN', '0.8'])
def test_invalid_native_accuracy_estimates_do_not_publish(inputs, value):
    records = {c: {'est_acc_adapted': .8, 'est_acc_frozen': .7} for c in inputs['conditions']}
    records[inputs['conditions'][0]]['est_acc_frozen'] = value
    write_json(inputs['logs'], records)
    inputs['out'].write_text('previous-good-output\n')
    result = convert(inputs, '--stage')
    assert result.returncode != 0
    assert inputs['out'].read_text() == 'previous-good-output\n'


@pytest.mark.parametrize('scale', [1, 100])
def test_accuracy_conversion_preserves_fraction_and_native_percentage_units(inputs, scale):
    # Vendored AETTA/learner/dnn.py emits 100 * (1 - updated). No implicit
    # rescaling or newly imposed fraction-only scientific contract is allowed.
    records = {c: {'est_acc_adapted': .8 * scale, 'est_acc_frozen': .7 * scale}
               for c in inputs['conditions']}
    records[inputs['conditions'][0]]['est_acc_adapted'] = .6 * scale
    write_json(inputs['logs'], records)
    result = convert(inputs, '--stage')
    assert result.returncode == 0, result.stderr
    decisions = json.loads(inputs['out'].read_text())['decisions']
    assert decisions[inputs['conditions'][0]] == 'freeze'
    assert decisions[inputs['conditions'][1]] == 'adapt'


def test_item11_runbook_orders_staging_audit_required_promotion_then_scoring():
    runbook = ROOT / 'docs/research/kbound/runbooks/run_item11_official_baselines.sh'
    raw = runbook.read_text()
    # Parse continued command text only: never execute the workflow or expand
    # variables. These arguments complement the synthetic Python entrypoint tests.
    commands = [shlex.split(line) for line in raw.replace('\\\n', ' ').splitlines()
                if line.lstrip().startswith('"$PY" ')]
    assert [Path(command[1]).name for command in commands] == [
        'convert_official_logs_to_decisions.py', 'audit_official_baselines.py',
        'convert_official_logs_to_decisions.py', 'official_baselines_headtohead.py']
    stage, audit, convert_command, scorer = commands
    assert '--stage' in stage and '--provenance-audit' not in stage
    assert stage[stage.index('--out') + 1] == '$OUT/aetta_decisions.staged.json'
    for command in (audit, scorer):
        assert command[command.index('--locked-stream') + 1] == '$STREAM'
    for command in (audit, convert_command, scorer):
        assert command[command.index('--environment-receipt') + 1] == '$ENVIRONMENT_RECEIPT'
        assert command[command.index('--toolchain-receipt') + 1] == '$TOOLCHAIN_RECEIPT'
    for command in (convert_command, scorer):
        assert command[command.index('--provenance-audit') + 1] == '$AUDIT'
    assert '--require-official-label' in convert_command
    assert scorer[scorer.index('--source-logs') + 1] == 'aetta=$AETTA_LOG_JSON'
    assert '-s "$OUT/aetta_decisions.json"' not in raw, 'stale outputs must never trigger scoring'
    assert raw.index('POEM_LOG_JSON cannot be scored') < raw.index('"$PY" ')
    assert subprocess.run(['bash', '-n', str(runbook)], capture_output=True).returncode == 0


def test_poem_audit_retains_explicit_unimplemented_protocol_adapter_blocker(inputs):
    auditor = load_module('poem_protocol_auditor', AUDITOR)
    record = auditor.audit_poem(inputs['out'].parent, inputs['out'].parent)
    assert record['checks'].get('protocol_adapter_compatible') is False


def test_direct_cifar_scorer_refuses_official_poem_protocol_mismatch(inputs):
    scorer = load_module('poem_protocol_scorer', SCRIPTS/'official_baselines_headtohead.py')
    path = write_json(inputs['out'], dict(schema_version=3, status='VALIDATED_OFFICIAL',
        method='poem', official_label_allowed=True, label='official_implementation_under_protocol_adapter',
        decisions={c:'adapt' for c in inputs['conditions']}))
    with pytest.raises(ValueError, match='POEM.*CIFAR|protocol adapter'):
        scorer.load_external(path, inputs['conditions'], method='poem')


def test_release_source_inventory_seals_the_shared_provenance_helper():
    # AST literal inspection, never import/build a real source seal.
    tree = ast.parse((SCRIPTS/'build_release_source_seal.py').read_text())
    assignment = next(node for node in tree.body if isinstance(node, ast.AnnAssign)
                      and isinstance(node.target, ast.Name) and node.target.id == 'EXPLICIT_FILES')
    explicit = ast.literal_eval(assignment.value)
    assert 'docs/research/kbound/scripts/official_decision_artifact.py' in explicit['release_code']
    assert 'tests/test_release_provenance_integration.py' in explicit['release_validation']


def test_release_validation_declares_new_synthetic_regression_selector():
    path = ROOT/'docs/research/kbound/runbooks/release_candidate.sh'
    raw = path.read_text()
    section = raw.split('MODE validate-results', 1)[1].split('validate_manuscript_claims()', 1)[0]
    assert 'tests/test_release_provenance_integration.py' in section
    assert subprocess.run(['bash', '-n', str(path)], capture_output=True).returncode == 0


def saved_audit_metadata_fixture(schema=3):
    return dict(schema_version=schema, all_promotable=False,
        provenance_path_binding=dict(schema='git-repository-relative-posix-v1', root='.',
            root_role='git_repository_root', content_scope='working_tree_at_generation',
            generation_base_git_head='a'*40),
        methods={method: {'official_label_allowed': False} for method in ('aetta', 'poem')},
        promotion_binding=dict(locked_stream_sha256='b'*64, environment_receipt_sha256='c'*64,
            toolchain_receipt_sha256='d'*64, condition_count=432))


def exercise_saved_audit_test(tmp_path, monkeypatch, artifact):
    module = load_module('saved_audit_metadata_contract', ROOT/'tests/test_official_baseline_provenance.py')
    monkeypatch.setattr(module, 'REPO', tmp_path)
    write_json(tmp_path/'experiments/kbound/results/official_repro_v1/OFFICIAL_BASELINE_AUDIT.json', artifact)
    module.test_saved_audit_uses_only_repo_relative_provenance_paths()


def test_saved_audit_contract_accepts_bound_schema3_metadata(tmp_path, monkeypatch):
    exercise_saved_audit_test(tmp_path, monkeypatch, saved_audit_metadata_fixture())


def test_saved_legacy_claims_remain_metadata_not_promotion(tmp_path, monkeypatch):
    artifact = saved_audit_metadata_fixture(2)
    artifact['methods']['aetta']['official_label_allowed'] = True
    artifact['all_promotable'] = True
    exercise_saved_audit_test(tmp_path, monkeypatch, artifact)
    import official_baseline_provenance as provenance
    path = tmp_path/'experiments/kbound/results/official_repro_v1/OFFICIAL_BASELINE_AUDIT.json'
    with pytest.raises(ValueError, match='schema_version 3'):
        provenance.validate_promotable_audit(path, method='aetta', decisions={'synthetic': 'adapt'},
            source_log_sha256='e'*64, locked_stream_sha256='b'*64,
            environment_receipt_sha256='c'*64, toolchain_receipt_sha256='d'*64)


@pytest.mark.parametrize('field,value', [
    ('locked_stream_sha256', None), ('environment_receipt_sha256', 'not-a-hash'),
    ('toolchain_receipt_sha256', 'e'*63), ('condition_count', 431), ('condition_count', 432.0),
])
def test_saved_schema3_audit_contract_rejects_malformed_bindings(tmp_path, monkeypatch, field, value):
    artifact = saved_audit_metadata_fixture()
    artifact['promotion_binding'][field] = value
    with pytest.raises(AssertionError):
        exercise_saved_audit_test(tmp_path, monkeypatch, artifact)
