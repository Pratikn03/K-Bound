"""Projection-only tests; no original private receipt or Lean run required."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]/'docs/research/kbound/scripts'

def module():
    path=ROOT/'project_proof_receipts.py'
    assert path.exists(), 'Projection implementation is not present yet'
    spec=importlib.util.spec_from_file_location('projection_under_test',path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

def sample(role):
    data=dict(schema='historical-schema',date='2026-09-07',status='SCOPED_NOT_PAPER_WIDE',
        registered_declarations=303,paper_wide_closure=False,empirical_assumptions_verified=False,
        source_sha256={'formal/Example.lean':'a'*64},remaining=['A fixed observation model is required.'],
        incremental_build_with_cached_dependencies=True,cold_dependency_rebuild=False,
        allowed_observed_axioms=['Classical.choice','Quot.sound','propext'])
    if role=='verification':
        data.update(independent_review_reports_sha256={'/private-location/review-a/report.txt':'b'*64,
            '/private-location/review-b/report.txt':'c'*64},build_snapshot='/private-location/source-snapshot',
            bound_nonformal_snapshot='/private-location/prose-snapshot',reused_compiled_build_workspace='/private-location/build-workspace')
    else:
        data.update(independent_review_report='/private-location/report.txt',independent_review_report_sha256='b'*64,
            independent_replay_receipt='/private-location/replay.json',independent_replay_receipt_sha256='c'*64,
            base_index_sha256='d'*64,new_lean_build_needed_for_status_only_delta=False)
    return data

@pytest.mark.parametrize('role,count',[('verification',5),('followon',2)])
def test_only_declared_locations_change_and_original_object_survives(role,count):
    m=module();original=sample(role);before=copy.deepcopy(original)
    projected,redactions=m.redact_payload(original,role)
    assert original==before
    assert len(redactions)==count
    changed={'independent_review_reports_sha256','build_snapshot','bound_nonformal_snapshot','reused_compiled_build_workspace'} if role=='verification' else {'independent_review_report','independent_replay_receipt'}
    assert {k:v for k,v in projected.items() if k not in changed}=={k:v for k,v in before.items() if k not in changed}
    assert '/private-location/' not in json.dumps((projected,redactions))
    assert projected['registered_declarations']==303
    assert projected['paper_wide_closure'] is False
    assert all(len(r['redacted_utf8_sha256'])==64 for r in redactions)
    if role=='verification':
        assert list(projected['independent_review_reports_sha256'].values())==['b'*64,'c'*64]
        assert redactions[0]['redacted_utf8_sha256']==hashlib.sha256(b'/private-location/review-a/report.txt').hexdigest()

def test_redaction_is_deterministic():
    m=module();one=m.redact_payload(sample('verification'),'verification');two=m.redact_payload(sample('verification'),'verification')
    assert one==two

@pytest.mark.parametrize('change',[
    lambda d:d.update(unexpected_private_path='/private-location/not-in-policy'),
    lambda d:d['source_sha256'].update({'/private-location/private-key':'d'*64}),
    lambda d:d.pop('build_snapshot'),
    lambda d:d.update(build_snapshot=123),
    lambda d:d.update(independent_review_reports_sha256={}),
])
def test_unexpected_or_malformed_redaction_surface_is_rejected(change):
    m=module();d=sample('verification');change(d)
    with pytest.raises(ValueError):m.redact_payload(d,'verification')

@pytest.mark.parametrize('payload',[b'{}',b'{"a":1,"a":2}',b'{"a":NaN}'])
def test_unapproved_or_malformed_original_bytes_are_rejected(payload):
    with pytest.raises(ValueError):module().project_bytes(payload,'verification')

def test_unknown_role_rejected():
    with pytest.raises(ValueError):module().redact_payload({},'invented')

def test_cli_rejects_wrong_original_without_creating_output(tmp_path):
    module();source=tmp_path/'original.json';source.write_text('{}')
    output=tmp_path/'portable.json'
    result=subprocess.run([sys.executable,str(ROOT/'project_proof_receipts.py'),'--role','verification',
        '--source',str(source),'--output',str(output)],capture_output=True,text=True)
    assert result.returncode!=0 and not output.exists()
    assert source.read_text()=='{}'
