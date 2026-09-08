#!/usr/bin/env python3
"""Path-only portable companions for two pinned historical proof receipts.

No proof, source file, dependency, build workspace or linked artifact is opened.
The tool reads exactly its supplied original receipt and exclusively creates a
new companion. Its preserved historical build assertions are not fresh checks.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

RECEIPTS = {
    'verification': ('actual_fibre_radius_verification_20260907.json',
                     '81db04bd9ab156f554a58661a01d81db407995a273f833b2f78380907701ed54'),
    'followon': ('actual_fibre_radius_followon_20260907.json',
                 'c464569c3b1379b53db2e2b3e1512b2705458f3acc14c3b8c90d0cef2bc0b793'),
}
VALUE_ROLES = {
    'verification': {
        'build_snapshot': 'private-workspace/build-source-snapshot',
        'bound_nonformal_snapshot': 'private-workspace/bound-nonformal-snapshot',
        'reused_compiled_build_workspace': 'private-workspace/reused-compiled-build',
    },
    'followon': {
        'independent_review_report': 'private-artifact/independent-303-review-report',
        'independent_replay_receipt': 'private-artifact/independent-303-replay-receipt',
    },
}

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')

def _private_path(value):
    return isinstance(value,str) and bool(value.startswith(('/', '~/', '\\\\')) or re.match(r'^[A-Za-z]:[\\/]',value)
        or re.search(r'(?:^|\s)/(?:Volumes|Users|home|private|tmp)/',value))

def _reject_private(value):
    if isinstance(value,dict):
        for key,item in value.items():
            if _private_path(key):raise ValueError('Unmapped private path in dictionary key')
            _reject_private(item)
    elif isinstance(value,list):
        for item in value:_reject_private(item)
    elif _private_path(value):
        raise ValueError('Unmapped private path in value')

def redact_payload(document,role):
    if role not in RECEIPTS or not isinstance(document,dict):
        raise ValueError('Unknown receipt role or invalid receipt object')
    payload=copy.deepcopy(document);redactions=[]
    if role=='verification':
        key='independent_review_reports_sha256';reports=payload.get(key)
        if not isinstance(reports,dict) or len(reports)!=2:
            raise ValueError('Expected exactly two historical review path keys')
        replacements={}
        for index,(private_path,report_hash) in enumerate(reports.items(),1):
            if not _private_path(private_path) or not isinstance(report_hash,str) or not re.fullmatch('[0-9a-f]{64}',report_hash):
                raise ValueError('Invalid historical review-path binding')
            label=f'private-artifact/independent-review-report-{index}'
            replacements[label]=report_hash
            redactions.append(dict(kind='dictionary_key',container_pointer='/'+key,
                original_key_position=index-1,replacement_key=label,
                redacted_utf8_sha256=sha(private_path.encode('utf-8')),
                reason='Private machine path relabeled; original linked-artifact SHA256 value is unchanged.'))
        payload[key]=replacements
    for key,label in VALUE_ROLES[role].items():
        value=payload.get(key)
        if not _private_path(value):raise ValueError('Missing or nonpath historical location: '+key)
        payload[key]=label
        redactions.append(dict(kind='string_value',json_pointer='/'+key,replacement_value=label,
            redacted_utf8_sha256=sha(value.encode('utf-8')),
            reason='Private machine path relabeled; no assertion that the workspace or linked artifact is distributed.'))
    _reject_private(payload)
    return payload,redactions

def _pairs(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate JSON key')
        result[key]=value
    return result

def _nonfinite(value):
    raise ValueError('Nonfinite JSON constant: '+value)

def project_bytes(raw,role):
    if role not in RECEIPTS:raise ValueError('Unknown receipt role')
    name,expected=RECEIPTS[role]
    if sha(raw)!=expected:raise ValueError('Original bytes do not match the approved dated receipt')
    document=json.loads(raw,object_pairs_hook=_pairs,parse_constant=_nonfinite)
    payload,redactions=redact_payload(document,role)
    return dict(schema='kbound-portable-dated-proof-receipt-v1',projection_version=1,
        activity='PATH_ONLY_PROJECTION_NO_NEW_PROOF_BUILD_OR_SCIENTIFIC_REVIEW',
        original_receipt=dict(logical_path='formal/'+name,sha256=expected,date=document['date'],original_bytes_included=False),
        historical_receipt=payload,redactions=redactions,redaction_count=len(redactions),
        projected_payload_sha256=sha(canonical(payload)),
        preservation='All original fields and values are preserved except the explicitly listed private path values and dictionary keys. Proof statuses, counts, source/artifact hashes, assumptions, historical build results and limitations are unchanged.',
        interpretation='This is a portable companion to dated evidence, not a new Lean build, original signed attestation, whole-paper proof claim or verification of current manuscript applicability. Private-workspace/private-artifact labels are redacted locators, not delivered file paths. Original and linked-artifact SHA256 values identify bytes; hashes alone do not establish proof correctness or public availability.',
        new_lean_build_performed=False,original_receipt_modified=False)

def render(projected):
    return (json.dumps(projected,indent=2,sort_keys=True,ensure_ascii=True,allow_nan=False)+'\n').encode('ascii')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role',choices=tuple(RECEIPTS),required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    payload=render(project_bytes(args.source.read_bytes(),args.role))
    with args.output.open('xb') as handle:handle.write(payload)

if __name__=='__main__':main()
