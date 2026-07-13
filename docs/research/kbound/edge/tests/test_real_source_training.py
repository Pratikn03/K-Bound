import pytest
from kbound_edge.model import source_datasets

@pytest.fixture
def real_manifest():
    return {
        "clips": [
            {"clip_id": "c1", "session_id": "S01", "sha256": "sha1", "class_id": "ok"},
            {"clip_id": "c2", "session_id": "S02", "sha256": "sha2", "class_id": "missing_label"},
            {"clip_id": "c3", "session_id": "S03", "sha256": "sha3", "class_id": "damaged_label"},
        ]
    }

def test_training_loader_never_reads_calibration_or_test(real_manifest):
    train, val = source_datasets(real_manifest)
    assert {x.session_id for x in train} == {"S01"}
    assert {x.session_id for x in val} == {"S02"}
