"""Execute the public synthetic example against the real authority boundary."""
from pathlib import Path
import re

import pytest


def test_readme_quickstart_executes_and_rejects_wrong_external_identity():
    readme = Path(__file__).resolve().parents[1] / "README.md"
    snippet = re.search(r"```python\n(.*?)```", readme.read_text(), re.S)
    assert snippet is not None
    namespace = {}
    exec(compile(snippet.group(1), "README.md:quickstart", "exec"), namespace)
    assert namespace["certificate"] is not None
    with pytest.raises(ValueError):
        namespace["kga"].certify_evidence(
            namespace["estimator"],
            protocol_sha256=namespace["protocol_sha"],
            expected_estimator_payload_sha256="0" * 64,
        )
