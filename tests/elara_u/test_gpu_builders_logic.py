"""CPU smoke for the GPU builders' non-GPU logic (glob expansion + exclusion)."""
import os, tempfile
from scripts.elara_u.gpu_build_image_embeddings import _expand


def test_expand_includes_and_excludes(tmp_path):
    (tmp_path / "good").mkdir(); (tmp_path / "bad").mkdir()
    for d in ("good", "bad"):
        for i in range(3):
            (tmp_path / d / f"{i}.png").write_text("x")
    allp = _expand([str(tmp_path / "*/*.png")], None)
    assert len(allp) == 6
    no_good = _expand([str(tmp_path / "*/*.png")], "*/good/*")
    assert len(no_good) == 3 and all("/bad/" in p for p in no_good)
