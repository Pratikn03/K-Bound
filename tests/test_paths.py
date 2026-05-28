from uais.utils.paths import DATA_DIR, PROJECT_ROOT


def test_paths_exist():
    CONFIG_DIR = PROJECT_ROOT / "configs"
    assert PROJECT_ROOT.exists()
    assert DATA_DIR.exists()
    assert CONFIG_DIR.exists()

