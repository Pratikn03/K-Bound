import inspect
from pathlib import Path

from uais.data import load_behavior_data, load_cyber_data, load_datasets, load_fraud_data


def test_public_domain_loaders_make_synthetic_fallback_opt_in():
    loaders = [
        load_fraud_data.load_fraud_data,
        load_fraud_data.load_creditcard,
        load_cyber_data.load_cyber_data,
        load_behavior_data.load_behavior_data,
        load_datasets.load_fraud_data,
        load_datasets.load_cyber_data,
        load_datasets.load_behavior_data,
    ]

    for loader in loaders:
        signature = inspect.signature(loader)
        assert signature.parameters["allow_synthetic"].default is False


def test_cyber_csv_discovery_prefers_official_unsw_split_files(tmp_path: Path):
    train = tmp_path / "UNSW_NB15_training-set.csv"
    test = tmp_path / "UNSW_NB15_testing-set.csv"
    extra = tmp_path / "notes_export.csv"
    for path in [train, test, extra]:
        path.write_text("label,value\n0,1\n", encoding="utf-8")

    selected = load_cyber_data._find_cyber_csvs(tmp_path)

    assert selected == [test, train]
