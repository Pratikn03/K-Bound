import inspect

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
