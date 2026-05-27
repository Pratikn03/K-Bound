import pytest


def test_distilbert_forward_smoke():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from uais_v.models.nlp_text_model import DistilBERTClassifier, get_tokenizer

    tokenizer = get_tokenizer("distilbert-base-uncased")
    enc = tokenizer("hello world", return_tensors="pt")
    model = DistilBERTClassifier("distilbert-base-uncased", num_labels=2)
    out = model(enc["input_ids"], enc["attention_mask"])
    assert out.shape[1] == 2
