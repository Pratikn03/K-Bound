import pytest


def test_distilbert_forward_smoke():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    enc = tokenizer("hello world", return_tensors="pt")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    out = model(enc["input_ids"], enc["attention_mask"])
    assert out.logits.shape[1] == 2
