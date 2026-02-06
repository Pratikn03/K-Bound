# Attention Fusion API

The enhanced API exposes an attention fusion endpoint when a checkpoint exists at
`models/fusion/attention/attention_fusion.pt`.

## Endpoint

`POST /predict_attention_fusion`

## Request

```json
{
  "domains": [
    {"domain": "fraud", "score": 0.82, "confidence": 0.91, "embeddings": [0.1, 0.2]},
    {"domain": "cyber", "score": 0.40, "confidence": 0.55, "embeddings": [0.3, 0.4]}
  ]
}
```

- `domain` must match a domain in the trained checkpoint’s `domain_order`.
- `embeddings` length can be shorter; missing values are zero-padded.

## Response

```json
{
  "fusion_risk": 0.73,
  "domain_order": ["behavior", "cyber", "fraud"],
  "attention_mean": [0.31, 0.28, 0.41],
  "domain_confidence": [0.72, 0.64, 0.83]
}
```

## Notes

- If confidence is omitted for a provided domain, it defaults to 1.0.
- Missing domains are masked automatically.
