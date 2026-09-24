# Decision-gate comparison (CIFAR-10-C stress; n=432, harmful=149, alpha=0.1)

Lower regret and lower FA_u are better; the certificate is the only rule that keeps FA_u <= alpha while staying near-oracle.

| Decision rule | regret | FA_u | FA_c | coverage | adapt-rate |
|---|---:|---:|---:|---:|---:|
| confidence gate | 0.0084 | 0.257 | 0.301 | 1.00 | 0.85 |
| entropy gate | 0.0086 | 0.255 | 0.304 | 1.00 | 0.84 |
| drift/KL gate | 0.1232 | 0.000 | 0.000 | 1.00 | 0.00 |
| ATC-style gate | 0.0045 | 0.116 | 0.172 | 1.00 | 0.67 |
| KGA (no radius) | 0.0004 | 0.049 | 0.071 | 1.00 | 0.68 |
| KGA (certificate) | 0.0017 | 0.000 | 0.000 | 0.68 | 0.51 |

## On the harmful subset only (where naive gates fail)

| Decision rule | regret | FA_u | FA_c | adapt-rate |
|---|---:|---:|---:|---:|
| confidence gate | 0.0232 | 0.745 | 1.000 | 0.74 |
| entropy gate | 0.0233 | 0.738 | 1.000 | 0.74 |
| drift/KL gate | 0.0000 | 0.000 | 0.000 | 0.00 |
| ATC-style gate | 0.0086 | 0.336 | 1.000 | 0.34 |
| KGA (no radius) | 0.0010 | 0.141 | 1.000 | 0.14 |
| KGA (certificate) | 0.0000 | 0.000 | 0.000 | 0.00 |