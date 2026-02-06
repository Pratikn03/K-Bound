# Dataset Inventory

Use this table to track candidate datasets, licenses, and split plans for Phase 2.

| Dataset | Domain | Modality | Labels | License | Source | Split Plan | Notes |
|---------|--------|----------|--------|---------|--------|------------|-------|
| creditcard.csv | fraud | tabular | binary | TBD | local repo | stratified 70/15/15 | high imbalance |
| UNSW-NB15 | cyber | tabular + sequential | attack | TBD | public | time-based split | align by time window |
| CERT Insider Threat v4 | behavior | logs + sequential | insider activity | TBD | public | time-based split | align by user_id |
| Enron Email | nlp | text | weak/noisy | TBD | public | time-based split | weak supervision |
| MVTec AD | vision | image | anomaly class | TBD | public | stratified split | score-based outputs |

## Notes

- Fill in licenses and source URLs before data use.
- Track any preprocessing or filtering applied to the raw data.
