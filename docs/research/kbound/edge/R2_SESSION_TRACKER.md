# R2 capture session tracker

Update after each recording block. **Do not** open held-out sessions (S07–S10) until dev is sealed.

| Session | Split | Boxes | Phone | Target clips | Captured | Date | Notes |
|---------|-------|-------|-------|-------------:|---------:|------|-------|
| S01 | source_train | P01–P06 | phone_a | 240 | 0 | | |
| S02 | source_val | P07–P08 | phone_a | 80 | 0 | | **0.80 gate** |
| S03 | calibration_fit_a | mixed | phone_a | 128 | 0 | | |
| S04 | calibration_fit_b | mixed | phone_a | 128 | 0 | | |
| S05 | calibration_conformal_a | mixed | phone_a | 128 | 0 | | different day |
| S06 | calibration_conformal_b | mixed | phone_a | 128 | 0 | | different day |
| — | **SEAL** | | | | | | `start_r2_physical_capture.sh seal` |
| S07 | heldout_a | P09–P10 | phone_a | 128 | 0 | | after seal |
| S08 | heldout_b | P09–P10 | phone_a | 128 | 0 | | |
| S09 | replication_a | P09–P10 | phone_b | 128 | 0 | | second phone |
| S10 | replication_b | P09–P10 | phone_b | 128 | 0 | | |

## Quick commands

```bash
cd /Volumes/T9/uav/AutoML_Flagship_V8
bash scripts/start_r2_physical_capture.sh preflight
bash scripts/start_r2_physical_capture.sh pilot          # 4-clip warm-up
bash scripts/start_r2_physical_capture.sh session S01    # full session
```

## Staging props (4 classes)

| Class | Stage |
|-------|--------|
| `ok` | Clean label, flat on box |
| `missing_label` | Bare cardboard, no label |
| `misaligned_label` | Label rotated ~45–90° |
| `damaged_label` | Torn / peeling label |

**Never use `--mock` for publication.**
