# R2 Physical Capture Tracker

Do not record S07-S10 until S01-S06 are validated and the conformal development
split is sealed.

| Session | Split | Objects | Phone | Physical clips | Captured | Date | Notes |
|---|---|---|---|---:|---:|---|---|
| S01 | source train | P01-P06 | A | 120 | 0 | | |
| S02 | source validation | P07-P08 | A | 40 | 0 | | source gate |
| S03 | calibration fit A | P01-P04 | A | 64 | 0 | | |
| S04 | calibration fit B | P01-P04 | A | 48 | 0 | | 32 derived windows later |
| S05 | conformal A | P01-P04 | A | 64 | 0 | | separate session/day |
| S06 | conformal B | P01-P04 | A | 48 | 0 | | seal after completion |
| SEAL | development lock | | | | | | before S07 |
| S07 | held-out A | P09-P10 | A | 64 | 0 | | after seal |
| S08 | held-out B | P09-P10 | A | 48 | 0 | | 32 derived windows later |
| S09 | replication A | P09-P10 | B | 64 | 0 | | second phone |
| S10 | replication B | P09-P10 | B | 48 | 0 | | 32 derived windows later |

## Commands

~~~bash
bash scripts/start_r2_physical_capture.sh prepare
bash scripts/start_r2_physical_capture.sh preflight
bash scripts/start_r2_physical_capture.sh session S01
bash scripts/start_r2_physical_capture.sh seal
bash scripts/start_r2_physical_capture.sh pipeline
~~~

Never use --mock for publication.
