# Deployment and cost observations

The delivered lifecycle is a trusted in-process reference integration. The Docker run exercises the existing certificate API. They are distinct execution surfaces, and neither is an external production deployment or a statistical safety guarantee.

## Final lifecycle run

Authority: `output/next_phase/deployment_v4/summary.json` (SHA-256 `06ec4be8616c46fae352c7e32080d16d3de421cc9fc95640965d52d39191d00d`).

The run completed 1,000 measured requests after 20 warmups, with 334 ADAPT, 333 FREEZE and 333 ABSTAIN fixture decisions. Its 1,027 journal records include warmups and fault cases and pass complete chain/hash verification. Source checkpoint bytes remain unchanged. The callbacks generate byte fixtures, so these timings do not measure neural adaptation.

| Observation | Milliseconds |
| --- | ---: |
| Lifecycle median | 0.1136 |
| Lifecycle p95 | 0.1842 |
| Lifecycle p99 | 0.2992 |
| Lifecycle maximum | 0.4394 |
| Source checkpoint read | 54.8565 |
| Source checkpoint hash median | 18.8545 |

The real source checkpoint hash measurement covers 44,805,140 bytes, not model execution. Measured process maximum RSS is 152,879,104 bytes for this fixture/hash workload. Final lifecycle regression coverage is 46 cases, including concurrent retries, invalid identities, outcome injection, candidate/estimator failures, expiry during work or journal persistence, partial journal failure, receipt capacity and exact journal-byte limits.

The contract defaults cap retained receipts at 10,000 and journal writes at 16 MiB, including rejected requests. Exhaustion raises `SessionCapacityError`, permanently closes the session and makes all previous candidate selections return the immutable source. Callers must treat this exception as source fallback, not bypass the gate. Cached retries are idempotent while the session is open; no eviction or silent replay occurs. A failed write can leave an unacknowledged/partial journal tail; that journal is incomplete, and no deployed-action history is inferred from it. Recovery requires a new reviewed session.

Candidate storage is bounded by the configured number of assessments, not a fixed byte budget; the integrating application must size that cap to actual checkpoint sizes or use a content-addressed model store. Callback resource use is trusted application code. The implementation is not an adversarial callback sandbox, distributed transaction coordinator or durable model registry.

## Final local Docker run

Authority: `output/next_phase/docker_v2/summary.json` (SHA-256 `4663b351b6f7b0ebbff475c3bcb671223036e7843b8403fed5c41dc8d7f80dec`). Image `sha256:0f3c7e9c9a994c13b9edecc9fbd48cc03d2a5726b1d021c650d461705e18850c`. All 54 code/lock file hashes read from the image match current local sources.

| Workload | Requests | Median ms | p95 ms | p99 ms |
| --- | ---: | ---: | ---: | ---: |
| Sequential | 100 | 2.3991 | 3.7768 | 6.2977 |
| Concurrency four | 100 | 6.0037 | 12.8961 | 37.6157 |

The service runs as UID 1000 with a read-only root filesystem, all Linux capabilities dropped, no-new-privileges, an ephemeral API key and a loopback-only published port. Missing credentials return 403; malformed alpha returns 422; the proxy fixture abstains; the supplied full-certificate fixture adapts. The sampled Docker memory usage was 91.09MiB / 7.654GiB; this is a point observation, not a peak. The smoke container was stopped after measurement.

The endpoint accepts supplied certificate inputs and does not execute a neural candidate. The new lifecycle is separately tested, not wired automatically into that HTTP endpoint. No uptime, public endpoint, independent field deployment, long-duration load, energy consumption, or production SLA has been established.

## Real neural stage profile

Authority: `output/next_phase/natural_cost_v1/summary.json`, with all 15 stage rows, the pre-execution protocol, executed-source snapshots and an independent arithmetic audit. This is a cost-only evaluation on the already-opened Cairo development city: 8,606 probe images and 8,839 evaluation images, five verified source checkpoints, three repetitions, MPS, batch size 128 and four CPU threads. No label values or protected target pixels were read.

| Stage | Median seconds per checkpoint | Executions |
| --- | ---: | ---: |
| Source checkpoint reset | 0.1531 | 15 |
| Source probe inference and BN statistics | 24.3292 | 15 |
| Tent probe adaptation | 9.6613 | 15 |
| Candidate probe inference | 6.8943 | 15 |
| Label-free feature extraction | 0.00431 | 15 |
| Source evaluation-pixel inference | 7.0374 | 15 |
| Candidate evaluation-pixel inference | 7.0649 | 15 |

The measured sum of all five shadow-candidate routing pipelines plus the city-mean prediction was **207.438, 208.238 and 203.222 seconds**, with median **207.438 seconds**. This sum excludes serving inference, cleanup and setup. A city-mean prediction alone took median 0.483 ms. Both source and candidate evaluation alternatives were timed separately; they are not an ensemble serving result. Each checkpoint is reset before every repetition. Identity/setup work took 69.097 seconds and total wall time was 902.842 seconds (15.05 minutes), including all measured alternatives and cleanup.

Process maximum RSS was 708,214,784 bytes (675.41 MiB); maximum sampled post-stage MPS driver allocation was 1,193,590,784 bytes (1,138.30 MiB), not a continuous GPU peak. Timing boundaries explicitly synchronize MPS and include the relevant HDF5 reads, normalization and transfers. There was no explicit warmup; cache state was uncontrolled following full-container identity checks. The host was not reserved exclusively: brief PDF/Word build work overlapped some repetitions. These workload-specific observations do not estimate isolated production latency, energy, a service-level objective or independent efficacy. Three repeats on one opened city are not three deployments. The timing receipt snapshots seven named modules, not the complete transitive import graph. The largest repeated point-prediction spread was 0.01450 percentage points; exact numerical replay is not claimed.

Natural calibration maintenance is a different workload: 19 cities and 95 cells took 2,193.110 seconds, plus 32.182 seconds preparation, for **37.09 minutes**. Historical source-model training and operational label acquisition are excluded. A new deployment still needs a label/calibration maintenance plan and a net-utility evaluation that includes the five shadow candidates.

## Scientific and operational limits

Per-request calibration is not adaptive repeated-use control. Validity windows, hashes, transaction caps and fail-closed behavior prevent specific implementation errors; they do not make an invalid interval valid. Fresh independent environments, justified shift assumptions, an operational label/calibration maintenance plan and net utility after compute costs remain requirements for deployment claims. Superseded lifecycle v1–v3 and Docker v1 records remain preserved rather than overwritten.

Real neural stage measurements and natural-study calibration maintenance costs are recorded separately, so this lightweight overhead cannot be mistaken for total adaptation cost.

## Running local reference service

A separate persistent local container, `kbound-nextphase-local-20260921`, was launched from the same verified image and passed production readiness on 2026-09-21. Its loopback address is `http://127.0.0.1:55939`; `/health` and `/ready` are available for inspection. The API key is stored only in the owner's private file `/Users/pratik_n/.local/share/kbound-nextphase-20260921/api.env` with mode 0600 and is excluded from the source/evidence archives. The Docker daemon must remain running. Stop with `docker stop kbound-nextphase-local-20260921`. This is a local production-mode reference service; it does not change the scientific or external-deployment scope above. Its launch receipt is `output/next_phase/local_service_v1/summary.json`.
