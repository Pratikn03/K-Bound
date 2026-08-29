# K-Bound Phase-1 provenance audit

Date: 2026-08-27
Status: partial retrospective seal; new publication-grade runs still required

## Bottom line

Phase 1 can seal the current release authority, archived source bytes, recoverable configuration
payloads, and several checkpoint or dataset snapshots. It cannot retroactively prove identities
that the historical runners never recorded. The machine-readable record is
`audits/phase1_provenance_2026_08_27/provenance_seal.json`.

The audit intentionally records the repository as a **dirty working-copy snapshot** at commit
`a7969c05409bf676eea75988a57b71f9179345e3`. Every current code file named in the audit has its own
SHA-256, but those hashes are not represented as the code that produced an old experiment.

## What is now verified

| Surface | Result | Interpretation |
|---|---:|---|
| Canonical source lineage | 106/106 original and compact sources match | Current Table 4 input bytes are internally sealed. |
| Canonical/source-manifest binding | Pass | Canonical JSON records the live source-manifest digest. |
| Recoverable legacy configurations | 10/10 full SHA-256 values recovered | Each archived `config_sha8` matches the prefix of the recomputed full digest. |
| Archived expected checkpoint hashes | 6/6 match | CIFAR-10-C SAR plus five Office-Home opportunity-audit checkpoints. |
| Current release/rerun code | 21 files hashed | Working-copy snapshot only; the tree is not clean. |
| Current-policy CIFAR family sensitivity | Artifact, runtime, analysis script, live policy/certificate, and preregistered protocol are SHA-256 bound | Retrospective six-family analysis only. Tent's ordinary intervals are positive, but the preregistered six-comparison Holm gate fails. |
| ImageNet-R materialization | 30,000 images, 2,167,990,742 bytes | Post-hoc tree SHA-256 `3f1bbfb98fe6fcaea3f2cf4ac22071330d5bee3ce3ba7ff2e86e84860ed62409`. |
| Office-Home materialization | 15,588 images, 2,186,812,856 bytes | Post-hoc tree SHA-256 `b995bd0f1ece7b589344c05d03fa61c200fb46fd24df15850d62d668998f8b66`. |
| ImageNet-R torchvision weights | 11/11 current cache files hashed | Post-hoc cache snapshot; the old run did not record full weight digests. |

The ImageNet-R and Office-Home tree hashes are useful fixed inputs for a corrected rerun. They are
not evidence that identical bytes were precommitted before the historical evaluations.

The current-policy CIFAR family-sensitivity artifact is
`experiments/kbound/results/reconciled_panels_v1/current_policy_cluster_inference.json`. It records
Python 3.12.13, NumPy 2.4.6, the analysis-script hash, and live hashes for `kga/policy.py`,
`kga/certificate.py`, and `research_lock/STRESS_GRID_MULTISEED_PROTOCOL_A_v1.yaml`. This runtime is
separate from the canonical panel's NumPy 2.4.4 reconciliation runtime. The artifact reproduces the
canonical action counts, but it remains retrospective and conditional on one archived checkpoint;
its hashes do not create independent-checkpoint or prospective evidence.

## Configuration hashes recovered without rerunning

| Artifact | Full configuration SHA-256 |
|---|---|
| Office-Home primary calibration | `361a1e8ce9ced9cff0d58ce065c9a12057c369fc4cd93e6b1ae19b3c57cbd363` |
| Office-Home primary test | `6605675d1b192cc790b645aee863675851332be2a070ede5d061e9238a769d39` |
| Office-Home replication calibration | `eb504dd6b8b86a66dbd65f68b68b035d49b6555623474f7a5cf8a11260391472` |
| Office-Home replication test | `f761540b9c6bce0cbde64161ccd6c71e262b91dd3e62101f2d23fb323c06026b` |
| iWildCam historical test | `e40faf29dd831ae5bce5f82a93af0089efb9d38b7c4d7f94eb6a5b68b4625ed8` |
| ImageNet-R Protocol-D historical manifest | `224624b1fa8f47b070a44b492971f7deab2600350ea8ca07eab4ef2605c49457` |
| RxRx1 model seed 0 | `3f579e721f6c083f6d06cb5c3071de0c10859c463fea607740a977138a293c06` |
| RxRx1 model seed 1 | `eef46aea69c6b6200705fe82f690533cd785f9e858fb8b43e3d67b16dddc1cfa` |
| RxRx1 model seed 2 | `6585f5b7a7a006fa2492bcb1b961080a43896c879cb08c6db6590a6ec65cda5c` |
| Camelyon17 archived diagnostic | `884129ba8cb17cc551f8304f79b7108a160d6f15cd7bb33d7a9b383b1ff45410` |

These hashes bind the exact serialized configuration objects. They do not fill absent dataset,
checkpoint, or code identities.

## What remains impossible to seal retrospectively

1. Most historical natural-shift artifacts contain no dataset/population content hash.
2. Most historical natural-shift artifacts contain no checkpoint file or tensor-state hash.
3. Stream or sampling seeds cannot be converted into independent model seeds after execution.
4. The ImageNet-R v0.5 manifest lists seeds 0--2, while the canonical source set also consumes
   seed-3 files. It also lacks native-transform and full model-weight receipts.
5. The iWildCam archive does not bind a complete population or the official WILDS scorer.
6. ImageNet-C, PACS, Camelyon17, iWildCam, and RxRx1 still lack publication-grade historical
   content seals for the populations used.
7. Two required CIFAR-10-C arrays (`defocus_blur.npy` and `contrast.npy`) are currently macOS
   dataless placeholders. They were not materialized merely to make this audit appear complete.
8. The iWildCam `f0_resnet50_erm_seed0_last.pt` file is also dataless and is not assigned the
   misleading SHA-256 of an empty byte stream.

These gaps require restoration or new runs that write the hardened record schema at execution
time. Deleting old artifacts, recomputing a digest today, or renaming an opened test set cannot
make the historical evaluation prospective.

## Rebuild command

The audit is read-only apart from its generated JSON:

```bash
KBOUND_OFFICEHOME_ROOT=/path/to/office-home \
PYTHONDONTWRITEBYTECODE=1 python3 \
  docs/research/kbound/scripts/audit_phase1_provenance.py
```

The final outer release checksum was generated after the claim ledger, result manifests, PDFs, and
DOCX stopped changing and is stored in `KBOUND_RELEASE_SHA256SUMS.txt`. That byte seal is not a
substitute for the historical dataset/checkpoint identities listed as unavailable above.
