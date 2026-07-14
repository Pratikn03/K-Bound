# K-Bound Research Dashboard

The dashboard is a local, researcher-facing view of theorem scope, promoted
results, provenance, and physical-study readiness.

## Build

From the repository root:

~~~bash
bash docs/research/kbound/scripts/build_dashboard.sh
~~~

This compiles the TypeScript interface and regenerates data/snapshot.json from
paper/generated/kbound_result_manifest.json and the active edge result tree.
Legacy archives are not dashboard inputs.

## Serve

~~~bash
python3 -m http.server 8765 --directory docs/research/kbound
~~~

Open http://127.0.0.1:8765/kbound_dashboard.html.

Use localhost rather than opening the HTML directly because browser camera
access and JSON loading require a secure local origin.

## Camera Boundary

The browser camera is only a connectivity preview. It does not write study
frames, labels, or decisions.

Publication evidence must be captured by the locked edge scripts under
edge/artifacts_real/raw/S01 through S10. The dashboard remains pending until
experiments/kbound/results/edge_real_phone_v1/publication_gate.json exists and
reports passed: true.

## Data Contract

The snapshot must include exact paper-manifest provenance, controlled
beats-both tracks, natural no-harm tracks, diagnostic and incomplete tracks,
theorem scope with formalization caveats, and physical session blockers.

~~~bash
python -m pytest docs/research/kbound/dashboard/tests -q
~~~

