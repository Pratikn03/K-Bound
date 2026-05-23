# Phase 1.1 Reproduction Commands

## Step 0 — branch + archive
```bash
git checkout -b fix/elara-phase1-1-pdf-source-consistency
mkdir -p docs/research/archive/pre_phase1_1/tables docs/research/archive/pre_phase1_1/output
cp docs/research/PAPER_DRAFT_v1.tex docs/research/archive/pre_phase1_1/PAPER_DRAFT_v1.before_p11.tex
cp docs/research/THESIS_CHAPTER_v1.tex docs/research/archive/pre_phase1_1/THESIS_CHAPTER_v1.before_p11.tex
cp output/pdf/PAPER_DRAFT_v1.pdf docs/research/archive/pre_phase1_1/output/PAPER_DRAFT_v1.before_p11.pdf
cp output/pdf/THESIS_CHAPTER_v1.pdf docs/research/archive/pre_phase1_1/output/THESIS_CHAPTER_v1.before_p11.pdf
cp -r docs/research/tables docs/research/archive/pre_phase1_1/tables/
```

## Step 4 — canonical cleanup
```bash
PYTHONPATH=src .venv/bin/python src/scripts/phase1_1_canonical_cleanup.py
```

## Step 6 — regenerate comparator-aware tables
```bash
PYTHONPATH=src .venv/bin/python src/scripts/emit_rga_plus_ablation.py
PYTHONPATH=src .venv/bin/python src/scripts/emit_milestone1_comparison.py
```

## Step 11 — full clean rebuild
```bash
rm -f .tex_build/*.aux .tex_build/*.log .tex_build/*.out .tex_build/*.bbl .tex_build/*.blg
rm -f .tex_build_thesis/*.aux .tex_build_thesis/*.log .tex_build_thesis/*.out
./scripts/rebuild_paper.sh
cp output/pdf/PAPER_DRAFT_v1.pdf output/pdf/PAPER_DRAFT_PHASE1_1_VERIFIED.pdf
cp output/pdf/THESIS_CHAPTER_v1.pdf output/pdf/THESIS_CHAPTER_PHASE1_1_VERIFIED.pdf
```

## Step 12+14 — validators
```bash
PYTHONPATH=src .venv/bin/python src/scripts/validate_phase1_1_pdf_claims.py
PYTHONPATH=src:. .venv/bin/python -m pytest tests/ -q
```
