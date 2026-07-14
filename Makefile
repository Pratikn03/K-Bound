PYTHON ?= python3
PAPER_DIR := docs/research/kbound
DASHBOARD_DIR := $(PAPER_DIR)/dashboard
EDGE_DIR := $(PAPER_DIR)/edge

.PHONY: install install-research test snapshot dashboard paper formal physical-preflight verify-fast multiseed-preflight multiseed-plan multiseed-run multiseed-status multiseed-analyze

install:
	$(PYTHON) -m pip install -e ".[test]"

install-research:
	$(PYTHON) -m pip install -e ".[research,test]"

test:
	$(PYTHON) -m pytest -q

snapshot:
	$(PYTHON) $(PAPER_DIR)/scripts/build_dashboard_snapshot.py

dashboard: snapshot
	cd $(DASHBOARD_DIR) && npm ci && npm run build

paper:
	cd $(PAPER_DIR) && latexmk -pdf -interaction=nonstopmode -halt-on-error kbound_short.tex
	cp $(PAPER_DIR)/kbound_short.pdf $(PAPER_DIR)/kbound_short_final_draft.pdf

formal:
	cd $(PAPER_DIR)/formal && bash build.sh

physical-preflight:
	$(PYTHON) $(EDGE_DIR)/scripts/preflight_r2.py
	$(PYTHON) $(EDGE_DIR)/scripts/13_check_publication_gate.py --strict

multiseed-preflight:
	bash $(PAPER_DIR)/scripts/kbtrain.sh preflight

multiseed-plan:
	bash $(PAPER_DIR)/scripts/kbtrain.sh plan

multiseed-run:
	bash $(PAPER_DIR)/scripts/kbtrain.sh run --yes

multiseed-status:
	bash $(PAPER_DIR)/scripts/kbtrain.sh status

multiseed-analyze:
	bash $(PAPER_DIR)/scripts/kbtrain.sh analyze

verify-fast: test snapshot
	cd $(DASHBOARD_DIR) && npm ci && npm run build
