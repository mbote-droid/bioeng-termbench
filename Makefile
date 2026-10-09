# Every workflow is one command. All targets run inside the locked uv environment.
SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c
.DEFAULT_GOAL := help

UV ?= uv
RUN := $(UV) run --frozen
VERSION := $(shell $(RUN) python -c "import bioengbench; print(bioengbench.__version__)")
REGISTRY ?= ghcr.io/mbote-droid

.PHONY: help setup lint typecheck test test-fast check build images clean validate validate-quick redteam mutation gpu-bench

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup:  ## create the locked environment (uv.lock)
	$(UV) sync --frozen

lint:  ## ruff lint + format check, shellcheck
	$(RUN) ruff check src tests tasks validation gpu agent_eval
	$(RUN) ruff format --check src tests
	$(RUN) shellcheck tasks/*/solution/solve.sh tasks/*/tests/test.sh scripts/*.sh

typecheck:  ## mypy --strict on the package
	$(RUN) mypy

test-fast:  ## unit tests and negative controls only
	$(RUN) pytest -m "not slow"

test:  ## full suite, including reference solutions on dev seeds (Octave if installed)
	$(RUN) pytest

check: lint typecheck test  ## everything CI runs

build:  ## wheel for the grader image
	$(UV) build --wheel --out-dir dist

images: build  ## build agent and grader base images
	docker build -f docker/agent-base.Dockerfile -t $(REGISTRY)/bioengbench-agent-base:$(VERSION) .
	docker build -f docker/grader-base.Dockerfile -t $(REGISTRY)/bioengbench-grader-base:$(VERSION) .

SNAKEMAKE ?= uvx --from snakemake==9.11.6 snakemake
JOBS ?= 4

validate:  ## full grader validation campaign (probes, mutation, red team) -> docs/VALIDATION_REPORT.md
	$(SNAKEMAKE) -s workflow/Snakefile -c $(JOBS)

validate-quick:  ## campaign without the red team (no root needed)
	$(SNAKEMAKE) -s workflow/Snakefile -c $(JOBS) --config redteam=False

redteam:  ## attacks on the sandbox (needs root and a `sandbox` user)
	$(RUN) python validation/campaign.py redteam

mutation:  ## grader mutation testing for one task: make mutation TASK=t03-glucose-closed-loop
	$(RUN) python validation/grader_mutation.py $(TASK)

gpu-bench:  ## 3D bioheat CPU/GPU benchmark (GPU parts need CuPy and a CUDA device)
	$(RUN) python gpu/benchmark.py

clean:
	rm -rf dist results .pytest_cache .mypy_cache .ruff_cache .hypothesis
