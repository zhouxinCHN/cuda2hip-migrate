PYTHON ?= python3
PYTHONPATH := $(CURDIR)/src
export PYTHONPATH
ARTIFACTS ?= artifacts
ARCH ?=

.PHONY: all test env list verify run-p0 fetch-samples fetch-library-samples build clean

all: test

test:
	$(PYTHON) -m unittest discover -s tests -v

env:
	$(PYTHON) -m rocm10_samples_compat check-env --strict

list:
	$(PYTHON) -m rocm10_samples_compat list-batches

verify:
	$(PYTHON) -m rocm10_samples_compat verify-checkout

run-p0:
	$(PYTHON) -m rocm10_samples_compat run-batch \
		--batch P0_baseline --artifacts $(ARTIFACTS) \
		$(if $(ARCH),--arch $(ARCH),)

fetch-samples:
	./scripts/fetch_cuda_samples.sh

fetch-library-samples:
	./scripts/fetch_cuda_library_samples.sh

build:
	$(PYTHON) -m build

clean:
	rm -rf build dist .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +
