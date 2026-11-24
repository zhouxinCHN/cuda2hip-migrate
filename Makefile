PYTHON ?= python3
export PYTHONPATH := $(CURDIR)/src

SAMPLES ?= third-party/cuda-samples
VECTORADD_SOURCE ?= $(SAMPLES)/Samples/0_Introduction/vectorAdd/vectorAdd.cu
SAMPLES_COMMON ?= $(SAMPLES)/Common

.PHONY: all test list probe-malloc probe-pinned probe-stream probe-managed probe-smem probe-cublas probe-texture probes sample-vectoradd install build clean

all: test

test:
	$(PYTHON) -m unittest discover -s tests -v

list:
	$(PYTHON) -m cuda2hip_migrate list

probe-malloc:
	$(PYTHON) -m cuda2hip_migrate run --probe probe_malloc_memcpy

probe-pinned:
	$(PYTHON) -m cuda2hip_migrate run --probe probe_pinned_host_mem

probe-stream:
	$(PYTHON) -m cuda2hip_migrate run --probe probe_stream_event

probe-managed:
	$(PYTHON) -m cuda2hip_migrate run --probe probe_managed_memory

probe-smem:
	$(PYTHON) -m cuda2hip_migrate run --probe probe_kernel_dynamic_smem

probe-cublas:
	$(PYTHON) -m cuda2hip_migrate run --probe probe_cublas_gemm

probe-texture:
	$(PYTHON) -m cuda2hip_migrate run --probe probe_texture_object

probes: probe-malloc probe-pinned probe-stream probe-managed probe-smem probe-cublas probe-texture

sample-vectoradd:
	$(PYTHON) -m cuda2hip_migrate run \
		--source $(VECTORADD_SOURCE) \
		--include-dir $(SAMPLES_COMMON) \
		--success-marker "Test PASSED"

install:
	$(PYTHON) -m pip install -e .

build:
	$(PYTHON) -m build

clean:
	rm -rf build dist .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +
