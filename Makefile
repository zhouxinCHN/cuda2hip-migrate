PYTHON ?= python3
export PYTHONPATH := $(CURDIR)/src

.PHONY: all test list probe-malloc probe-pinned probe-stream probe-managed probe-smem probe-cublas probe-texture probes smoke-vectoradd sample-simpleoccupancy sample-fp16scalarproduct sample-matrixmul install build clean

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

# Smoke test of the external-source path: HIPIFY maps vectorAdd's helper
# include away and the sample calls no helper function, so this says nothing
# about Common helper support.
smoke-vectoradd:
	$(PYTHON) -m cuda2hip_migrate run --spec samples/cuda-samples-v12.0/vectorAdd.json

sample-simpleoccupancy:
	$(PYTHON) -m cuda2hip_migrate run --spec samples/cuda-samples-v12.0/simpleOccupancy.json

sample-fp16scalarproduct:
	$(PYTHON) -m cuda2hip_migrate run --spec samples/cuda-samples-v12.0/fp16ScalarProduct.json

sample-matrixmul:
	$(PYTHON) -m cuda2hip_migrate run --spec samples/cuda-samples-v12.0/matrixMul.json

install:
	$(PYTHON) -m pip install -e .

build:
	$(PYTHON) -m build

clean:
	rm -rf build dist .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +
