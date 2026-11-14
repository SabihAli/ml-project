# cv_service setup: dependencies and the gitignored runtime assets of a fresh checkout.
#
#   make setup      venv + dependencies + calibration checkpoints + exports (TensorRT engines if a GPU is found)
#   make dataset    SoccerNet GSR split(s) into data/gsr/ (default: valid, ~10 GB; SPLITS="train valid test" for all)
#                   no password needed; `make data` is an alias
#   make check      pytest
#   make profile    live-path latency profile (SEQ=<GSR sequence folder with img1/>)
#
# Works with GNU make on Linux and on Windows (cmd or Git Bash): every recipe is one Python call.
# GPU = an NVIDIA GPU visible to nvidia-smi. On a GPU host, torch comes from the CUDA wheel index
# (CUDA_TAG, default cu124), onnxruntime is swapped for onnxruntime-gpu, tensorrt is installed and the
# TensorRT engines are built (they are per GPU/driver/TRT version, so never committed).

ifeq ($(OS),Windows_NT)
SYS_PY ?= python
PY := .gsr_venv/Scripts/python.exe
else
SYS_PY ?= python3
PY := .gsr_venv/bin/python
endif

CUDA_TAG ?= cu124
# TensorRT 10 for CUDA 12 (matches CUDA_TAG): TensorRT 11 removed weakly-typed FP16 builds (BuilderFlag.FP16)
TRT_SPEC ?= tensorrt-cu12>=10,<11
DEVICE   ?= cuda:0
SPLITS   ?= valid
SEQ      ?=
GPU := $(shell $(SYS_PY) scripts/setup/setup_assets.py gpu)

.PHONY: all setup deps checkpoints export dataset data check profile help

all: setup

help:
	@echo targets: setup deps checkpoints export dataset check profile. GPU detected: $(GPU)

setup: deps checkpoints export
	@echo setup done. GPU: $(GPU)

$(PY):
	$(SYS_PY) -m venv .gsr_venv
	$(PY) -m pip install --upgrade pip

deps: $(PY)
ifeq ($(GPU),1)
	$(PY) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/$(CUDA_TAG)
endif
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e gsr_pipeline/nbjw_calib
ifeq ($(GPU),1)
	$(PY) -m pip uninstall -y onnxruntime tensorrt tensorrt_cu13 tensorrt_cu13_libs tensorrt_cu13_bindings
	$(PY) -m pip install "$(TRT_SPEC)"
	# onnxruntime and onnxruntime-gpu share the onnxruntime/ folder: uninstalling the CPU build
	# above deletes it, and pip skips an already-installed onnxruntime-gpu, so force a reinstall.
	$(PY) -m pip install --force-reinstall --no-deps onnxruntime-gpu
endif

checkpoints: $(PY)
	$(PY) scripts/setup/setup_assets.py checkpoints

export: checkpoints
ifeq ($(GPU),1)
	$(PY) scripts/path_b/calibration/export_nbjw.py --build_engine
	$(PY) scripts/setup/setup_assets.py engines --device $(DEVICE)
else
	$(PY) scripts/path_b/calibration/export_nbjw.py
	@echo No NVIDIA GPU: NBJW ONNX exported, TensorRT engines skipped.
endif

dataset: $(PY)
	$(PY) -m pip install SoccerNet
	$(PY) scripts/setup/setup_assets.py data --splits $(SPLITS)

data: dataset

check: $(PY)
	$(PY) -m pytest tests -q

profile: $(PY)
	$(PY) scripts/pipeline_improvement_1/phase3/profile_latency.py --sequence_dir $(SEQ) --device $(DEVICE) --calib_backend auto
