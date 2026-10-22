PYTHON ?= python3
VENV ?= .venv
RUNTIME ?= $(VENV)/bin/python
WORK_DIR ?= $(CURDIR)/work
TORCH_INDEX ?= https://download.pytorch.org/whl/cu121
PIP_INDEX ?= https://pypi.org/simple
RUN_ARGS ?=
DOWNLOAD_ARGS ?=
IMAGENET_VAL ?=
IMAGENET_ARCHIVE ?=

.DEFAULT_GOAL := setup
.NOTPARALLEL:
.PHONY: setup prepare download download-bert download-deit download-efficientnet \
	download-mlgcn download-tinyllama download-imagenet all bert-large deit-tiny \
	efficientnet-b0 mlgcn-coco tinyllama dry-run lfst

IMAGE_ARGS = $(if $(IMAGENET_VAL),--imagenet-val "$(IMAGENET_VAL)") \
	$(if $(IMAGENET_ARCHIVE),--imagenet-archive "$(IMAGENET_ARCHIVE)")

setup:
	$(PYTHON) -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Use Python 3.10, 3.11, or 3.12 (set PYTHON=python3.11)"'
	$(PYTHON) -m venv "$(VENV)"
	"$(VENV)/bin/python" -m pip install --index-url "$(PIP_INDEX)" --upgrade pip
	"$(VENV)/bin/python" -m pip install --index-url "$(TORCH_INDEX)" torch==2.5.1 torchvision==0.20.1
	"$(VENV)/bin/python" -m pip install --index-url "$(PIP_INDEX)" -r src/model/requirements.txt

prepare: setup download

download:
	"$(RUNTIME)" -B src/model/download.py --work-dir "$(WORK_DIR)" $(IMAGE_ARGS) $(DOWNLOAD_ARGS)

download-bert:
	"$(RUNTIME)" -B src/model/download.py --work-dir "$(WORK_DIR)" --assets bert-large $(DOWNLOAD_ARGS)

download-deit:
	"$(RUNTIME)" -B src/model/download.py --work-dir "$(WORK_DIR)" --assets deit-tiny $(DOWNLOAD_ARGS)

download-efficientnet:
	"$(RUNTIME)" -B src/model/download.py --work-dir "$(WORK_DIR)" --assets efficientnet-b0 $(DOWNLOAD_ARGS)

download-mlgcn:
	"$(RUNTIME)" -B src/model/download.py --work-dir "$(WORK_DIR)" --assets mlgcn-coco $(DOWNLOAD_ARGS)

download-tinyllama:
	"$(RUNTIME)" -B src/model/download.py --work-dir "$(WORK_DIR)" --assets tinyllama $(DOWNLOAD_ARGS)

download-imagenet:
	"$(RUNTIME)" -B src/model/download.py --work-dir "$(WORK_DIR)" --assets imagenet $(IMAGE_ARGS) $(DOWNLOAD_ARGS)

all:
	"$(RUNTIME)" -B src/model/run.py all --work-dir "$(WORK_DIR)" $(RUN_ARGS)

bert-large deit-tiny efficientnet-b0 mlgcn-coco tinyllama:
	"$(RUNTIME)" -B src/model/run.py $@ --work-dir "$(WORK_DIR)" $(RUN_ARGS)

dry-run:
	$(PYTHON) -B src/model/run.py all --work-dir "$(WORK_DIR)" --dry-run $(RUN_ARGS)

lfst:
	"$(RUNTIME)" -B src/model/lfst.py --output "$(WORK_DIR)/lut/coefficients.json"
