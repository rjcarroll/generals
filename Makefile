# ===== Dictionary Pipeline Makefile =====
# Usage:
#   make ingest        # render PDFs -> page images + manifest
#   make preprocess    # clean/deskew/etc (stub)
#   make ocr           # run OCR (stub)
#   make segment       # split entries (stub)
#   make structure     # parse fields (stub)
#   make qc            # generate QC report (stub)
#   make -j4 ingest    # run in parallel when many volumes
#   make clean         # wipe intermediates

# ---- Config (override via environment or make vars) ----
PY := python
OUT_INGEST   ?= data/interim/ingested
OUT_PREPROC  ?= data/interim/preprocessed
OUT_OCR      ?= data/interim/ocr
OUT_SEGMENT  ?= data/interim/segments
OUT_STRUCT   ?= data/processed/entries
REPORTS_DIR  ?= reports
MANIFEST     ?= $(OUT_INGEST)/manifest.csv
PDF_GLOB     ?= data/raw/*.pdf
PAGE_EXT     ?= .png
PAD          ?= 4
DPI          ?= 300
# Tessdata in-repo (traineddata + user-words/patterns)
TESSDATA_DIR := tessdata
# Preprocess uses same ranges file
PREPROCESS_RANGES := config/ocr_ranges.yaml

# knobs you can override per-run
PREPROCESS_ARGS ?= --crop-mode volume --calib-sample 64 --auto-crop-pad-pct 0.5 --crop-margins-pct 0,1.5,0,1.5 --ranges-config $(PREPROCESS_RANGES)

# OCR knobs
OCR_LANGS      ?= fra+lat     # was: fra
OCR_SPLIT_MODE ?= auto        # was: none (try 'auto' first; fall back to 'none' if you prefer)
OCR_PSM        ?= 4
OCR_OEM        ?= 1

# Tools (Python scripts / CLIs)
INGEST_SCRIPT     := src/pipeline/ingest.py            # provided
PREPROCESS_SCRIPT := src/pipeline/preprocess.py
OCR_SCRIPT        := src/pipeline/ocr.py
SEGMENT_SCRIPT    := src/pipeline/segment.py
STRUCT_SCRIPT     := src/pipeline/structure.py
QC_SCRIPT         := src/pipeline/qc.py

# Discover inputs/volumes from PDFs
PDFS    := $(wildcard $(PDF_GLOB))
VOLS    := $(notdir $(basename $(PDFS)))
VOL_DIRS:= $(addprefix $(OUT_INGEST)/,$(VOLS))

# Default target
.PHONY: all
all: ingest

# Ensure base directories exist
$(OUT_INGEST):
	@mkdir -p $@
$(OUT_PREPROC):
	@mkdir -p $@
$(OUT_OCR):
	@mkdir -p $@
$(OUT_SEGMENT):
	@mkdir -p $@
$(OUT_STRUCT):
	@mkdir -p $@
$(REPORTS_DIR):
	@mkdir -p $@

# ===== Stage 1: INGEST =====
.PHONY: ingest
ingest: $(addsuffix /.done,$(VOL_DIRS))
	@echo "[ingest] Complete → $(OUT_INGEST)"
	@echo "[ingest] Manifest → $(MANIFEST)"

# Each volume is considered done when a stamp file exists
$(OUT_INGEST)/%/.done: data/raw/%.pdf | $(OUT_INGEST) $(INGEST_SCRIPT)
	@echo ">>> Ingesting $< as volume $*"
	$(PY) $(INGEST_SCRIPT) $< --volume $* \
		--out-root $(OUT_INGEST) \
		--manifest $(MANIFEST) \
		--dpi $(DPI)
	@touch $@

# ===== Stage 2: PREPROCESS =====
.PHONY: preprocess
preprocess: $(addsuffix /.done,$(addprefix $(OUT_PREPROC)/,$(VOLS)))
	@echo "[preprocess] Complete → $(OUT_PREPROC)"

# Example rule (expects PREPROCESS_SCRIPT to read manifest + write cleaned pages per volume)
$(OUT_PREPROC)/%/.done: $(OUT_INGEST)/%/.done | $(OUT_PREPROC) $(PREPROCESS_SCRIPT)
	@echo ">>> Preprocessing volume $*"
	$(PY) $(PREPROCESS_SCRIPT) \
	  --in-root $(OUT_INGEST)/$* \
	  --out-root $(OUT_PREPROC)/$* \
	  --ext $(PAGE_EXT) \
	  $(PREPROCESS_ARGS)
	@touch $@
	
# ===== Stage 3: OCR  =====
OCR_RANGES := config/ocr_ranges.yaml
OCR_WORKERS ?= 4
VOLUMES := dict1 dict2

.PHONY: ocr
ocr:
	@for vol in $(VOLUMES); do \
		echo "Running OCR for $$vol..."; \
		TESSDATA_PREFIX=$(TESSDATA_DIR) \
		$(PY) $(OCR_SCRIPT) \
			--in-root $(OUT_PREPROC)/$$vol \
			--out-root $(OUT_OCR)/$$vol \
			--lang "$(OCR_LANGS)" \
			--psm 4 --oem 1 \
			--formats txt,tsv,hocr \
			--split-columns $(OCR_SPLIT_MODE) \
			--header-height-pct 7 --skip-existing \
			--ranges-config $(OCR_RANGES) \
			--workers $(OCR_WORKERS); \
	done

# ===== Stage 4: SEGMENT (stub) =====
.PHONY: segment
segment: $(addsuffix /.done,$(addprefix $(OUT_SEGMENT)/,$(VOLS)))
	@echo "[segment] Complete → $(OUT_SEGMENT)"

$(OUT_SEGMENT)/%/.done: $(OUT_OCR)/%/.done | $(OUT_SEGMENT) $(SEGMENT_SCRIPT)
	@echo ">>> Segment entries for volume $*"
	$(PY) $(SEGMENT_SCRIPT) --in-root $(OUT_OCR)/$* --out-root $(OUT_SEGMENT)/$*
	@touch $@

# ===== Stage 5: STRUCTURE (stub) =====
.PHONY: structure
structure: $(addsuffix /.done,$(addprefix $(OUT_STRUCT)/,$(VOLS)))
	@echo "[structure] Complete → $(OUT_STRUCT)"

$(OUT_STRUCT)/%/.done: $(OUT_SEGMENT)/%/.done | $(OUT_STRUCT) $(STRUCT_SCRIPT)
	@echo ">>> Parse structured fields for volume $*"
	$(PY) $(STRUCT_SCRIPT) --in-root $(OUT_SEGMENT)/$* --out-path $(OUT_STRUCT)/$*.jsonl
	@touch $@

# ===== Stage 6: QC (stub) =====
.PHONY: qc
qc: $(REPORTS_DIR)/qc.html
	@echo "[qc] Report → $(REPORTS_DIR)/qc.html"

$(REPORTS_DIR)/qc.html: $(addsuffix /.done,$(addprefix $(OUT_STRUCT)/,$(VOLS))) | $(REPORTS_DIR) $(QC_SCRIPT)
	$(PY) $(QC_SCRIPT) --inputs $(OUT_STRUCT)/*.jsonl --out $@

# ===== Utilities =====
.PHONY: list
list:
	@echo "PDFS:    $(PDFS)"
	@echo "VOLUMES: $(VOLS)"
	@echo "OUT_INGEST=$(OUT_INGEST)"

.PHONY: clean
clean:
	rm -rf $(OUT_INGEST) $(OUT_PREPROC) $(OUT_OCR) $(OUT_SEGMENT) $(OUT_STRUCT) $(REPORTS_DIR)

# Keep intermediates if a command fails
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
