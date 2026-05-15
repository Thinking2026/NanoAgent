PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
BIN_DIR ?= bin
BINARY_NAME ?= nanoagent
BINARY_PATH := $(BIN_DIR)/$(BINARY_NAME)

.PHONY: install install-chromadb install-dev run check compile test test-check

install:
	@$(PIP) install -r requirements.txt
	@echo "Installed runtime dependencies."

install-chromadb:
	@$(PIP) install -r requirements-chromadb.txt
	@echo "Installed ChromaDB dependencies."

install-dev:
	@$(PIP) install -r requirements-dev.txt
	@echo "Installed development dependencies."

run:
	@$(PYTHON) src/main.py

check:
	@$(PYTHON) -m compileall src/ >/dev/null
	@echo "Compile check passed."

compile: check
	@mkdir -p $(BIN_DIR)
	@printf '%s\n' '#!/bin/sh' \
	'SCRIPT_DIR="$$(CDPATH= cd -- "$$(dirname -- "$$0")" && pwd)"' \
	'PROJECT_ROOT="$$(CDPATH= cd -- "$$SCRIPT_DIR/.." && pwd)"' \
	'cd "$$PROJECT_ROOT" || exit 1' \
	'exec "$${PYTHON:-python3}" "$$PROJECT_ROOT/src/main.py" "$$@"' \
	> $(BINARY_PATH)
	@chmod +x $(BINARY_PATH)
	@echo "Built $(BINARY_PATH)"

test-check:
	@$(PYTHON) -m compileall tests/ >/dev/null
	@echo "Test compile check passed."

test: test-check
	@$(PYTHON) -m pytest tests/ -v
