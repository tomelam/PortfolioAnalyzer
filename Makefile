# PortfolioAnalyzer — sweep automation
#
# Targets:
#   make all       — render PNG + CSV for every portfolio in $(PORT_DIR)
#   make summary   — one combined CSV at portfolio_metrics.csv
#   make outputs/<name>.png   — render just that one portfolio
#   make clean     — remove generated outputs/
#
# Knobs:
#   PYTHON   — Python interpreter (default ./venv/bin/python)
#   PORT_DIR — portfolio TOMLs to sweep (default port)
#   ARGS     — extra args passed through to main.py
#              e.g. ARGS="--metrics-method monthly --lookback 5Y"
#
# Examples:
#   make -j 4 all                          # 4 portfolios in parallel
#   make ARGS="--lookback 5Y" all          # all trimmed to last 5 years
#   make outputs/port-everything.png       # just one portfolio

PYTHON   ?= ./venv/bin/python
PORT_DIR ?= port
CONFIG   ?= tests/fixtures/golden_master_config.toml
ARGS     ?=

# CONFIG defaults to the staleness-bypass config so the Makefile produces
# results out-of-the-box even when the bundled data/ CSVs are months old.
# When you refresh data/NIFTY Total Returns Historical Data.csv and
# data/India 10-Year Bond Yield Historical Data.csv (see KANBAN "Data
# freshness"), override with CONFIG=config.toml (or unset entirely).

PORTFOLIOS := $(wildcard $(PORT_DIR)/*.toml)
PNGS       := $(patsubst $(PORT_DIR)/%.toml,outputs/%.png,$(PORTFOLIOS))

.PHONY: all summary clean help

help:
	@echo "PortfolioAnalyzer Makefile targets:"
	@echo "  make all       — render PNG + CSV for every $(PORT_DIR)/*.toml ($(words $(PORTFOLIOS)) found)"
	@echo "  make summary   — one combined CSV at portfolio_metrics.csv"
	@echo "  make clean     — remove generated outputs/"
	@echo "  make outputs/<name>.png  — render just one"
	@echo
	@echo "Knobs: PYTHON=$(PYTHON)  PORT_DIR=$(PORT_DIR)  ARGS=$(ARGS)"

all: $(PNGS)

# Each portfolio renders in one main.py invocation that emits both a
# PNG and a CSV. Make tracks the PNG; the CSV is a co-produced
# side-effect (same basename, .csv extension).
outputs/%.png: $(PORT_DIR)/%.toml
	@mkdir -p outputs
	$(PYTHON) main.py \
		--config $(CONFIG) \
		--quiet \
		--disable-plot-display \
		--output-snapshot \
		--output-csv \
		--output-dir outputs \
		$(ARGS) \
		$<

summary: portfolio_metrics.csv

portfolio_metrics.csv: $(PORTFOLIOS)
	scripts/run_all_metrics_to_csv.sh -o $@

clean:
	rm -rf outputs/ portfolio_metrics.csv
