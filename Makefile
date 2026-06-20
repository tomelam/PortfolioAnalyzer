# PortfolioAnalyzer — sweep automation
#
# Targets:
#   make all       — render PNG + CSV for every portfolio in $(PORT_DIR),
#                    incrementally (only when missing or older than its TOML)
#   make summary   — one combined CSV at portfolio_metrics.csv
#   make rerender  — force-rebuild every PNG + CSV regardless of mtime
#                    (preserves any other files in outputs/; only the targets
#                    are overwritten)
#   make outputs/<name>.png   — render just that one portfolio
#   make clean     — remove only the summary CSV (portfolio_metrics.csv).
#                    NEVER touches outputs/ — historical renders are
#                    expensive to recompute (mfapi round-trips) and the
#                    user often wants to keep them for comparison.
#   make distclean — remove outputs/ wholesale. Explicit + confirmation gate.
#
# See docs/OUTPUTS.md for the rationale behind the "preserve by default" policy.
#
# Knobs:
#   PA       — portfolio-analyzer console entry point (default ./venv/bin/portfolio-analyzer)
#   PORT_DIR — portfolio TOMLs to sweep (default examples/port)
#   ARGS     — extra args passed through to portfolio-analyzer
#              e.g. ARGS="--metrics-method monthly --lookback 5Y"
#
# Examples:
#   make -j 4 all                          # 4 portfolios in parallel
#   make ARGS="--lookback 5Y" all          # all trimmed to last 5 years
#   make outputs/port-everything.png       # just one portfolio

PA       ?= ./venv/bin/portfolio-analyzer
PORT_DIR ?= examples/port
CONFIG   ?= tests/fixtures/golden_master_config.toml
ARGS     ?=

# CONFIG defaults to the staleness-bypass config so the Makefile produces
# results out-of-the-box even when the bundled data/ CSVs are months old.
# When you refresh data/reference/NIFTY Total Returns Historical Data.csv and
# data/reference/India 10-Year Bond Yield Historical Data.csv (see KANBAN "Data
# freshness"), override with CONFIG=config.toml (or unset entirely).

PORTFOLIOS := $(wildcard $(PORT_DIR)/*.toml)
PNGS       := $(patsubst $(PORT_DIR)/%.toml,outputs/%.png,$(PORTFOLIOS))

.PHONY: all summary rerender clean distclean help

help:
	@echo "PortfolioAnalyzer Makefile targets:"
	@echo "  make all       — render PNG + CSV for every $(PORT_DIR)/*.toml ($(words $(PORTFOLIOS)) found, incremental)"
	@echo "  make rerender  — force-rebuild every PNG + CSV (preserves other files in outputs/)"
	@echo "  make summary   — one combined CSV at portfolio_metrics.csv"
	@echo "  make clean     — remove portfolio_metrics.csv only (outputs/ is preserved)"
	@echo "  make distclean — remove outputs/ wholesale (asks for confirmation)"
	@echo "  make outputs/<name>.png  — render just one"
	@echo
	@echo "Knobs: PA=$(PA)  PORT_DIR=$(PORT_DIR)  ARGS=$(ARGS)"

all: $(PNGS)

# Each portfolio renders in one portfolio-analyzer invocation that emits
# both a PNG and a CSV. Make tracks the PNG; the CSV is a co-produced
# side-effect (same basename, .csv extension).
outputs/%.png: $(PORT_DIR)/%.toml
	@mkdir -p outputs
	$(PA) \
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

# Force-rebuild every PNG + CSV without deleting outputs/. Other files
# in outputs/ (historical renders, stale CSVs, sibling .assets.csv /
# .drawdowns.csv files for current portfolios) are preserved; only the
# targets named by $(PNGS) and their co-produced CSVs are overwritten.
rerender:
	$(MAKE) -B all

clean:
	rm -f portfolio_metrics.csv

distclean:
	@printf "This will rm -rf outputs/ (every cached PNG + CSV). Continue? [y/N] " && \
		read ans && [ "$$ans" = "y" ] || [ "$$ans" = "Y" ] || { echo "Aborted."; exit 1; }
	rm -rf outputs/ portfolio_metrics.csv
