# PortfolioAnalyzer — sweep automation
#
# Targets:
#   make all       — render PNG + CSV for every portfolio in $(PORT_DIR),
#                    incrementally (only when missing or older than its TOML)
#   make summary   — one combined CSV at reports/portfolio_metrics.csv
#   make rerender  — force-rebuild every PNG + CSV regardless of mtime
#                    (preserves any other files in outputs/; only the targets
#                    are overwritten)
#   make outputs/<name>.png   — render just that one portfolio
#   make clean     — remove only the summary CSV (reports/portfolio_metrics.csv).
#                    NEVER touches outputs/ — historical renders are
#                    expensive to recompute (mfapi round-trips) and the
#                    user often wants to keep them for comparison.
#   make distclean — remove outputs/ wholesale. Explicit + confirmation gate.
#
# See docs/OUTPUTS.md for the rationale behind the "preserve by default" policy.
#
#   make test      — the offline suite (528 tests, ~27 s). A git pre-commit hook runs
#                    this target, so it is the one to keep green.
#   make test-network — the full suite including the live niftyindices and VRO
#                    browser tests. CLAUDE.md § "The merge gate, concretely" makes
#                    this the gate before a merge, not before every commit.
#
# Knobs:
#   PA       — portfolio-analyzer console entry point (default ./venv/bin/portfolio-analyzer)
#   PY       — interpreter for the tests (default ./venv/bin/python, the same one
#              ./pa runs on, so the suite validates the interpreter the code uses)
#   PORT_DIR — portfolio TOMLs to sweep (default examples/port)
#   ARGS     — extra args passed through to portfolio-analyzer
#              e.g. ARGS="--metrics-method monthly --lookback 5Y"
#
# Examples:
#   make -j 4 all                          # 4 portfolios in parallel
#   make ARGS="--lookback 5Y" all          # all trimmed to last 5 years
#   make outputs/port-everything.png       # just one portfolio

PA       ?= ./venv/bin/portfolio-analyzer
PY       ?= ./venv/bin/python
PORT_DIR ?= examples/port
REPORT   ?= reports/portfolio_metrics.csv
CONFIG   ?= tests/fixtures/golden_master_config.toml
ARGS     ?=

# CONFIG defaults to the staleness-bypass config so the Makefile produces
# results out-of-the-box even when the bundled data/ CSVs are months old.
# When you refresh data/reference/NIFTY Total Returns Historical Data.csv and
# data/reference/India 10-Year Bond Yield Historical Data.csv (see KANBAN "Data
# freshness"), override with CONFIG=config.toml (or unset entirely).

PORTFOLIOS := $(wildcard $(PORT_DIR)/*.toml)
PNGS       := $(patsubst $(PORT_DIR)/%.toml,outputs/%.png,$(PORTFOLIOS))

.PHONY: all summary rerender clean distclean help test test-network

## test — the offline suite, on the interpreter ./pa itself runs on.
## pyproject.toml sets testpaths = ["tests"] and leaves `network` deselected, so
## this needs no path or marker argument. Named `test` because that is the target a
## git pre-commit hook looks for; without it these 528 tests were reachable only by
## knowing the incantation, and a suite nothing can run is a suite nobody runs.
test:
	$(PY) -m pytest -q

## test-network — everything, live calls included. The pre-merge gate.
test-network:
	$(PY) -m pytest -q -m "not network or network"

help:
	@echo "PortfolioAnalyzer Makefile targets:"
	@echo "  make all       — render PNG + CSV for every $(PORT_DIR)/*.toml ($(words $(PORTFOLIOS)) found, incremental)"
	@echo "  make rerender  — force-rebuild every PNG + CSV (preserves other files in outputs/)"
	@echo "  make summary   — one combined CSV at $(REPORT)"
	@echo "  make clean     — remove $(REPORT) only (outputs/ is preserved)"
	@echo "  make distclean — remove outputs/ wholesale (asks for confirmation)"
	@echo "  make test      — the offline suite (what the git pre-commit runs)"
	@echo "  make test-network — the full suite, live calls included (pre-merge gate)"
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

summary: $(REPORT)

$(REPORT): $(PORTFOLIOS)
	scripts/run_all_metrics_to_csv.sh -o $@

# Force-rebuild every PNG + CSV without deleting outputs/. Other files
# in outputs/ (historical renders, stale CSVs, sibling .assets.csv /
# .drawdowns.csv files for current portfolios) are preserved; only the
# targets named by $(PNGS) and their co-produced CSVs are overwritten.
rerender:
	$(MAKE) -B all

clean:
	rm -f $(REPORT)

distclean:
	@printf "This will rm -rf outputs/ (every cached PNG + CSV). Continue? [y/N] " && \
		read ans && [ "$$ans" = "y" ] || [ "$$ans" = "Y" ] || { echo "Aborted."; exit 1; }
	rm -rf outputs/ $(REPORT)
