# List all portfolio configs
PORTFOLIOS := $(wildcard port/*.toml)
# Transform to list of output CSVs
METRICS := $(patsubst port/%.toml,outputs/%.metrics.csv,$(PORTFOLIOS))

# Default target: make all metrics files
all: $(METRICS)

# Rule: build metrics CSV from each TOML
outputs/%.metrics.csv: port/%.toml
	@mkdir -p outputs
	python main.py --output-dir outputs $<

# Clean up generated files
clean:
	rm -f outputs/*.metrics.csv
