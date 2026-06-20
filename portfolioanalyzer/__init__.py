"""PortfolioAnalyzer — Indian portfolio analysis (backtests, metrics, stress tests).

Flat package: all formerly top-level modules (main, metrics, utils, data_loader,
the loaders/ and timeseries/ subpackages, the asset/calc/reporting helpers) live
here under one importable namespace. Canonical entry points:

    ./pa <portfolio.toml> ...          # bundled-venv wrapper
    python -m portfolioanalyzer.main   # equivalent module invocation
"""
