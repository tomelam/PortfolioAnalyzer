# tests/fixtures/cli_configs/

Hand-test config fixtures exercising the CLI output-flag combinations
(`output_csv` × `output_snapshot` × `show_plot` × `output_dir`), plus
`no_benchmark_config.toml` (benchmark disabled). They are **not** user-facing
configs and are loaded by no automated test — they exist for manual CLI smoke
runs, e.g.:

```
./pa port/port-1.toml --config tests/fixtures/cli_configs/output_csv-output_snapshot-show_plot.toml --allow-stale
```

Real, documented user configs live in `config/` (`example_config.toml`,
`mid-cap_config.toml`), which the `tests/unit/test_docs_consistency.py` guards validate.
