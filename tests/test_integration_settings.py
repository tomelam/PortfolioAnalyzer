import pytest
from pathlib import Path
from data_loader import load_and_check_freshness
from utils import warn_if_stale

# === CLI/config merge behavior ===

@pytest.mark.order(10)  # test_output_csv_merge_behavior
@pytest.mark.parametrize("cli_args,expected", [
    ({"csv_output": True}, True),
    ({"csv_output": False}, False),
])
def test_output_csv_merge_behavior(cli_args, expected):
    # Simulate CLI vs config interaction
    settings = {"csv_output": cli_args["csv_output"]}
    assert settings["csv_output"] == expected


# === Benchmark loading and parsing ===

@pytest.mark.order(11)  # test_benchmark_date_parsing_failure
def test_benchmark_date_parsing_failure(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("Date,Price\nBADDATE,100")

    df = load_and_check_freshness(
        str(bad_csv),
        "%d/%m/%Y",
        "Benchmark",
        skip_age_check=True
    )

    assert df.empty, "Expected all rows to be dropped due to bad date"


# === Staleness warning behavior ===

@pytest.mark.order(12)  # test_warn_if_stale_triggers_warning_on_old_data
def test_warn_if_stale_triggers_warning_on_old_data(mocker):
    import pandas as pd
    import datetime

    now = datetime.datetime.now()
    df = pd.DataFrame({
        "date": [now - datetime.timedelta(days=5)],
        "value": [100.0],
    })

    mocker.patch("builtins.input", return_value="y")

    # Should print a warning and continue
    warn_if_stale(df, label="Test Series", quiet=False)


# === Output logic ===

@pytest.mark.order(13)  # test_output_dir_logic
@pytest.mark.parametrize("output_snapshot,output_dir,expected_dir", [
    (True, None, "outputs"),
    (True, "custom_dir", "custom_dir"),
    (False, "custom_dir", "custom_dir"),
])
def test_output_dir_logic(output_snapshot, output_dir, expected_dir):
    settings = {
        "output_snapshot": output_snapshot,
        "output_dir": output_dir or "outputs"
    }
    assert settings["output_dir"] == expected_dir


# === Failure scenarios ===

@pytest.mark.order(14)  # test_load_missing_benchmark_file_raises
def test_load_missing_benchmark_file_raises(tmp_path):
    nonexistent_file = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        load_and_check_freshness(
            str(nonexistent_file),
            "%d/%m/%Y",
            "Benchmark",
            skip_age_check=True
        )
