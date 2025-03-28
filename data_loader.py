# Data Loader Module
# This module provides tools for loading and processing data related to
# financial portfolios, including functions to fetch NAV data, risk-free rates,
# and benchmark indices, align data to common ranges, and calculate fund
# allocations.
#
# Function Table
# 1. Top-Level Functions:
#    - get_aligned_portfolio_civs
#    - align_portfolio_civs
#    - get_benchmark_navs
#    - load_portfolio_details
#    - fetch_and_standardize_risk_free_rates
#    - align_dynamic_risk_free_rates
#    - extract_fund_allocations
#    - fetch_navs_of_mutual_fund
#    - load_ppf_interest_rates
#    - extract_fom_values
#    - align_fund_data
#    - get_dynamic_risk_free_rate
# 2. Low-Level Functions:
#    - fetch_yahoo_finance_data
#    - download_csv
#    - file_modified_within

import pandas as pd
import requests
import toml
import os
from datetime import timedelta, datetime

#import portfolio_calculator
from utils import (
    info,
    warn_if_stale,
)


def load_config_toml(config_path: str) -> dict:
    """Load general runtime settings from a config TOML file."""
    if not os.path.exists(config_path):
        return {}
    return toml.load(config_path)


def load_portfolio_toml(portfolio_path: str) -> dict:
    """Load portfolio-specific structure and fund metadata."""
    if not os.path.exists(portfolio_path):
        raise FileNotFoundError(f"Portfolio file not found: {portfolio_path}")
    return toml.load(portfolio_path)


def load_and_check_freshness(path, date_format, label, skip_age_check, quiet=False):
    df = pd.read_csv(path)

    # Attempt to parse dates
    df["date"] = pd.to_datetime(df["Date"], format=date_format, errors="coerce")

    # Standardize columns
    if "Price" in df.columns:
        df = df[["date", "Price"]].rename(columns={"Price": "value"})
    elif "Close" in df.columns:
        df = df[["date", "Close"]].rename(columns={"Close": "value"})
    else:
        raise ValueError(f"Expected column 'Price' or 'Close' in {label} file, but got: {df.columns.tolist()}")

    # Drop unparseable or missing rows
    df = df.dropna(subset=["date", "value"])

    # Strip commas from numeric strings if necessary
    if df["value"].dtype == object:
        df["value"] = df["value"].str.replace(',', '').astype(float)

    # Age check
    if not skip_age_check:
        warn_if_stale(df, label=label, quiet=quiet)

    return df


def get_aligned_portfolio_civs(portfolio):
    """
    Load and align the CIVs from each fund in a portfolio.

    Parameters:
        portfolio: portfolio allocations to each component fund and
            each fund's asset allocations

    Returns:
        pd.DataFrame: Aligned NAV data for all funds in the portfolio.
    """

    portfolio_civs = fetch_portfolio_civs(portfolio)
    aligned_civs = align_portfolio_civs(portfolio_civs)
    # Flatten the MultiIndex columns by removing the second level ('nav')
    aligned_civs.columns = aligned_civs.columns.droplevel(1)
    return aligned_civs


# Get portfolio CIVs
def fetch_portfolio_civs(portfolio):
    portfolio_civs = {
        fund["name"]: fetch_navs_of_mutual_fund(fund["url"])
        for fund in portfolio["funds"]
    }
    return portfolio_civs


# Align and combine CIV data
def align_portfolio_civs(portfolio_civs):
    """
    Align and combine CIV data for all funds to a common date range.

    Parameters:
        portfolio_civs (dict): Dictionary of fund names to DataFrames containing CIV data.

    Returns:
        pd.DataFrame: Combined CIV data aligned to a common date range.
    """
    # Determine the overlapping date range for all funds
    common_start_date = max(civ.index.min() for civ in portfolio_civs.values())
    common_end_date = min(civ.index.max() for civ in portfolio_civs.values())

    # Align each fund's CIV data to the common date range
    aligned_civs = {
        name: civ.loc[common_start_date:common_end_date]
        for name, civ in portfolio_civs.items()
    }

    # Combine aligned CIV data into a single DataFrame.
    # Starts with a dictionary and ends with a DataFrame.
    combined_civs = pd.concat({name: civ for name, civ in aligned_civs.items()}, axis=1)
    aligned_combined_civs = combined_civs.ffill()
    return aligned_combined_civs


def load_index_data(filepath, source, skip_age_check=False):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Index data file not found: {filepath}")

    df = pd.read_csv(filepath)

    if source == "investing.com":
        expected_cols = {"Date", "Price"}
        if not expected_cols.issubset(df.columns):
            raise ValueError(f"Expected columns {expected_cols} in file from investing.com, got {df.columns.tolist()}")

        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="raise")
        df = df.rename(columns={"Date": "date", "Price": "value"})

    elif source == "niftyindices.com":
        expected_cols = {"Date", "Close", "Index Name"}
        if not expected_cols.issubset(df.columns):
            raise ValueError(f"Expected columns {expected_cols} in file from niftyindices.com, got {df.columns.tolist()}")

        try:
            df["Date"] = pd.to_datetime(df["Date"], format="%d %b %Y", errors="raise")
        except Exception as e:
            raise ValueError(f"Date parsing failed for niftyindices.com data: {e}")

        df = df.rename(columns={"Date": "date", "Close": "value"})
        df = df[["date", "value"]]  # Drop other columns

    else:
        raise ValueError(f"Unrecognized data source: {source}")

    df = df.sort_values("date").reset_index(drop=True)

    if not skip_age_check:
        _warn_if_stale(df, label=source)

    return df


def get_benchmark_gain_daily(benchmark_data):
    """
    Get usefully indexed benchmark historical NAVs.

    Parameters:
        benchmark_data: pd.DataFrame containing historical data indexed by date.

    Returns:
        pd.Series: Benchmark daily returns indexed by date.
    """
    # Ensure the index (dates) is treated as a datetime column
    benchmark_data.index = pd.to_datetime(benchmark_data.index, errors="coerce").tz_localize(None)
    if "Close" in benchmark_data.columns:
        benchmark_data.rename(columns={"Close": "Close"}, inplace=True)
    # Assign the index name to "date"
    benchmark_data.index.name = "date"
    # Calculate daily returns
    benchmark_gain_daily = benchmark_data["Close"].pct_change().fillna(0)
    # Set the index name to "date" so it matches expected_result
    benchmark_gain_daily.index.name = "date"
    return benchmark_gain_daily


# Function to download data from a given URL and save it as a CSV file
def download_csv(url, output_file):
    """
    Download a CSV file from a given URL and save it locally.

    Parameters:
        url (str): The URL to download the CSV file from.
        output_file (str): The path to save the downloaded CSV file.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(output_file, "wb") as file:
            file.write(response.content)
    except requests.RequestException as e:
        raise RequestException(f"Failed to download data from {url}: {e}")


# Check if the file was modified within a specific time frame
def file_modified_within(file_path, hours):
    """
    Check if a file was modified within the last specified hours.

    Parameters:
        file_path (str): Path to the file.
        hours (int): The time frame in hours.

    Returns:
        bool: True if the file was modified within the time frame, False otherwise.
    """
    if os.path.exists(file_path):
        modification_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        return (datetime.now() - modification_time).total_seconds() < hours * 3600
    return False


def load_scss_interest_rates():
    import urllib3
    from bs4 import BeautifulSoup

    def fetch_html(url, verify_ssl=False):
        response = requests.get(url, verify=verify_ssl)
        response.raise_for_status()
        return response.text

    def find_target_table(soup):
        """
        Look for a container (either a <table> or a <tbody>) whose first row has
        at least two cells with the first cell equal to "YEAR" and the second cell containing
        "INTEREST".
        """
        for container in soup.find_all(['table', 'tbody']):
            first_row = container.find('tr')
            if first_row:
                cells = first_row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    header1 = cells[0].get_text(strip=True).lower()
                    header2 = cells[1].get_text(strip=True).lower()
                    if header1 == "year" and "interest" in header2:
                        return container
        return None

    def extract_rate_series(html):
        """
        Extracts the interest rate series from the identified table.
        Returns a list of dictionaries (one per row) using the first row as header keys.
        """
        soup = BeautifulSoup(html, 'html.parser')
        target_table = find_target_table(soup)
        if not target_table:
            raise ValueError("Target table not found.")

        header_row = target_table.find('tr')
        headers = [cell.get_text(strip=True) for cell in header_row.find_all(['td', 'th'])]

        rate_series = []
        for row in target_table.find_all('tr')[1:]:
            cells = row.find_all(['td', 'th'])
            cells_text = [cell.get_text(strip=True) for cell in cells]
            if len(cells_text) < len(headers):
                cells_text.extend([""] * (len(headers) - len(cells_text)))
            else:
                cells_text = cells_text[:len(headers)]
            rate_series.append(dict(zip(headers, cells_text)))
        return rate_series

    def parse_financial_year_start(year_str):
        """
        Given a string like "02-08-2004 to 31-03-2012", extract and parse the start date.
        Try common formats (e.g. "dd-mm-YYYY" and "dd.mm.YYYY").
        """
        start_str = year_str.split("to")[0].strip()
        for fmt in ("%d-%m-%Y", "%d.%m.%Y"):
            try:
                return pd.to_datetime(start_str, format=fmt)
            except Exception:
                continue
        # Fall back to dateutil parser with dayfirst=True.
        return pd.to_datetime(start_str, dayfirst=True, errors='coerce')

    def process_rate_series(raw_series):
        """
        Process the raw rate series list (with keys 'YEAR' and 'RATE OF INTEREST (%)')
        into a list of dictionaries with keys "date" and "interest".
        """
        processed = []
        for row in raw_series:
            try:
                start_date = parse_financial_year_start(row["YEAR"])
                interest = float(row["RATE OF INTEREST (%)"])
                processed.append({"date": start_date, "interest": interest})
            except Exception as e:
                info(f"Skipping row due to error: {e}")
        return processed

    url = "https://www.nsiindia.gov.in/(S(2xgxs555qwdlfb2p4ub03n3n))/InternalPage.aspx?Id_Pk=181"
    # Suppress SSL certificate warnings.
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    try:
        html = fetch_html(url, verify_ssl=False)
        raw_rate_series = extract_rate_series(html)
        #info("SCSS rate series:")
        #for row in raw_rate_series:
        #    info(row)
        processed_rates = process_rate_series(raw_rate_series)
    except Exception as e:
        info(f"Error: {e}")
        processed_rates = []

    # Build a DataFrame with the processed data and ensure proper dtypes.
    rate_df = pd.DataFrame(processed_rates, columns=["date", "interest"])
    rate_df["date"] = pd.to_datetime(rate_df["date"])
    rate_df = rate_df.dropna(subset=["date"])
    rate_df = rate_df.set_index("date")
    rate_df.sort_index(inplace=True)

    rate_df["interest"] = pd.to_numeric(rate_df["interest"], errors="coerce")
    rate_df = rate_df.dropna(subset=["interest"])
    return rate_df


# Load the TOML file
def load_portfolio_details(toml_file_path):
    """
    Load portfolio details from a TOML file with validation.

    Parameters:
        toml_file_path (str): Path to the TOML file.

    Returns:
        dict: Parsed portfolio details.

    Raises:
        ValueError: If the TOML data is invalid or missing required fields.
        FileNotFoundError: If the TOML file does not exist.
    """
    try:
        portfolio_details = load_portfolio_toml(toml_file_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {toml_file_path}") from e
    except toml.TomlDecodeError as e:
        raise ValueError(f"Invalid TOML format: {e}") from e

    # Validate the top-level keys
    if "label" not in portfolio_details:
        raise ValueError("Missing required top-level key: 'label'")
    
    # Check that at least one asset exists (including new asset types)
    valid_asset_keys = ["funds", "ppf", "gold", "sgb", "scss", "rec_bond"]
    if not any(key in portfolio_details for key in valid_asset_keys):
        raise ValueError("TOML file specifies no assets")
    
    errors = []

    # Validate funds if present
    if "funds" in portfolio_details:
        if not isinstance(portfolio_details["funds"], list):
            errors.append("'funds' must be a list")
        else:
            for i, fund in enumerate(portfolio_details["funds"], start=1):
                fund_id = fund.get("name", f"fund #{i}")
                required_fund_keys = ["name", "url", "allocation", "asset_allocation"]
                for key in required_fund_keys:
                    if key not in fund:
                        errors.append(f"Missing required key '{key}' in investment '{fund_id}'")
                if "allocation" in fund:
                    if not isinstance(fund["allocation"], (float, int)) or not (0 <= fund["allocation"] <= 1):
                        errors.append(f"Invalid allocation value for investment '{fund_id}': Must be between 0 and 1")
                if "asset_allocation" in fund:
                    if not isinstance(fund["asset_allocation"], dict):
                        errors.append(f"'asset_allocation' must be a dictionary for investment '{fund_id}'")
                    else:
                        required_asset_keys = ["equity", "debt", "real_estate", "commodities", "cash"]
                        for key in required_asset_keys:
                            if key not in fund["asset_allocation"]:
                                errors.append(f"Missing key in 'asset_allocation' for investment '{fund_id}': '{key}'")
                            else:
                                value = fund["asset_allocation"][key]
                                if not isinstance(value, (float, int)) or value < 0:
                                    errors.append(f"Invalid value for '{key}' in 'asset_allocation' of investment '{fund_id}': Must be a non-negative number")

    # Validate PPF if present
    if "ppf" in portfolio_details:
        ppf = portfolio_details["ppf"]
        ppf_id = ppf.get("name", "PPF section")
        if "allocation" not in ppf:
            errors.append(f"Missing required key 'allocation' in {ppf_id}")
        else:
            if not isinstance(ppf["allocation"], (float, int)) or not (0 <= ppf["allocation"] <= 1):
                errors.append(f"Invalid allocation value for {ppf_id}: Must be between 0 and 1")

    # Validate Gold if present
    if "gold" in portfolio_details:
        gold = portfolio_details["gold"]
        gold_id = gold.get("name", "Gold section")
        if "allocation" not in gold:
            errors.append(f"Missing required key 'allocation' in {gold_id}")
        else:
            if not isinstance(gold["allocation"], (float, int)) or not (0 <= gold["allocation"] <= 1):
                errors.append(f"Invalid allocation value for {gold_id}: Must be between 0 and 1")
    
    # Validate SGB if present
    if "sgb" in portfolio_details:
        sgb = portfolio_details["sgb"]
        sgb_id = sgb.get("name", "SGB section")
        if "allocation" not in sgb:
            errors.append(f"Missing required key 'allocation' in {sgb_id}")
        else:
            if not isinstance(sgb["allocation"], (float, int)) or not (0 <= sgb["allocation"] <= 1):
                errors.append(f"Invalid allocation value for {sgb_id}: Must be between 0 and 1")
    
    # Validate SCSS if present
    if "scss" in portfolio_details:
        scss = portfolio_details["scss"]
        scss_id = scss.get("name", "SCSS section")
        if "allocation" not in scss:
            errors.append(f"Missing required key 'allocation' in {scss_id}")
        else:
            if not isinstance(scss["allocation"], (float, int)) or not (0 <= scss["allocation"] <= 1):
                errors.append(f"Invalid allocation value for {scss_id}: Must be between 0 and 1")
    
    # Validate REC Bond if present
    if "rec_bond" in portfolio_details:
        rec = portfolio_details["rec_bond"]
        rec_id = rec.get("name", "REC Bond section")
        if "allocation" not in rec:
            errors.append(f"Missing required key 'allocation' in {rec_id}")
        else:
            if not isinstance(rec["allocation"], (float, int)) or not (0 <= rec["allocation"] <= 1):
                errors.append(f"Invalid allocation value for {rec_id}: Must be between 0 and 1")
        # Optionally, validate coupon if provided.
        if "coupon" in rec:
            if not isinstance(rec["coupon"], (float, int)) or rec["coupon"] <= 0:
                errors.append(f"Invalid coupon value for {rec_id}: Must be a positive number")
    
    if errors:
        all_errors = "\n".join(errors)
        raise ValueError("TOML file errors detected:\n" + all_errors)

    # Validate total allocation (this function remains unchanged)
    validate_allocations(portfolio_details)

    return portfolio_details

def validate_allocations(portfolio_details, tol=0.01):
    total_allocation = 0
    if "funds" in portfolio_details:
        total_allocation += sum(fund["allocation"] for fund in portfolio_details["funds"])
    if "ppf" in portfolio_details:
        total_allocation += portfolio_details["ppf"].get("allocation", 0)
    if "gold" in portfolio_details:
        total_allocation += portfolio_details["gold"].get("allocation", 0)
    if "sgb" in portfolio_details:
        total_allocation += portfolio_details["sgb"].get("allocation", 0)
    if "scss" in portfolio_details:
        total_allocation += portfolio_details["scss"].get("allocation", 0)
    if "rec_bond" in portfolio_details:
        total_allocation += portfolio_details["rec_bond"].get("allocation", 0)
    if abs(total_allocation - 1.0) > tol:
        raise ValueError(f"Total allocation is {total_allocation:.4f}, but it must sum to 1.00 within tolerance {tol}.")


# Fetch NAV data
def fetch_navs_of_mutual_fund(url, retries=10, timeout=20):
    """
    Fetch NAV data for a fund from the given API URL.

    Parameters:
        url (str): API endpoint for fetching NAV data.
        retries (int): Number of retry attempts on failure.
        timeout (int): Request timeout duration in seconds.

    Returns:
        pd.DataFrame: DataFrame containing NAV data indexed by date.
    """
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            if "data" not in data or not data["data"]:
                raise KeyError(
                    f"'data' key missing or empty in API response from {url}"
                )
            nav_data = pd.DataFrame(data["data"])
            nav_data["date"] = pd.to_datetime(
                nav_data["date"], dayfirst=True, errors="coerce"
            )
            nav_data["nav"] = nav_data["nav"].astype(float)
            return nav_data.set_index("date").sort_index()
        except requests.RequestException as e:
            info(
                f"[Error] Request failed for {url} (Attempt {attempt + 1}/{retries}): {e}"
            )
        except (ValueError, KeyError) as e:
            info(
                f"[Error] Data processing error for {url} (Attempt {attempt + 1}/{retries}): {e}"
            )
    raise RuntimeError(f"Failed to fetch NAV data from {url} after {retries} retries")


# Load the PPF interest rates from a CSV file
def load_ppf_interest_rates(csv_file_path="data/ppf_interest_rates.csv"):
    """
    Parameters:
        csv_file_path (str): Path to the CSV file containing PPF interest rates.

    Returns:
        pd.DataFrame: A DataFrame with columns "rate" and "date", indexed by "date" and sorted.
    """
    try:
        if csv_file_path is None:
            csv_file_path = "data/ppf_interest_rates.csv"

        # Read the CSV file into a DataFrame
        ppf_data = pd.read_csv(csv_file_path)

        # Ensure the required columns are present
        if "date" not in ppf_data.columns or "rate" not in ppf_data.columns:
            raise ValueError("CSV file must contain 'date' and 'rate' columns.")

        # Convert "date" to datetime and set it as the index
        ppf_data["date"] = pd.to_datetime(
            ppf_data["date"], format="%Y-%m-%d", errors="coerce"
        )
        ppf_data.dropna(subset=["date"], inplace=True)  # Drop rows with invalid dates
        ppf_data.set_index("date", inplace=True)

        # Ensure "rate" column is numeric
        ppf_data["rate"] = pd.to_numeric(ppf_data["rate"], errors="coerce")
        ppf_data.dropna(subset=["rate"], inplace=True)  # Drop rows with invalid rates

        # Sort the DataFrame by the date index
        ppf_data.sort_index(inplace=True)

        return ppf_data

    except Exception as e:
        raise FileNotFoundError


def calculate_gold_cumulative_gain(gold_data, portfolio_start_date):
    """
    Compute a relative cumulative gain series from a gold price series.
    Assumes that at the portfolio start date, the relative value is 1.0.
    """
    # Restrict to dates on/after portfolio start.
    gold_data = gold_data.loc[gold_data.index >= portfolio_start_date]
    if gold_data.empty:
        raise ValueError("No gold price data available after portfolio start date.")

    # Normalize: divide by the price on portfolio_start_date.
    base_price = gold_data.iloc[0]["price"]
    gold_data = gold_data.copy()
    gold_data["gold"] = gold_data["price"] / base_price  # Should be 1.0

    # Reindex to daily frequency.
    gold_data = gold_data.asfreq("D", method="ffill")  # Ensure no missing dates.

    return gold_data[["gold"]]


# Extract first-of-the-month (FOM) values
def extract_fom_values(nav_data):
    """
    Extract the first-of-the-month (FOM) values from CIV data.

    Parameters:
        nav_data (pd.DataFrame): DataFrame containing CIV data indexed by date.

    Returns:
        pd.DataFrame: FOM values.
    """
    fom_values = nav_data.loc[nav_data.index.is_month_start]
    return fom_values


# Fetch Yahoo Finance data
def fetch_yahoo_finance_data(ticker, refresh_hours=6, period="max"):
    """
    Fetch historical data from Yahoo Finance for a given ticker symbol.

    Parameters:
        ticker (str): Yahoo Finance ticker symbol.
        period (str): The time period for historical data (default is "max").
        refresh_hours (int): Number of hours before refreshing the data (default is 6).

    Returns:
        pd.DataFrame: Historical data indexed by date.
    """

    file_path = f"{ticker.replace('^', '').replace('/', '_')}.csv"
    if not file_modified_within(file_path, refresh_hours):
        info(
            f"The data for {ticker} is outdated or missing. Fetching it using yfinance..."
        )
        yf_ticker = yf.Ticker(ticker)
        raw = yf_ticker.history(period=period)
        raw.to_csv(file_path)
        info(f"Data downloaded successfully and saved to {file_path}.")
    else:
        raw = pd.read_csv(file_path)

    data = rename_yahoo_data_columns(raw)
    """
    # Convert the index to a datetime column
    data["date"] = pd.to_datetime(data.index, errors="coerce")
    data.reset_index(
        drop=True, inplace=True
    )  # Drop the old index    if "Date" not in data.columns:
    """
    assert data.index.name == "Date", "Index name mismatch: expected 'Date'"
    return data


# Load risk-free rate data
def fetch_and_standardize_risk_free_rates(file_path):
    """
    Load risk-free rate data from a CSV file and check if it's outdated based on the date content.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Risk-free rate data indexed by date.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If the file format is invalid.
        RuntimeError: If the data is outdated.
    """
    import pandas as pd
    import os
    from utils import info

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Risk-free rate file '{file_path}' not found.")

    try:
        df = pd.read_csv(file_path)

        if "observation_date" not in df.columns:
            raise ValueError("CSV must contain an 'observation_date' column.")

        # Expect exactly one other column
        rate_cols = [col for col in df.columns if col != "observation_date"]
        if len(rate_cols) != 1:
            raise ValueError(f"Expected exactly one rate column besides 'observation_date', found: {rate_cols}")

        rate_col = rate_cols[0]
        df.rename(columns={rate_col: "rate"}, inplace=True)

        df["date"] = pd.to_datetime(df["observation_date"], errors="coerce")
        df["rate"] = pd.to_numeric(df["rate"], errors="coerce") / 100

        df.dropna(subset=["date", "rate"], inplace=True)
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)

        # Freshness test
        last_date = df.index[-1]
        today = pd.Timestamp.today().normalize()
        max_allowed_delay = pd.DateOffset(days=61)
        expected_latest = (today - max_allowed_delay).replace(day=1)

        if last_date < expected_latest:
            raise RuntimeError(
                f"Risk-free rate data is outdated (last date: {last_date.date()}). "
                f"Please fetch a fresh version from the web."
            )

    except Exception as e:
        raise ValueError(f"Failed to load risk-free rate data: {e}")

    info("The risk-free-rate data appears up to date.")
    return df


# Interpolate risk-free rates to match portfolio dates
def align_dynamic_risk_free_rates(portfolio_returns, risk_free_data):
    """
    Align and interpolate risk-free rates to match portfolio return dates.

    Parameters:
        portfolio_returns (pd.Series): Portfolio daily returns.
        risk_free_data (pd.DataFrame): Risk-free rate data.

    Returns:
        pd.Series: Interpolated risk-free rates.
    """
    aligned_rates = risk_free_data.reindex(portfolio_returns.index)
    rate_series = aligned_rates["rate"].infer_objects(copy=False).interpolate(method="time")
    filled_rates = rate_series.ffill().bfill()
    # return filled_rates.mean()
    return filled_rates
