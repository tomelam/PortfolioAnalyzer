def calculate_ppf_relative_civ(ppf_interest_rates):
    """
    Calculate the monthly current investment value (CIV) gain for PPF, accruing monthly interest and
    crediting annually.

    Parameters:
        ppf_interest_rates (pd.DataFrame): DataFrame with 'rate' column and 'date' index.

    Returns:
        pd.DataFrame: DataFrame with 'civ' column indexed by date.
    """
    # Ensure the input is a Series.
    if isinstance(ppf_interest_rates, pd.DataFrame):
        ppf_interest_rates = ppf_interest_rates["rate"]

    first_date = ppf_interest_rates.index.min()
    extended_dates = pd.date_range(
        start=first_date - pd.DateOffset(months=1),
        end=ppf_interest_rates.index.max(),
        freq="M",
    )
    monthly_rates = ppf_interest_rates.reindex(extended_dates, method="pad")

    yearly_civ = 1000.0  # Starting CIV for the financial year
    monthly_civ = yearly_civ
    accrued_interest = 0.0
    civs = []

    for i, (date, annual_rate) in enumerate(monthly_rates.items()):
        if i == 0:
            civs.append({"date": date, "civ": yearly_civ})
            continue

        monthly_interest = yearly_civ * ((annual_rate / 12) / 100)
        accrued_interest += monthly_interest
        monthly_civ = yearly_civ + accrued_interest

        if date.month == 3:
            yearly_civ += accrued_interest
            accrued_interest = 0.0

        civs.append({"date": date, "civ": monthly_civ})

    civ_df = pd.DataFrame(civs)
    civ_df["date"] = pd.to_datetime(civ_df["date"])
    civ_df.set_index("date", inplace=True)
    civ_df["civ"].fillna(method="ffill", inplace=True)

    return civ_df
