import sys, datetime
import pandas as pd

# DEBUG is injected by main.py when the ‑d/‑‑debug flag is used; the fallback is `False` for unit tests
try:
    from main import DEBUG          # noqa: E402
except ImportError:
    DEBUG = False

def info(msg):
    '''Print informational messages to stderr so that structured output (stdout e.g. CSV) stays clean'''
    print(msg, file=sys.stderr)


def dbg(msg):
    """Print only if DEBUG is enabled (stderr, like info())."""
    try:
        from main import DEBUG        # Import lazily to avoid circular refs
    except ImportError:
        DEBUG = False
    else:
        DEBUG = DEBUG
    if DEBUG:
        print(msg, file=sys.stderr)


def warn_if_stale(df, label="Data", quiet=False):
    now = datetime.datetime.now()
    last_date = df["date"].max()

    # Skip age check on Sunday, or Monday before 9am
    if now.weekday() == 6 or (now.weekday() == 0 and now.hour < 9):
        return

    age = (now.date() - last_date.date()).days
    if DEBUG:
        info(f"⏳ Freshness check for {label}: "
             f"last date {last_date.date()}, {age} days old (limit 1 day).")
    if age > 1:
        print(f"\n⚠️  {label} data is {age} days old. Last date: {last_date.date()}")

        if quiet:
            print("Proceeding because quiet mode is enabled.")
            return

        try:
            print("Continue anyway? [Y/n] ", end="", flush=True)
            answer = input().strip().lower()
        except EOFError:
            answer = "y"

        if answer not in ("y", "yes", ""):
            print("Aborting.")
            sys.exit(1)


def to_cutoff_date(tag: str) -> pd.Timestamp:
    """
    Convert a look‑back tag ('YTD', '3M', '5Y', …) to the cutoff date.
    """
    from pandas.tseries.offsets import DateOffset
    
    today = pd.Timestamp.today().normalize()

    if tag == "YTD":
        return pd.Timestamp(today.year, 1, 1)

    number, unit = int(tag[:-1]), tag[-1].upper()
    if unit == "M":
        return today - DateOffset(months=number)
    if unit == "Y":
        return today - DateOffset(years=number)

    raise ValueError(f"Unsupported look‑back tag: {tag}")
