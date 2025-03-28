import sys
import datetime

def info(msg):
    '''Print informational messages to stderr so that structured output (stdout e.g. CSV) stays clean'''
    print(msg, file=sys.stderr)


def warn_if_stale(df, label="Data", quiet=False):
    now = datetime.datetime.now()
    last_date = df["date"].max()

    # Skip age check on Sunday, or Monday before 9am
    if now.weekday() == 6 or (now.weekday() == 0 and now.hour < 9):
        return

    age = (now.date() - last_date.date()).days
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
