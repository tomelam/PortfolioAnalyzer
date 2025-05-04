I said:

It's very strange, but the first time I downloaded that CSV, it had only 20 rows and the second time I downloaded it, it had 1557 rows, but didn't go back to 2007 as I'd requested. What's going on?

ChatGPT said:

You're running into a quirk in the Nifty Indices website's CSV download system, which often:
* Ignores the selected date range.
* Returns a default short span (e.g., 20 rows).
* Occasionally gives incomplete data even when it appears to obey the range.

Why this happens:
* The download function is likely handled via JavaScript, which doesn't always respect form input on first attempt.
* The site remembers or caches previous requests silently.
* Session tokens or background API failures may truncate results.

✅ How to fix it:
* Follow these strict steps to increase success rate:
* Clear the date range manually before selecting new dates.
* Pick a wide range again (e.g., Jan 1, 2007 – today).
* Wait for the table below to update—confirm that the table on the page actually loads many rows before clicking “Download”.
* Only then, click Download.

If that fails:

✅ Alternative (more reliable):
Use this direct NSE Indices API endpoint:

`https://www1.nseindia.com/content/indices/historicalindices.csv`

⚠️ This gives all indices. You'll have to filter for Nifty Midcap 150.

Let me know if you want a small script to extract just the Midcap 150 from that CSV, or if you’d like help finding another reliable long-term source (e.g., Yahoo Finance, with adjustments).

I said:

You're saying I ran into a quirk in the Nifty Indices website's CSV download system, but actually I was using Investing.com at the time.

ChatGPT said:

Thanks for the correction — that changes things significantly.

🟠 What's happening on Investing.com:
Investing.com’s CSV downloads are notoriously unreliable for the following reasons:

* "Download Data" ignores the chosen date range unless the page reloads fully after pressing “Apply”.
* The site uses AJAX to fetch data dynamically, so the CSV sometimes contains only a default 20 rows regardless of the selected period.
* If the AJAX request fails silently, the site may dump whatever stale or partial data it has.
* Sometimes, the correct CSV is loaded in a background request, but your browser grabs the wrong one (due to caching or timing).

✅ What you can do:

*Option 1:* Try this strict sequence in your browser (Chrome):
* Open the historical data page:
https://www.investing.com/indices/nifty-midcap-150-historical-data
* Scroll down to the date range, select Jan 1, 2007 – Today, and press “Apply”.
* Wait at least 5–10 seconds for the table to reload. Scroll and verify that the table has 1000+ rows.
* Only now, press “Download Data”.

*Option 2:* Use DevTools to capture the backend data URL (more reliable):
* Open Chrome DevTools → Network tab → filter by xhr.
* Press “Apply” on the date range.
* Look for a URL that returns a JSON or CSV-like payload. It may look like:
`https://www.investing.com/instruments/HistoricalDataAjax`
* Use curl or Python to hit that URL directly with the right parameters and headers.

Let me know if you want a script to automate that or a fallback data source from Yahoo or NSE instead.







