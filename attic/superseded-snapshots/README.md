# attic/superseded-snapshots/

Older/redundant CSV snapshots, parked (not deleted) for provenance. Each is
superseded by a still-live file under `data/reference/`. Nothing here is loaded.

| File | Superseded by | Why parked |
|---|---|---|
| `NIFTRI.csv` | `data/reference/NIFTY Total Returns Historical Data.csv` | Same schema, older NIFTY TRI snapshot (ends 2025-03-21 vs 2025-05-02; 4492 vs 4537 rows). Poorly named (`NIFTRI` = NIFTY TRI). |
| `rbi_91day_tbills.csv` | `data/reference/rbi_91day_tbills_from_dbie.csv` | Truncated 2022–2024 slice (127 rows) of the fuller DBIE series (1677 rows, 1993→). |
| `NIFTY MIDCAP 150_Historical_PR_01012007to27032025.csv` | `data/reference/NIFTY MIDCAP 150_Historical_PR_01012005to24032025.csv` | Redundant Midcap-150 snapshot; the 2005-start file has longer coverage. |
| `NIFTY MIDCAP 150_Historical_PR_01012007to28032025.csv` | `data/reference/NIFTY MIDCAP 150_Historical_PR_01012005to24032025.csv` | Redundant Midcap-150 snapshot (differs only by export date). |
