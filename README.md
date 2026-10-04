# NEPSE Sector-Wise Screener & Adaptive Paper-Trading Engine

A Python system that automates data collection from Nepal's stock exchange
(nepalstock.com), scores stocks with a layered technical/sentiment model,
and forward-tests trading strategies as simulated ("paper") trades — with
every result rendered into one self-contained HTML report. Built for
personal research; **no real orders are ever placed anywhere in this
codebase.**

> **This repo ships with a sample dataset.** `sample_report.html` and the
> screenshots in `screenshots/` were produced by running the real,
> unmodified pipeline against a small generated dataset (`demo/`) —
> fictional tickers, fictional broker names, fictional prices — so the
> report you see is genuine output of this code, not a mockup. See
> [Demo data](#demo-data) below.

## What it does

- **Scrapes daily price history and broker-wise trade data (the "floorsheet")**
  from nepalstock.com via Playwright browser automation. The site has no
  public API, requires a live browser session (cookies + a rotating auth
  token), and actively resists plain scripted requests — this handles that,
  including automatic re-authentication if a long paginated download
  outlives its session token.
- **Computes a layered score per stock**: a technical layer (RSI, MACD,
  volume ratio, momentum), a sentiment layer (market breadth + sector
  relative strength), and an optional fundamentals layer (P/E, EPS, dividend
  history, when available) — blended with weights that redistribute
  automatically when a layer has no data, instead of faking a neutral score.
- **A separate, deterministic 100-point rule-based leaderboard**: sector
  rotation (20 pts) + broker-concentration detection (40 pts) + late-session
  breakout detection (40 pts), reading directly from the day's floorsheet.
- **An adaptive paper-trading engine** (`adaptive_engine.py`): runs an
  immutable baseline strategy plus labelled experimental "shadow" strategies
  in parallel, paper-tracks every signal through entry and exit (stop-loss,
  partial profit-taking, EMA exit, distribution-warning exit, a holding-
  period cap), applies a realistic local brokerage/tax fee model, and writes
  daily and weekly performance reviews — plus a proposed strategy change
  for a human to review, never applied automatically.
- **One HTML report**, no server required: tabbed views for the screener,
  the leaderboard, tomorrow's candidates, the adaptive engine's findings,
  and a run log — with all routine status going to that log tab rather than
  the terminal.
- **Resilient to the real failure modes of scraping a live site**: expired
  auth tokens mid-download, a non-trading day that leaves a stale (not
  empty) table on screen instead of clearing it, and files named by the
  date NEPSE itself reports rather than a clock-based guess.

## Stack

Python · Playwright · pandas · plain HTML/CSS (no JS framework, no backend)

## Project structure

```
nepse_screener.py      main pipeline: scraping, indicators, scoring, report
adaptive_engine.py      paper-trading engine (plugs into nepse_screener.py)
demo/                   script that generates the fictional sample dataset
sample_report.html      the report produced from that sample dataset
screenshots/            PNGs of each report tab, for quick viewing
```

## Running it

```bash
pip install playwright pandas requests
playwright install chromium
python nepse_screener.py --adaptive
```

Downloads whatever price/floorsheet history is missing, runs the scoring
and adaptive paper-trading cycle, and opens the finished report in your
browser automatically.

## Demo data

`demo/generate_demo.py` builds a small fictional market (8 invented sectors,
40 invented tickers, 45 days of randomly-walked prices, one day of
synthetic broker-level trades engineered to trip the broker-concentration
and late-session-breakout rules on purpose) and then calls the *real*
`nepse_screener.py` / `adaptive_engine.py` functions against it — nothing
about the pipeline is special-cased for the demo. `sample_report.html` and
`screenshots/` are that run's actual output. A banner at the top of the
report makes clear it's sample data, not real market data.

## Status

Actively maintained personal project. Historical backtesting (replaying the
strategy across existing price history to get instant performance stats,
rather than only forward paper-trading) is planned but not yet built.

## License

MIT — see `LICENSE`.
