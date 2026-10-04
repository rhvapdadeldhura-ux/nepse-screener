"""
NEPSE Adaptive Evaluation Engine
---------------------------------
PAPER TRADING / RESEARCH ONLY. This module never places real orders.

This plugs into nepse_screener.py via `--adaptive`. It runs an immutable
baseline strategy plus two labelled experimental "shadow" strategies in
parallel, paper-tracks every recommendation through entry and exit,
generates daily/weekly learning reports, and only ever *proposes* a
baseline change to a file for a human to review -- it never edits the
active baseline automatically.

=====================================================================
IMPORTANT -- READ THIS BEFORE TRUSTING ANY NUMBER THIS MODULE PRODUCES
=====================================================================
The spec this was built from assumed an existing trade-management engine
(stop-loss, 1.5R partial target, 5-day EMA exit, distribution warning,
10-session holding cap) and an existing fee model. Neither existed
anywhere in nepse_screener.py before this file -- the only prior exit
logic was a flat 5% stop shown in the (separate, cosmetic) risk radar.
Every rule below is therefore THIS MODULE'S OWN FIRST-TIME DEFINITION,
not a pre-existing, battle-tested system. Concretely, these are original
assumptions you should sanity-check against your own trading rules
before trusting the P&L:

  - Entry zone: breakout-day close, to breakout-day close x (1 + 2%).
    Next day's open above that band = "skipped due to gap."
  - Stop-loss: flat 5% below assumed entry (matches the existing radar).
  - 1.5R partial target: at 1.5x initial risk, book 50% of the position
    and move the stop on the rest to breakeven.
  - 5-day EMA exit: once a close falls below the stock's 5-day EMA
    (checked only after a partial has been taken, so a single early
    wiggle doesn't stop the trade out), exit the remainder at next open.
  - Distribution warning: a session with volume >= 1.5x its 20-day
    average, a close in the bottom 30% of that day's range, and a
    negative day. Two of these during the hold triggers a full exit.
  - 10-session holding cap: force-exit at the open of session 11.
  - Fees: a typical/representative NEPSE-style schedule (tiered
    brokerage, SEBON fee, flat DP fee, short-term capital gains tax on
    the sell leg). These are DEFAULTS, not your broker's real published
    rates -- see COST MODEL below and update the constants if yours
    differ.
  - Position sizing: fixed-fractional risk (1% of a configurable paper
    capital base per trade), capped at 20% of capital per position.

None of these numbers came from the ChatGPT prompt (it didn't specify
them) or from anywhere else in this codebase. Treat them as a starting
point to tune, not as "the correct" NEPSE trading rules.
"""

import csv
import datetime as dt
import glob
import json
import os
import time

import pandas as pd

import nepse_screener as ns  # reuse paths, scrapers, and the existing (untouched) scoring engine

# ---------------------------------------------------------------------------
# CONFIG -- all of this is a first-time default; tune freely, but note any
# change to the *baseline* strategy config must create a new version (see
# STRATEGY_VERSIONS below), never edit baseline_v1 in place.
# ---------------------------------------------------------------------------
ADAPTIVE_DIR = os.path.join(ns.BASE_DIR, "adaptive")
INDEX_DIR = os.path.join(ns.BASE_DIR, "index_data")

JOURNAL_PATH = os.path.join(ADAPTIVE_DIR, "paper_trade_journal.csv")
REGISTRY_PATH = os.path.join(ADAPTIVE_DIR, "experiment_registry.csv")
COMPARISON_PATH = os.path.join(ADAPTIVE_DIR, "strategy_comparison.csv")
NO_TRADE_LOG_PATH = os.path.join(ADAPTIVE_DIR, "no_trade_log.csv")
OUTCOMES_PATH = os.path.join(ADAPTIVE_DIR, "recommendation_outcomes.csv")
PROPOSAL_PATH = os.path.join(ADAPTIVE_DIR, "proposed_strategy_change.md")
STATE_PATH = os.path.join(ADAPTIVE_DIR, "engine_state.json")  # small bookkeeping: last processed dates etc.

# ---- Trade-management rules (see module docstring) -------------------------
ENTRY_GAP_BAND = 0.02        # entry zone = breakout close .. breakout close x 1.02
STOP_PCT = 0.05              # flat 5% initial stop, off assumed entry
PARTIAL_R = 1.5              # book 50% at 1.5R
PARTIAL_FRACTION = 0.5
EMA_EXIT_SPAN = 5
DIST_VOL_MULT = 1.5          # volume >= 1.5x 20-day avg
DIST_CLOSE_LOC_MAX = 0.30    # close in bottom 30% of day's range
DIST_WARNINGS_TO_EXIT = 2
MAX_HOLD_SESSIONS = 10
BREAKOUT_LOOKBACK = 10

# ---- Position sizing ---------------------------------------------------
PAPER_CAPITAL = 1_000_000.0  # purely notional -- no real money, no real orders
RISK_PCT_PER_TRADE = 0.01    # 1% of paper capital risked per trade
MAX_POSITION_PCT = 0.20      # never size a single paper position over 20% of capital

# ---- Cost model (see module docstring -- defaults, verify against your own
# broker/SEBON/CDSC published rates) -------------------------------------
SEBON_FEE_PCT = 0.00015          # 0.015% of transaction value, each leg
DP_FEE_FLAT = 25.0               # flat NPR per scrip on the sell leg (CDSC)
CGT_SHORT_TERM_PCT = 0.075       # 7.5% on profit, sell leg, short-term (<365 days) individual


def brokerage_pct(txn_value: float) -> float:
    """Tiered NEPSE-style brokerage schedule. DEFAULT/approximate -- replace
    with your actual broker's published tiers if they differ."""
    if txn_value <= 50_000:
        return 0.0036
    elif txn_value <= 500_000:
        return 0.0033
    elif txn_value <= 2_000_000:
        return 0.0031
    elif txn_value <= 10_000_000:
        return 0.0027
    return 0.0024


def trade_costs(entry_price: float, exit_price: float, shares: float) -> dict:
    """Round-trip cost breakdown for a paper trade. CGT only applies to a
    net profit on the sell leg (losses owe no CGT)."""
    buy_value = entry_price * shares
    sell_value = exit_price * shares
    brokerage = buy_value * brokerage_pct(buy_value) + sell_value * brokerage_pct(sell_value)
    brokerage = max(brokerage, 20.0)  # nominal minimum brokerage, both legs combined
    sebon = (buy_value + sell_value) * SEBON_FEE_PCT
    dp = DP_FEE_FLAT
    gross_pnl = sell_value - buy_value
    cgt = max(0.0, gross_pnl) * CGT_SHORT_TERM_PCT
    total_cost = brokerage + sebon + dp + cgt
    return {"brokerage": brokerage, "sebon": sebon, "dp": dp, "cgt": cgt,
            "gross_pnl": gross_pnl, "net_pnl": gross_pnl - total_cost}


# ---------------------------------------------------------------------------
# STRATEGY VERSIONS -- baseline is immutable. Any parameter change creates a
# NEW version id below (never edit baseline_v1's numbers in place); the
# experiment registry is append-only and will record whichever versions have
# actually been instantiated over time.
# ---------------------------------------------------------------------------
STRATEGY_VERSIONS = {
    "baseline_v1": {
        "label": "NEPSE Breakout With Confirmed Broker Accumulation",
        "kind": "baseline",
        "score_threshold": 70,
        "relvol_min": 1.25, "relvol_max": 2.5,
        "require_breakout": True,
        "close_location_min": None,
        "broker_flow_sessions_required": 1,
        "created": "2026-09-14",
        "reason": "Initial immutable baseline per the adaptive-engine spec.",
    },
    "relaxed_shadow_v1": {
        "label": "Experimental: relaxed selection",
        "kind": "shadow_relaxed",
        "score_threshold": 65,
        "relvol_min": 1.10, "relvol_max": 2.5,
        "require_breakout": True,
        "close_location_min": None,
        "broker_flow_sessions_required": 1,
        "created": "2026-09-14",
        "reason": "Always-on shadow: looser score/relvol floor, same risk rules as baseline.",
    },
    "strict_shadow_v1": {
        "label": "Experimental: strict selection",
        "kind": "shadow_strict",
        "score_threshold": 75,
        "relvol_min": 1.5, "relvol_max": 2.5,
        "require_breakout": True,
        "close_location_min": 0.80,
        "broker_flow_sessions_required": 3,
        "created": "2026-09-14",
        "reason": "Always-on shadow: tighter score/relvol floor, close-location and multi-session "
                  "broker-flow confirmation, for later comparison if baseline underperforms.",
    },
}
BASELINE_ID = "baseline_v1"


def ensure_dirs():
    os.makedirs(ADAPTIVE_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# INDEX DATA -- NEPSE main index + sector sub-indices, for "relative
# performance against the NEPSE index and sector index" in next-day eval.
#
# UNVERIFIED against the live site, same caveat as the rest of this
# codebase's scrapers (see nepse_screener.py's docstring for the floorsheet
# scraper) -- nepalstock.com's index summary widget location/selectors
# aren't confirmed here. If this returns nothing, re-run with --debug --show
# and check index_data/debug_*.png; the fix is almost always a one-line
# selector change in download_index_for_date().
#
# Because that scrape is unverified and NEPSE gives no historical index
# lookup either, this module ALWAYS ALSO builds a synthetic local proxy
# index (turnover-weighted average % change across all downloaded stocks,
# and per-sector versions of the same) from data you already have. The
# synthetic proxy is used automatically whenever the scraped index is
# missing for a given date, so relative-performance numbers keep working
# even if the scraper needs a selector fix.
# ---------------------------------------------------------------------------
INDEX_URL = "https://nepalstock.com/sector-summary"  # confirmed by the user: same date-filter +
                                                       # Filter-button pattern as the floorsheet page


def _find_sector_row_list(obj):
    """Search a parsed JSON structure for the list of dicts holding
    sector-index rows. We don't have a confirmed schema for this endpoint
    yet (unlike the floorsheet one, which was confirmed from a real
    captured sample) -- this heuristic looks for a list whose dict items
    have at least one key mentioning 'sector' or 'index', preferring the
    longest such list. The exact response gets saved to
    debug_sector_response_sample_<date>.json in --debug mode either way,
    so if this heuristic picks the wrong list (or none), that file has
    what's needed to fix it precisely, the same way the floorsheet
    endpoint's real schema was confirmed from a real sample rather than
    guessed."""
    best = [None]

    def looks_like_sector_row(d):
        keys = [str(k).lower() for k in d.keys()]
        return any("sector" in k or "index" in k for k in keys)

    def walk(node):
        if isinstance(node, list):
            if node and all(isinstance(item, dict) for item in node) and any(looks_like_sector_row(item) for item in node):
                if best[0] is None or len(node) > len(best[0]):
                    best[0] = node
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)

    walk(obj)
    return best[0]


def download_index_for_date(page, target_date: str, debug: bool = False) -> bool:
    """Primary path: capture the JSON API response the sector-summary page
    itself fetches (same approach that worked for the floorsheet, after
    the earlier guess-a-table-selector approach proved unreliable on this
    site). Falls back to a raw-table scrape of the rendered page if no
    usable JSON response shows up, since that costs nothing extra to try
    once we're already on the page."""
    out_path = os.path.join(INDEX_DIR, f"index_{target_date}.csv")
    if os.path.exists(out_path):
        return True

    captured = []

    def _on_response(resp):
        try:
            url = resp.url.lower()
            rtype = getattr(resp.request, "resource_type", "")
            if rtype in ("xhr", "fetch") and any(s in url for s in ("sector", "summary", "index")):
                captured.append({"url": resp.url, "body": resp.json()})
        except Exception:
            pass

    page.on("response", _on_response)
    try:
        page.goto(INDEX_URL, wait_until="domcontentloaded", timeout=60000)
        time.sleep(2)
        if debug:
            os.makedirs(INDEX_DIR, exist_ok=True)
            page.screenshot(path=os.path.join(INDEX_DIR, f"debug_{target_date}.png"), full_page=True)

        # Same trigger as the floorsheet page -- confirmed by the user to
        # have the identical date-filter + Filter-button layout.
        filter_btn = page.locator("text=Filter").first
        if filter_btn.count() > 0:
            try:
                filter_btn.click(timeout=10000)
                time.sleep(2)
            except Exception:
                pass

        rows = None
        matched_source = None
        if captured:
            if debug:
                os.makedirs(INDEX_DIR, exist_ok=True)
                sample_path = os.path.join(INDEX_DIR, f"debug_sector_response_sample_{target_date}.json")
                with open(sample_path, "w", encoding="utf-8") as sf:
                    json.dump([c["body"] for c in captured], sf, indent=2)
            for c in captured:
                found = _find_sector_row_list(c["body"])
                if found:
                    rows = found
                    matched_source = c["url"]
                    break

        if rows:
            os.makedirs(INDEX_DIR, exist_ok=True)
            # Schema unconfirmed -- write whatever keys are actually
            # present rather than assuming column names, same lesson as
            # the floorsheet fix (guessed column names silently produced
            # wrong data there; writing the real keys avoids repeating
            # that mistake here before we've seen a real sample).
            fieldnames = []
            seen = set()
            for row in rows:
                for k in row.keys():
                    if k not in seen:
                        seen.add(k)
                        fieldnames.append(k)
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            if debug:
                ns.log(f"    [index-fetch] {len(rows)} row(s) from {matched_source}")
            return True

        # Fallback: the old raw-table scrape, in case this page (unlike
        # the floorsheet one) actually renders server-side HTML rather
        # than fetching JSON. Costs nothing extra since we're on the page
        # already.
        table_rows = []
        tables = page.locator("table")
        for t in range(tables.count()):
            table = tables.nth(t)
            header_cells = table.locator("thead tr th")
            headers = [header_cells.nth(i).inner_text().strip().lower() for i in range(header_cells.count())]
            if not headers or not any("sector" in h or "index" in h for h in headers):
                continue
            body_rows = table.locator("tbody tr")
            for i in range(body_rows.count()):
                cells = body_rows.nth(i).locator("td")
                row = [cells.nth(j).inner_text().strip() for j in range(cells.count())]
                if row:
                    table_rows.append(row)
            if table_rows:
                break

        if not table_rows:
            return False

        os.makedirs(INDEX_DIR, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Index Name", "Close", "Change", "PctChange"])
            writer.writerows(table_rows)
        return True
    except Exception as e:
        if debug:
            print(f"    index scrape error: {e}")
        return False
    finally:
        try:
            page.remove_listener("response", _on_response)
        except Exception:
            pass


def download_index_missing(headless: bool = True, debug: bool = False):
    """Runs once per adaptive cycle, same pattern as the floorsheet
    downloader -- best-effort scrape, never blocks the rest of the engine
    if it fails (the synthetic proxy covers for it)."""
    ensure_dirs()
    target = dt.date.today().isoformat()
    out_path = os.path.join(INDEX_DIR, f"index_{target}.csv")
    if os.path.exists(out_path):
        return
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        ns.log_error("Playwright not installed -- skipping index scrape (synthetic proxy index will be used instead).")
        return
    ns.log(f"Attempting NEPSE index scrape for {target} (unverified selectors -- synthetic proxy is the fallback)...")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless, args=["--disable-http2"])
            page = browser.new_context().new_page()
            ok = download_index_for_date(page, target, debug=debug)
            browser.close()
        if ok:
            ns.log(f"  Index data saved for {target}.")
        else:
            ns.log_error("  Index scrape found no matching table -- using the synthetic proxy index instead "
                  "(re-run with --debug --show and check index_data/debug_*.png to fix the selector).")
    except Exception as e:
        ns.log_error(f"  Index scrape failed ({e}) -- using the synthetic proxy index instead.")


def build_synthetic_index(all_df: pd.DataFrame, sector_map: dict) -> pd.DataFrame:
    """Turnover-weighted daily % change across all downloaded stocks (market
    proxy) and per-sector versions of the same, built entirely from data
    already on disk. Always available, unlike the live scrape."""
    df = all_df.copy()
    df["Sector"] = df["Symbol"].map(sector_map).fillna("Uncategorized")
    df = df.sort_values(["Symbol", "Business Date"])
    df["PctChange"] = df.groupby("Symbol")["Close Price"].pct_change() * 100
    turnover_col = ns._find_col(df, ["turnover", "traded value"])
    df["Turnover"] = (pd.to_numeric(df[turnover_col].astype(str).str.replace(",", ""), errors="coerce")
                       if turnover_col else df["Close Price"] * df["Total Traded Quantity"])

    def wavg(g):
        w = g["Turnover"].clip(lower=0)
        if w.sum() <= 0 or g["PctChange"].isna().all():
            return pd.NA
        return (g["PctChange"].fillna(0) * w).sum() / w.sum()

    market = df.groupby("Business Date").apply(wavg).rename("MarketPctChange").reset_index()
    sector = df.groupby(["Business Date", "Sector"]).apply(wavg).rename("SectorPctChange").reset_index()
    return market, sector


def _normalize_sector_name(s: str) -> str:
    """Loose match key for comparing NEPSE's own sector-summary naming
    (e.g. "Hydro Power Index", "Microfinance Sub-Index") against our
    sector_map's naming (e.g. "Hydropower", "Microfinance") -- strips the
    generic index/sector words and all non-alphanumerics so the two
    vocabularies line up even when the exact wording doesn't match."""
    import re
    s = str(s).lower()
    s = re.sub(r"\b(sector|index|sub[- ]?index)\b", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def load_index_history(all_df: pd.DataFrame, sector_map: dict):
    """Returns (market_df, sector_df). Prefers scraped index_data/*.csv
    where present; falls back to the synthetic proxy for any date (or
    sector) it's missing.

    index_*.csv is written by nepse_screener.download_index_for_date from
    whatever real column names the sector-summary API happens to return --
    there's no fixed "Index Name"/"PctChange" header to rely on (an earlier
    version assumed there was, and crashed with KeyError: 'Index Name' the
    moment the real scraper started writing files with different column
    names). So this uses ns._find_col's fuzzy, case-insensitive matching to
    locate the name/pct-change columns under whatever they're actually
    called, and matches each row to either the overall market index or one
    of our own sectors via _normalize_sector_name."""
    market_synth, sector_synth = build_synthetic_index(all_df, sector_map)

    files = sorted(glob.glob(os.path.join(INDEX_DIR, "index_*.csv")))
    if not files:
        return market_synth, sector_synth

    market_rows = []
    sector_rows = []
    our_sector_names = {_normalize_sector_name(sec): sec for sec in set(sector_map.values())}

    for f in files:
        date_str = os.path.basename(f).replace("index_", "").replace(".csv", "")
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if d.empty:
            continue

        name_col = ns._find_col(d, ["sectorname", "sector name", "indexname", "index name", "name"])
        pct_col = ns._find_col(d, ["percentchange", "pctchange", "percent change", "pct change",
                                    "changepercent", "perchange"])
        if name_col is None or pct_col is None:
            continue

        for _, row in d.iterrows():
            raw_name = str(row[name_col])
            pct = pd.to_numeric(str(row[pct_col]).replace("%", "").replace(",", ""), errors="coerce")
            if pd.isna(pct):
                continue
            if "nepse" in raw_name.lower():
                market_rows.append({"Business Date": pd.Timestamp(date_str), "MarketPctChange": pct})
                continue
            key = _normalize_sector_name(raw_name)
            matched_sector = our_sector_names.get(key)
            if matched_sector:
                sector_rows.append({"Business Date": pd.Timestamp(date_str),
                                     "Sector": matched_sector, "SectorPctChange": pct})

    # Both sides of the merge need the same dtype for "Business Date" --
    # market_synth/sector_synth can come through as object/date/datetime64
    # depending on how all_df was built, and pandas refuses to merge on
    # mismatched key dtypes.
    market_synth = market_synth.copy()
    market_synth["Business Date"] = pd.to_datetime(market_synth["Business Date"])
    sector_synth = sector_synth.copy()
    sector_synth["Business Date"] = pd.to_datetime(sector_synth["Business Date"])

    market = market_synth
    if market_rows:
        scraped_market = pd.DataFrame(market_rows)
        scraped_market["Business Date"] = pd.to_datetime(scraped_market["Business Date"])
        market = market_synth.merge(scraped_market, on="Business Date", how="left", suffixes=("_synth", ""))
        market["MarketPctChange"] = market["MarketPctChange"].fillna(market["MarketPctChange_synth"])
        market = market.drop(columns=["MarketPctChange_synth"])

    sector = sector_synth
    if sector_rows:
        scraped_sector = pd.DataFrame(sector_rows)
        scraped_sector["Business Date"] = pd.to_datetime(scraped_sector["Business Date"])
        sector = sector_synth.merge(scraped_sector, on=["Business Date", "Sector"], how="left", suffixes=("_synth", ""))
        sector["SectorPctChange"] = sector["SectorPctChange"].fillna(sector["SectorPctChange_synth"])
        sector = sector.drop(columns=["SectorPctChange_synth"])

    return market, sector


# ---------------------------------------------------------------------------
# FULL INDICATOR HISTORY -- nepse_screener.compute_indicators() keeps only
# the latest row per symbol. The adaptive engine needs the full per-day
# series (to gate signals on a real prior-10-day breakout, and to walk open
# trades forward day by day), so this rebuilds a comparable set of columns
# but keeps every row.
# ---------------------------------------------------------------------------
def build_full_indicator_history(all_df: pd.DataFrame) -> pd.DataFrame:
    open_col = ns._find_col(all_df, ["open price"])
    high_col = ns._find_col(all_df, ["high price"])
    low_col = ns._find_col(all_df, ["low price"])

    df = all_df.copy()
    df["Open"] = pd.to_numeric(df[open_col], errors="coerce") if open_col else df["Close Price"]
    df["High"] = pd.to_numeric(df[high_col], errors="coerce") if high_col else df["Close Price"]
    df["Low"] = pd.to_numeric(df[low_col], errors="coerce") if low_col else df["Close Price"]

    results = []
    for symbol, g in df.groupby("Symbol"):
        g = g.sort_values("Business Date").reset_index(drop=True)
        close = g["Close Price"]
        g["PctChange"] = close.pct_change() * 100
        g["EMA5"] = close.ewm(span=5, adjust=False).mean()
        g["VolAvg20"] = g["Total Traded Quantity"].rolling(20, min_periods=5).mean()
        g["VolRatio"] = g["Total Traded Quantity"] / g["VolAvg20"]
        # Prior N-day high EXCLUDES today (shift(1)) -- today's close has to
        # clear the high set before it to count as a breakout, not include
        # itself in the bar it's being measured against.
        g["PriorHigh10"] = close.shift(1).rolling(BREAKOUT_LOOKBACK, min_periods=5).max()
        day_range = (g["High"] - g["Low"]).replace(0, pd.NA)
        g["CloseLocation"] = (g["Close Price"] - g["Low"]) / day_range
        g["IsDistributionDay"] = (
            (g["VolRatio"] >= DIST_VOL_MULT) & (g["CloseLocation"] <= DIST_CLOSE_LOC_MAX) & (g["PctChange"] < 0)
        )
        results.append(g)
    return pd.concat(results, ignore_index=True).sort_values(["Symbol", "Business Date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# MULTI-SESSION FLOORSHEET HISTORY -- the strict shadow strategy needs
# "positive top-three broker flow on all three recent floorsheet sessions",
# which nepse_screener's rule1_broker_concentration doesn't provide (it only
# looks at the single most-recently-downloaded floorsheet file).
# ---------------------------------------------------------------------------
def load_floorsheet_history() -> dict:
    """Returns {date_str: floorsheet_df} for every floorsheet_data/floorsheet_*.csv
    on disk, parsed with the same column-detection logic as
    nepse_screener.load_latest_floorsheet(), so any local floorsheet
    history you've accumulated is usable here, not just today's file."""
    out = {}
    files = sorted(glob.glob(os.path.join(ns.FLOORSHEET_DIR, "floorsheet_*.csv")))
    for f in files:
        date_str = os.path.basename(f).replace("floorsheet_", "").replace(".csv", "")
        try:
            df = pd.read_csv(f, engine="python", on_bad_lines="skip")
        except Exception:
            continue
        sym_col = ns._find_col(df, ["symbol"])
        # Same fix as nepse_screener.load_latest_floorsheet(): prefer the
        # actual broker-name column over a numeric member-id column when
        # both exist, so multi-session broker-flow tracking groups by a
        # readable name instead of an ID (confirmed real format has both
        # buyerMemberId/sellerMemberId and buyerBrokerName/sellerBrokerName,
        # with the ID column ordered first).
        buyer_col = ns._find_col(df, ["buyerbrokername", "buyer broker", "buyer"])
        seller_col = ns._find_col(df, ["sellerbrokername", "seller broker", "seller"])
        qty_col = ns._find_col(df, ["quantity", "qty"])
        if not all([sym_col, buyer_col, seller_col, qty_col]):
            continue
        parsed = pd.DataFrame({
            "Symbol": df[sym_col].astype(str).str.strip().str.upper(),
            "Buyer": df[buyer_col].astype(str).str.strip(),
            "Seller": df[seller_col].astype(str).str.strip(),
            "Quantity": pd.to_numeric(df[qty_col].astype(str).str.replace(",", ""), errors="coerce"),
        }).dropna(subset=["Symbol", "Quantity"])
        if not parsed.empty:
            out[date_str] = parsed
    return out


def broker_flow_persistent(symbol: str, floorsheet_history: dict, sessions_required: int):
    """Does this symbol's top-3-by-volume brokers show net positive (buy >
    sell) flow on each of the most recent `sessions_required` floorsheet
    sessions available? Returns (bool, detail_str). If fewer sessions of
    history exist than required, this fails closed (returns False) rather
    than passing on partial evidence -- the strict strategy is supposed to
    be hard to qualify for."""
    dates = sorted(floorsheet_history.keys())[-sessions_required:]
    if len(dates) < sessions_required:
        return False, f"Only {len(dates)} floorsheet session(s) on disk, need {sessions_required}."

    session_results = []
    for d in dates:
        fdf = floorsheet_history[d]
        rows = fdf[fdf["Symbol"] == symbol.upper()]
        if rows.empty:
            session_results.append(f"{d}: no rows")
            continue
        buy_by_broker = rows.groupby("Buyer")["Quantity"].sum().sort_values(ascending=False)
        top3 = set(buy_by_broker.head(3).index)
        sell_by_broker = rows.groupby("Seller")["Quantity"].sum()
        top3_buy = buy_by_broker.reindex(top3, fill_value=0).sum()
        top3_sell = sell_by_broker.reindex(top3, fill_value=0).sum()
        positive = top3_buy > top3_sell
        session_results.append(f"{d}: top-3 buy {top3_buy:,.0f} vs sell {top3_sell:,.0f} ({'+' if positive else '-'})")
        if not positive:
            return False, "; ".join(session_results)

    return True, "; ".join(session_results)


# ---------------------------------------------------------------------------
# JOURNAL I/O -- the append-only-ish trade ledger. Existing rows are only
# ever updated in place to advance their own lifecycle (pending -> open ->
# closed); nothing is ever deleted, and closed trades are never reopened.
# ---------------------------------------------------------------------------
JOURNAL_COLUMNS = [
    "trade_id", "strategy_version", "symbol", "sector", "signal_date",
    "qualifying_reason", "adaptive_score", "main_risk", "evidence",
    "entry_zone_low", "entry_zone_high", "entry_date", "status",
    "entry_price", "stop_price", "initial_stop_price", "target_price_1_5r", "shares", "position_value",
    "next_day_evaluated", "next_day_return_pct", "next_day_held_breakout",
    "next_day_mfe_pct", "next_day_mae_pct", "next_day_vs_index_pct", "next_day_vs_sector_pct",
    "partial_taken", "partial_date", "partial_price", "dist_warning_count",
    "last_checked_date", "holding_sessions",
    "exit_date", "exit_price", "exit_reason", "gross_pnl", "brokerage_cost",
    "sebon_fee", "dp_fee", "cgt", "net_pnl", "r_multiple", "mfe_pct", "mae_pct",
]


def load_journal() -> pd.DataFrame:
    ensure_dirs()
    if not os.path.exists(JOURNAL_PATH):
        return pd.DataFrame(columns=JOURNAL_COLUMNS)
    df = pd.read_csv(JOURNAL_PATH)
    for c in JOURNAL_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[JOURNAL_COLUMNS]


def save_journal(df: pd.DataFrame):
    ensure_dirs()
    df.to_csv(JOURNAL_PATH, index=False)


def next_trade_id(journal_df: pd.DataFrame) -> str:
    n = len(journal_df) + 1
    return f"T{n:05d}"


def log_no_trade(date_str: str, version: str, category: str, detail: str):
    """category is one of the five plain-language buckets the spec asks
    for: poor_conditions, rules_may_be_restrictive, insufficient_data,
    gap_skipped, other."""
    ensure_dirs()
    row = {"date": date_str, "strategy_version": version, "reason_category": category, "reason_detail": detail}
    file_exists = os.path.exists(NO_TRADE_LOG_PATH)
    with open(NO_TRADE_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def log_outcome(row: dict):
    ensure_dirs()
    file_exists = os.path.exists(OUTCOMES_PATH)
    with open(OUTCOMES_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def ensure_registry():
    """Append-only: writes a row for any strategy version that isn't
    already recorded. Never rewrites or deletes an existing row."""
    ensure_dirs()
    existing_ids = set()
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_ids.add(row["version_id"])

    file_exists = os.path.exists(REGISTRY_PATH)
    fieldnames = ["version_id", "label", "kind", "created", "reason", "config_json", "status"]
    new_rows = []
    for vid, cfg in STRATEGY_VERSIONS.items():
        if vid in existing_ids:
            continue
        cfg_copy = {k: v for k, v in cfg.items() if k not in ("label", "kind", "created", "reason")}
        new_rows.append({
            "version_id": vid, "label": cfg["label"], "kind": cfg["kind"],
            "created": cfg["created"], "reason": cfg["reason"],
            "config_json": json.dumps(cfg_copy), "status": "baseline" if vid == BASELINE_ID else "shadow",
        })
    if new_rows:
        with open(REGISTRY_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(new_rows)


# ---------------------------------------------------------------------------
# SIGNAL GENERATION -- reuses nepse_screener's existing, untouched scoring
# functions (sector_rotation_marks / rule1_broker_concentration /
# rule2_late_session_breakout -> the same 0-100 "AdaptiveScore" as the
# --leaderboard Total Marks) and layers each strategy version's extra gates
# (relative volume band, a literal prior-10-day breakout, close location,
# multi-session broker-flow persistence) on top, WITHOUT modifying those
# functions -- this keeps the baseline's core scoring identical to the
# existing leaderboard rather than inventing a second parallel scoring
# system.
# ---------------------------------------------------------------------------
def build_signal_frame(latest: pd.DataFrame, extra_hist: pd.DataFrame) -> pd.DataFrame:
    last_extra = extra_hist.sort_values("Business Date").groupby("Symbol").tail(1)
    keep = last_extra[["Symbol", "PriorHigh10", "CloseLocation"]]
    return latest.merge(keep, on="Symbol", how="left")


def generate_candidates(strategy_id: str, signal_frame: pd.DataFrame, all_df: pd.DataFrame,
                         sector_map: dict, floorsheet_df, floorsheet_history: dict):
    """Returns (candidates: list[dict], rejects: list[dict]) for one strategy
    version, evaluated against today's (most recent trading day's) data.
    Each dict carries everything the journal / reports need, including
    `reasons_failed` (empty for candidates) so rule 4's "exactly which
    baseline condition prevented each shadow candidate from qualifying"
    can be answered directly from the baseline's own rejects list."""
    cfg = STRATEGY_VERSIONS[strategy_id]
    sector_marks, _top_sectors = ns.sector_rotation_marks(all_df, sector_map)

    candidates, rejects = [], []
    for _, row in signal_frame.iterrows():
        symbol = row["Symbol"]
        sec_marks = sector_marks.get(row.get("Sector"), 0)
        r1_marks, r1_detail = ns.rule1_broker_concentration(symbol, floorsheet_df)
        r2_marks, r2_detail = ns.rule2_late_session_breakout(symbol, row, floorsheet_df)
        total = sec_marks + r1_marks + r2_marks

        reasons_failed = []
        if total < cfg["score_threshold"]:
            reasons_failed.append(f"AdaptiveScore {total} below threshold {cfg['score_threshold']}")

        volr = row.get("VolRatio")
        if pd.isna(volr) or not (cfg["relvol_min"] <= volr <= cfg["relvol_max"]):
            shown = "n/a" if pd.isna(volr) else f"{volr:.2f}x"
            reasons_failed.append(f"Relative volume {shown} outside [{cfg['relvol_min']}, {cfg['relvol_max']}]")

        if cfg["require_breakout"]:
            prior_high, close = row.get("PriorHigh10"), row.get("Close Price")
            if pd.isna(prior_high) or pd.isna(close) or not (close > prior_high):
                reasons_failed.append("No confirmed close above the prior 10-day high")

        if r1_marks <= 0:
            reasons_failed.append("No confirmed single-broker accumulation in today's floorsheet")

        if cfg["close_location_min"] is not None:
            cl = row.get("CloseLocation")
            if pd.isna(cl) or cl < cfg["close_location_min"]:
                shown = "n/a" if pd.isna(cl) else f"{cl*100:.0f}%"
                reasons_failed.append(f"Close location {shown} below required top "
                                       f"{(1-cfg['close_location_min'])*100:.0f}% of day's range")

        if cfg["broker_flow_sessions_required"] > 1:
            ok, detail = broker_flow_persistent(symbol, floorsheet_history, cfg["broker_flow_sessions_required"])
            if not ok:
                reasons_failed.append(f"Broker-flow persistence failed ({detail})")

        record = {
            "symbol": symbol, "sector": row.get("Sector"), "adaptive_score": total,
            "sector_marks": sec_marks, "r1_marks": r1_marks, "r2_marks": r2_marks,
            "r1_detail": r1_detail, "r2_detail": r2_detail, "row": row,
            "reasons_failed": reasons_failed,
        }
        (candidates if not reasons_failed else rejects).append(record)

    return candidates, rejects


# ---------------------------------------------------------------------------
# TRADE LIFECYCLE: open -> next-day eval -> day-by-day exit walk
#
# SIMPLIFICATION NOTE: to avoid needing to persist "pending exit, act on
# next day's open" flags across separate script runs, the EMA exit and
# distribution-warning exit below trigger at that SAME session's close
# rather than the following session's open. The stop-loss (intrabar Low)
# and the 1.5R partial (intrabar High) are checked properly intrabar. This
# is a deliberate simplification, not an oversight -- flag it if you want
# true next-open exit timing, it's a moderate follow-up change.
# ---------------------------------------------------------------------------
def open_new_trades(candidates: list, strategy_id: str, signal_date, journal_df: pd.DataFrame) -> pd.DataFrame:
    open_symbols = set(journal_df.loc[
        (journal_df["strategy_version"] == strategy_id) &
        (journal_df["status"].isin(["pending_entry", "open"])), "symbol"
    ])
    new_rows = []
    for c in candidates:
        if c["symbol"] in open_symbols:
            continue  # don't double up on a symbol already live for this strategy version
        row = c["row"]
        breakout_level = float(row["Close Price"])
        evidence = f"{c['r1_detail']} {c['r2_detail']} AdaptiveScore {c['adaptive_score']}/100.".strip()
        new_rows.append({
            "trade_id": next_trade_id(journal_df) if not new_rows else f"T{len(journal_df)+len(new_rows)+1:05d}",
            "strategy_version": strategy_id, "symbol": c["symbol"], "sector": c["sector"],
            "signal_date": signal_date, "qualifying_reason": "Passed all strategy gates -- see evidence.",
            "adaptive_score": c["adaptive_score"], "main_risk": "Breakout failure / broker flow reversal.",
            "evidence": evidence,
            "entry_zone_low": breakout_level, "entry_zone_high": breakout_level * (1 + ENTRY_GAP_BAND),
            "entry_date": pd.NA, "status": "pending_entry",
            "entry_price": pd.NA, "stop_price": pd.NA, "initial_stop_price": pd.NA,
            "target_price_1_5r": pd.NA, "shares": pd.NA, "position_value": pd.NA,
            "next_day_evaluated": False, "next_day_return_pct": pd.NA, "next_day_held_breakout": pd.NA,
            "next_day_mfe_pct": pd.NA, "next_day_mae_pct": pd.NA,
            "next_day_vs_index_pct": pd.NA, "next_day_vs_sector_pct": pd.NA,
            "partial_taken": False, "partial_date": pd.NA, "partial_price": pd.NA, "dist_warning_count": 0,
            "last_checked_date": pd.NA, "holding_sessions": 0,
            "exit_date": pd.NA, "exit_price": pd.NA, "exit_reason": pd.NA,
            "gross_pnl": pd.NA, "brokerage_cost": pd.NA, "sebon_fee": pd.NA, "dp_fee": pd.NA, "cgt": pd.NA,
            "net_pnl": pd.NA, "r_multiple": pd.NA, "mfe_pct": pd.NA, "mae_pct": pd.NA,
        })
        log_outcome({"signal_date": signal_date, "symbol": c["symbol"], "strategy_version": strategy_id,
                     "qualifying_reason": evidence, "next_day_result": "pending", "final_trade_result": "pending"})
    if new_rows:
        journal_df = pd.concat([journal_df, pd.DataFrame(new_rows)], ignore_index=True)
    return journal_df


def evaluate_pending_entries(journal_df: pd.DataFrame, extra_hist: pd.DataFrame,
                              index_market: pd.DataFrame, index_sector: pd.DataFrame) -> pd.DataFrame:
    """Processes every pending_entry row whose signal day now has a
    following trading day of data available. Fills or marks "skipped due to
    gap", and records the full next-day evaluation either way."""
    for i, tr in journal_df[journal_df["status"] == "pending_entry"].iterrows():
        symbol = tr["symbol"]
        signal_date = pd.Timestamp(tr["signal_date"])
        sym_hist = extra_hist[extra_hist["Symbol"] == symbol].sort_values("Business Date")
        after = sym_hist[sym_hist["Business Date"] > signal_date]
        if after.empty:
            continue  # still waiting on tomorrow's data
        nd = after.iloc[0]  # next trading day
        entry_zone_low, entry_zone_high = tr["entry_zone_low"], tr["entry_zone_high"]

        next_open, next_close = nd["Open"], nd["Close Price"]
        next_high, next_low = nd["High"], nd["Low"]

        skipped = pd.notna(next_open) and next_open > entry_zone_high
        journal_df.at[i, "next_day_evaluated"] = True
        journal_df.at[i, "next_day_held_breakout"] = bool(next_close >= entry_zone_low) if pd.notna(next_close) else pd.NA

        idx_row = index_market[index_market["Business Date"] == nd["Business Date"]]
        idx_pct = idx_row["MarketPctChange"].iloc[0] if not idx_row.empty else pd.NA
        sec_row = index_sector[(index_sector["Business Date"] == nd["Business Date"]) &
                                (index_sector["Sector"] == tr["sector"])]
        sec_pct = sec_row["SectorPctChange"].iloc[0] if not sec_row.empty else pd.NA
        journal_df.at[i, "next_day_vs_index_pct"] = idx_pct
        journal_df.at[i, "next_day_vs_sector_pct"] = sec_pct

        if skipped:
            journal_df.at[i, "status"] = "skipped_gap"
            journal_df.at[i, "entry_date"] = nd["Business Date"]
            assumed_entry = entry_zone_high
            if pd.notna(next_close):
                journal_df.at[i, "next_day_return_pct"] = (next_close - assumed_entry) / assumed_entry * 100
            log_no_trade(str(nd["Business Date"].date()), tr["strategy_version"], "gap_skipped",
                         f"{symbol}: next open {next_open:.2f} above entry zone ceiling {entry_zone_high:.2f} "
                         f"-- not counted as a win or loss.")
            continue

        entry_price = max(next_open, entry_zone_low) if pd.notna(next_open) else entry_zone_low
        stop_price = entry_price * (1 - STOP_PCT)
        r_value = entry_price - stop_price
        target_price = entry_price + PARTIAL_R * r_value

        risk_budget = PAPER_CAPITAL * RISK_PCT_PER_TRADE
        shares_by_risk = risk_budget / r_value if r_value > 0 else 0
        shares_by_cap = (PAPER_CAPITAL * MAX_POSITION_PCT) / entry_price
        shares = max(0, int(min(shares_by_risk, shares_by_cap)))

        journal_df.at[i, "status"] = "open"
        journal_df.at[i, "entry_date"] = nd["Business Date"]
        journal_df.at[i, "entry_price"] = entry_price
        journal_df.at[i, "stop_price"] = stop_price
        journal_df.at[i, "initial_stop_price"] = stop_price
        journal_df.at[i, "target_price_1_5r"] = target_price
        journal_df.at[i, "shares"] = shares
        journal_df.at[i, "position_value"] = shares * entry_price
        journal_df.at[i, "last_checked_date"] = signal_date  # walk_open_trades starts from AFTER this
        journal_df.at[i, "holding_sessions"] = 0

        if pd.notna(next_close):
            journal_df.at[i, "next_day_return_pct"] = (next_close - entry_price) / entry_price * 100
        if pd.notna(next_high):
            journal_df.at[i, "next_day_mfe_pct"] = (next_high - entry_price) / entry_price * 100
        if pd.notna(next_low):
            journal_df.at[i, "next_day_mae_pct"] = (next_low - entry_price) / entry_price * 100

    return journal_df


def walk_open_trades(journal_df: pd.DataFrame, extra_hist: pd.DataFrame) -> pd.DataFrame:
    """Advances every status=='open' trade forward through any new trading
    days that have appeared since it was last checked, applying stop /
    partial / EMA-exit / distribution-warning / max-holding rules in that
    priority order each session (see module docstring for exact
    definitions and the same-session-close simplification)."""
    for i, tr in journal_df[journal_df["status"] == "open"].iterrows():
        symbol = tr["symbol"]
        sym_hist = extra_hist[extra_hist["Symbol"] == symbol].sort_values("Business Date")
        since = pd.Timestamp(tr["last_checked_date"])
        days = sym_hist[sym_hist["Business Date"] > since]
        if days.empty:
            continue

        stop_price = float(tr["stop_price"])
        initial_stop = float(tr["initial_stop_price"])
        entry_price = float(tr["entry_price"])
        target_price = float(tr["target_price_1_5r"])
        shares = float(tr["shares"])
        partial_taken = bool(tr["partial_taken"])
        partial_date, partial_price = tr["partial_date"], tr["partial_price"]
        dist_count = int(tr["dist_warning_count"]) if pd.notna(tr["dist_warning_count"]) else 0
        holding = int(tr["holding_sessions"]) if pd.notna(tr["holding_sessions"]) else 0
        mfe = float(tr["mfe_pct"]) if pd.notna(tr["mfe_pct"]) else -1e9
        mae = float(tr["mae_pct"]) if pd.notna(tr["mae_pct"]) else 1e9

        closed = False
        for _, d in days.iterrows():
            holding += 1
            if pd.notna(d["High"]):
                mfe = max(mfe, (d["High"] - entry_price) / entry_price * 100)
            if pd.notna(d["Low"]):
                mae = min(mae, (d["Low"] - entry_price) / entry_price * 100)

            if not partial_taken and pd.notna(d["High"]) and d["High"] >= target_price:
                partial_taken, partial_date, partial_price = True, d["Business Date"], target_price
                stop_price = entry_price  # move stop to breakeven on the remainder

            if pd.notna(d["Low"]) and d["Low"] <= stop_price:
                exit_price = stop_price
                reason = "breakeven_stop" if partial_taken and stop_price == entry_price else "stop_loss"
                closed = True
            elif holding >= MAX_HOLD_SESSIONS:
                exit_price, reason, closed = d["Close Price"], "max_holding", True
            elif d.get("IsDistributionDay", False):
                dist_count += 1
                if dist_count >= DIST_WARNINGS_TO_EXIT:
                    exit_price, reason, closed = d["Close Price"], "distribution_warning", True
            if not closed and partial_taken and pd.notna(d["Close Price"]) and pd.notna(d.get("EMA5")) \
                    and d["Close Price"] < d["EMA5"]:
                exit_price, reason, closed = d["Close Price"], "ema_exit", True

            last_day = d
            if closed:
                break

        journal_df.at[i, "partial_taken"] = partial_taken
        journal_df.at[i, "partial_date"] = partial_date
        journal_df.at[i, "partial_price"] = partial_price
        journal_df.at[i, "dist_warning_count"] = dist_count
        journal_df.at[i, "holding_sessions"] = holding
        journal_df.at[i, "stop_price"] = stop_price
        journal_df.at[i, "mfe_pct"] = mfe
        journal_df.at[i, "mae_pct"] = mae
        journal_df.at[i, "last_checked_date"] = last_day["Business Date"]

        if closed:
            if partial_taken:
                blended_exit = PARTIAL_FRACTION * partial_price + (1 - PARTIAL_FRACTION) * exit_price
            else:
                blended_exit = exit_price
            costs = trade_costs(entry_price, blended_exit, shares)
            r_value = entry_price - initial_stop
            journal_df.at[i, "status"] = "closed"
            journal_df.at[i, "exit_date"] = last_day["Business Date"]
            journal_df.at[i, "exit_price"] = blended_exit
            journal_df.at[i, "exit_reason"] = reason
            journal_df.at[i, "gross_pnl"] = costs["gross_pnl"]
            journal_df.at[i, "brokerage_cost"] = costs["brokerage"]
            journal_df.at[i, "sebon_fee"] = costs["sebon"]
            journal_df.at[i, "dp_fee"] = costs["dp"]
            journal_df.at[i, "cgt"] = costs["cgt"]
            journal_df.at[i, "net_pnl"] = costs["net_pnl"]
            journal_df.at[i, "r_multiple"] = costs["net_pnl"] / (r_value * shares) if r_value > 0 and shares > 0 else pd.NA

    return journal_df


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------
def compute_strategy_metrics(journal_df: pd.DataFrame, version: str) -> dict:
    closed = journal_df[(journal_df["strategy_version"] == version) & (journal_df["status"] == "closed")].copy()
    n = len(closed)
    if n == 0:
        return {"trades_completed": 0}

    closed = closed.sort_values("exit_date")
    wins = closed[closed["net_pnl"] > 0]
    losses = closed[closed["net_pnl"] <= 0]
    win_rate = len(wins) / n
    avg_win = wins["net_pnl"].mean() if len(wins) else 0.0
    avg_loss = losses["net_pnl"].mean() if len(losses) else 0.0
    gross_profit = wins["net_pnl"].sum()
    gross_loss = abs(losses["net_pnl"].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    expectancy_r = closed["r_multiple"].mean()

    equity = closed["net_pnl"].cumsum()
    running_max = equity.cummax()
    drawdown = (equity - running_max)
    max_drawdown = drawdown.min() if len(drawdown) else 0.0

    return {
        "trades_completed": n, "win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss,
        "expectancy_r": expectancy_r, "profit_factor": profit_factor, "max_drawdown": max_drawdown,
        "avg_holding_period": closed["holding_sessions"].mean(),
        "net_pnl_total": closed["net_pnl"].sum(),
    }


# ---------------------------------------------------------------------------
# ENGINE STATE -- small bookkeeping used only for rule 4 (consecutive
# no-candidate streak) and rule 5 (persistence across weekly reports). Never
# stores strategy results themselves -- those live in the CSVs, which are
# never overwritten in place.
# ---------------------------------------------------------------------------
def load_state() -> dict:
    ensure_dirs()
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"baseline_no_candidate_streak": 0, "last_weekly_iso_week": None,
            "consecutive_weekly_underperformance": 0}


def save_state(state: dict):
    ensure_dirs()
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# DAILY REPORT
# ---------------------------------------------------------------------------
def write_daily_review(signal_date, results_by_version: dict, journal_df: pd.DataFrame, market_up_ratio: float):
    ensure_dirs()
    date_str = str(pd.Timestamp(signal_date).date())
    lines = [f"# Daily Review -- {date_str}\n", "_Paper-trading research only. No real orders are placed._\n"]

    for vid, res in results_by_version.items():
        cfg = STRATEGY_VERSIONS[vid]
        lines.append(f"## {cfg['label']} (`{vid}`)\n")
        cands, rejects = res["candidates"], res["rejects"]
        if cands:
            lines.append(f"**{len(cands)} recommendation(s) today:**\n")
            for c in cands:
                lines.append(f"- **{c['symbol']}** ({c['sector']}) -- AdaptiveScore {c['adaptive_score']}/100. {c['r1_detail']} {c['r2_detail']}")
        else:
            lines.append("**No qualified trade today.** This is a valid outcome, not a failure.\n")
        lines.append("")

        # Entry-fill / skip / pending status for anything opened on a PRIOR
        # signal day that resolved today.
        resolved_today = journal_df[
            (journal_df["strategy_version"] == vid) &
            (pd.to_datetime(journal_df["entry_date"], errors="coerce") == pd.Timestamp(signal_date))
        ]
        if not resolved_today.empty:
            lines.append("**Entries resolved today:**\n")
            for _, tr in resolved_today.iterrows():
                if tr["status"] == "skipped_gap":
                    lines.append(f"- {tr['symbol']}: SKIPPED (gapped above entry zone {tr['entry_zone_high']:.2f}). "
                                 f"Not counted as a win or loss.")
                else:
                    held = "held" if tr["next_day_held_breakout"] else "failed to hold"
                    lines.append(f"- {tr['symbol']}: filled at {tr['entry_price']:.2f}, next-day return "
                                 f"{tr['next_day_return_pct']:+.2f}%, breakout {held}.")
            lines.append("")

        closed_today = journal_df[
            (journal_df["strategy_version"] == vid) &
            (pd.to_datetime(journal_df["exit_date"], errors="coerce") == pd.Timestamp(signal_date))
        ]
        if not closed_today.empty:
            lines.append("**Trades closed today:**\n")
            for _, tr in closed_today.iterrows():
                lines.append(f"- {tr['symbol']}: exit `{tr['exit_reason']}` at {tr['exit_price']:.2f}, "
                             f"net P&L {tr['net_pnl']:+.0f}, R {tr['r_multiple']:+.2f}.")
            lines.append("")

    lines.append("## Market Regime\n")
    mood = "positive" if market_up_ratio > 0.55 else ("negative" if market_up_ratio < 0.45 else "mixed")
    lines.append(f"{market_up_ratio*100:.0f}% of stocks advancing today -- regime classified as **{mood}**.\n")

    lines.append("## What worked / what didn't (plain language)\n")
    base_cands = results_by_version.get(BASELINE_ID, {}).get("candidates", [])
    if base_cands:
        lines.append("Baseline found qualifying setups today -- see the recommendations above for the specific "
                     "evidence (broker concentration, sector rotation, breakout confirmation) behind each one.\n")
    else:
        lines.append("Baseline found nothing today. Check the shadow strategies' rejects above for which exact "
                     "gate (score, relative volume, breakout, broker flow) is closest to being cleared -- that's "
                     "the plain-language read on 'what didn't work' today, not a verdict on the strategy itself.\n")

    path = os.path.join(ADAPTIVE_DIR, f"daily_review_{date_str}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    ns.log(f"Daily review written to: {path}")
    return path


# ---------------------------------------------------------------------------
# WEEKLY REPORT + PROMOTION / STRICT-TRIGGER LOGIC
# ---------------------------------------------------------------------------
def _fmt_metric(m: dict, key: str, pct=False, r=False) -> str:
    if key not in m or m.get("trades_completed", 0) == 0:
        return "n/a"
    v = m[key]
    if v == float("inf"):
        return "inf"
    if pct:
        return f"{v*100:.1f}%"
    if r:
        return f"{v:+.2f}R"
    return f"{v:+.0f}"


def write_weekly_review(week_ending, journal_df: pd.DataFrame, state: dict):
    ensure_dirs()
    date_str = str(pd.Timestamp(week_ending).date())
    metrics = {vid: compute_strategy_metrics(journal_df, vid) for vid in STRATEGY_VERSIONS}
    base_m = metrics[BASELINE_ID]

    lines = [f"# Weekly Strategy Review -- week ending {date_str}\n",
             "_Paper-trading research only. No real orders are placed. Six months of price data and a few "
             "weeks of paper trading are NOT enough to prove an edge -- treat every number below as "
             "preliminary until sample sizes are much larger._\n"]

    comparison_rows = []
    for vid, cfg in STRATEGY_VERSIONS.items():
        m = metrics[vid]
        lines.append(f"## {cfg['label']} (`{vid}`)\n")
        if m.get("trades_completed", 0) == 0:
            lines.append("No completed paper trades yet for this version.\n")
        else:
            lines.append(f"- Completed trades: {m['trades_completed']}")
            lines.append(f"- Win rate: {_fmt_metric(m,'win_rate',pct=True)}")
            lines.append(f"- Avg win / avg loss: {_fmt_metric(m,'avg_win')} / {_fmt_metric(m,'avg_loss')}")
            lines.append(f"- Expectancy: {_fmt_metric(m,'expectancy_r',r=True)}")
            lines.append(f"- Profit factor: {m['profit_factor']:.2f}" if m['profit_factor'] != float('inf') else "- Profit factor: inf (no losing trades yet)")
            lines.append(f"- Max drawdown (paper P&L): {_fmt_metric(m,'max_drawdown')}")
            lines.append(f"- Avg holding period: {m['avg_holding_period']:.1f} sessions")
            lines.append(f"- Net P&L after costs: {_fmt_metric(m,'net_pnl_total')}\n")
        comparison_rows.append({
            "week_ending": date_str, "version_id": vid,
            "trades_completed": m.get("trades_completed", 0),
            "win_rate": m.get("win_rate"), "avg_win": m.get("avg_win"), "avg_loss": m.get("avg_loss"),
            "expectancy_r": m.get("expectancy_r"), "profit_factor": m.get("profit_factor"),
            "max_drawdown": m.get("max_drawdown"), "avg_holding_period": m.get("avg_holding_period"),
            "net_pnl_total": m.get("net_pnl_total"),
        })

    # Rule 5: strict-shadow highlighting trigger. Never touches baseline.
    lines.append("## Baseline Health Check (Rule 5)\n")
    trigger_now = (base_m.get("trades_completed", 0) >= 20 and base_m.get("profit_factor", 99) < 1.0
                   and (base_m.get("expectancy_r") or 0) < 0)
    if trigger_now:
        state["consecutive_weekly_underperformance"] = state.get("consecutive_weekly_underperformance", 0) + 1
    else:
        state["consecutive_weekly_underperformance"] = 0

    if trigger_now and state["consecutive_weekly_underperformance"] >= 2:
        lines.append("**Baseline underperformance conditions met for 2+ consecutive weekly reports** "
                     "(>=20 completed trades, profit factor < 1.0, expectancy < 0R). Per rule 5, the baseline "
                     "is left unchanged and the strict shadow strategy is highlighted below for comparison. "
                     "No rule change is applied automatically.\n")
        strict_m = metrics.get("strict_shadow_v1", {})
        if strict_m.get("trades_completed", 0) >= 10:
            lines.append(f"Strict shadow so far: {strict_m['trades_completed']} trades, "
                         f"expectancy {_fmt_metric(strict_m,'expectancy_r',r=True)}, "
                         f"profit factor {strict_m.get('profit_factor', 0):.2f}. "
                         "Not yet enough completed strict-shadow trades to responsibly recommend a rule "
                         "change -- see Rule 6 (Promotion) below." if strict_m['trades_completed'] < 30 else
                         "See Rule 6 (Promotion) below for whether this now clears the promotion bar.")
        else:
            lines.append("Strict shadow doesn't yet have enough completed trades to diagnose against -- "
                         "continuing to paper-track it. A specific rule-change recommendation requires enough "
                         "strict-shadow trades to compare meaningfully, not just a hunch.")
    elif trigger_now:
        lines.append("Baseline underperformance conditions are met this week for the first time. Per rule 5 "
                     "this needs to persist across **two consecutive** weekly reports before the strict shadow "
                     "is specifically highlighted -- continuing to monitor, baseline unchanged.\n")
    else:
        lines.append("Baseline underperformance conditions not currently met (needs >=20 completed trades, "
                     "profit factor < 1.0, and expectancy < 0R, sustained for 2 weekly reports). No action.\n")

    # Rule 6: promotion check for each shadow strategy independently.
    lines.append("## Promotion Check (Rule 6)\n")
    any_proposed = False
    for vid, cfg in STRATEGY_VERSIONS.items():
        if cfg["kind"] == "baseline":
            continue
        m = metrics[vid]
        n = m.get("trades_completed", 0)
        if n < 30:
            lines.append(f"- **{vid}**: {n}/30 completed trades -- not yet eligible for promotion "
                         f"consideration.")
            continue
        pf_gap = (m.get("profit_factor", 0) or 0) - (base_m.get("profit_factor", 0) or 0)
        dd_ok = (base_m.get("trades_completed", 0) == 0) or \
                (abs(m.get("max_drawdown", 0) or 0) <= abs(base_m.get("max_drawdown", 0) or 0) * 1.10)
        expectancy_ok = (m.get("expectancy_r") or -1) > 0
        eligible = expectancy_ok and pf_gap >= 0.15 and dd_ok
        lines.append(f"- **{vid}**: {n} trades, expectancy {_fmt_metric(m,'expectancy_r',r=True)}, "
                     f"PF gap vs baseline {pf_gap:+.2f}, drawdown check {'OK' if dd_ok else 'FAILED'} "
                     f"-- {'MEETS promotion criteria' if eligible else 'does not yet meet promotion criteria'}.")
        if eligible:
            any_proposed = True
            write_proposal(vid, cfg, m, base_m, date_str)

    path = os.path.join(ADAPTIVE_DIR, f"weekly_strategy_review_{date_str}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    ns.log(f"Weekly review written to: {path}")

    comp_df = pd.DataFrame(comparison_rows)
    header_needed = not os.path.exists(COMPARISON_PATH)
    comp_df.to_csv(COMPARISON_PATH, mode="a", header=header_needed, index=False)

    return path, any_proposed


def write_proposal(vid: str, cfg: dict, shadow_metrics: dict, base_metrics: dict, date_str: str):
    """Writes proposed_strategy_change.md. NEVER alters the active baseline
    -- this is a proposal for a human to review and approve or reject."""
    lines = [
        f"# Proposed Strategy Change -- {date_str}\n",
        f"**Proposal: promote `{vid}` ({cfg['label']}) alongside/over `{BASELINE_ID}`.**\n",
        "This file is a PROPOSAL only. The active baseline has NOT been changed. Review the evidence below "
        "and decide manually whether to approve.\n",
        "## Evidence\n",
        f"- Completed out-of-sample paper trades: {shadow_metrics['trades_completed']} (threshold: 30)",
        f"- Expectancy: {_fmt_metric(shadow_metrics,'expectancy_r',r=True)} (must be positive after costs)",
        f"- Profit factor: {shadow_metrics.get('profit_factor',0):.2f} vs baseline "
        f"{base_metrics.get('profit_factor',0):.2f} (gap: "
        f"{(shadow_metrics.get('profit_factor',0) or 0) - (base_metrics.get('profit_factor',0) or 0):+.2f}, "
        f"threshold: +0.15)",
        f"- Max drawdown: {_fmt_metric(shadow_metrics,'max_drawdown')} vs baseline "
        f"{_fmt_metric(base_metrics,'max_drawdown')} (must not be worse by more than 10%)\n",
        "## Risks / caveats\n",
        "- This comparison has not yet been confirmed across multiple distinct market regimes or time "
        "windows -- verify the trade list spans more than one narrow period before approving.\n"
        "- Paper-trading P&L uses this engine's own cost-model defaults (see adaptive_engine.py docstring), "
        "not confirmed real brokerage rates.\n",
        "## Exact configuration difference vs baseline\n",
        f"```json\n{json.dumps({k: v for k, v in cfg.items() if k not in ('label','kind','created','reason')}, indent=2)}\n```\n",
        "## To approve\n",
        f"Manually create a new baseline version (e.g. `baseline_v2`) in `STRATEGY_VERSIONS` in "
        f"adaptive_engine.py with this configuration, add it to the experiment registry with a reason "
        f"referencing this file, and leave `{vid}` and `{BASELINE_ID}` in the registry unchanged (append-only "
        f"-- never delete prior versions' results).\n",
        "## To reject\n",
        f"No action needed -- `{vid}` keeps running as a shadow strategy, and this file can be left in place "
        f"as a record of the evidence considered.\n",
    ]
    with open(PROPOSAL_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    ns.log(f"Promotion proposal written to: {PROPOSAL_PATH} (baseline NOT changed automatically)")


# ---------------------------------------------------------------------------
# RULE 4 -- 5 consecutive no-candidate baseline sessions in a positive regime
# ---------------------------------------------------------------------------
def write_relaxed_shadow_watchlist(signal_date, baseline_rejects: list, relaxed_candidates: list):
    ensure_dirs()
    date_str = str(pd.Timestamp(signal_date).date())
    reject_by_symbol = {r["symbol"]: r["reasons_failed"] for r in baseline_rejects}
    lines = [f"# Relaxed Shadow Watchlist -- {date_str}\n",
             "Baseline has produced no candidate for 5+ consecutive sessions while the market regime is "
             "positive. Per rule 4, baseline recommendations are NOT being replaced -- this is a separate "
             "watchlist from the relaxed shadow strategy only, tracked as paper trades, not promoted.\n"]
    if not relaxed_candidates:
        lines.append("The relaxed shadow strategy also found nothing today.\n")
    else:
        for c in relaxed_candidates:
            blockers = reject_by_symbol.get(c["symbol"], ["not evaluated under baseline"])
            lines.append(f"## {c['symbol']} ({c['sector']})\n")
            lines.append(f"Qualifies under the relaxed shadow strategy (AdaptiveScore {c['adaptive_score']}/100). "
                         f"Exactly why baseline rejected it:\n")
            for b in blockers:
                lines.append(f"- {b}")
            lines.append("")
    path = os.path.join(ADAPTIVE_DIR, f"relaxed_shadow_watchlist_{date_str}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    ns.log(f"Relaxed shadow watchlist written to: {path}")
    return path


# ---------------------------------------------------------------------------
# ORCHESTRATOR -- called from nepse_screener.main() when --adaptive is passed.
# ---------------------------------------------------------------------------
def run_adaptive_cycle(headless: bool = True, debug: bool = False):
    ns.log("\n=== Adaptive Evaluation Engine (paper trading / research only) ===")
    ensure_dirs()
    ensure_registry()
    state = load_state()

    sector_map = ns.fetch_sector_map()
    all_df = ns.load_history()
    latest = ns.compute_indicators(all_df)
    latest = ns.attach_sectors(latest, sector_map)
    latest = ns.finalize_scores(latest)
    market_up_ratio = latest.attrs.get("market_up_ratio", 0.5)

    extra_hist = build_full_indicator_history(all_df)
    signal_frame = build_signal_frame(latest, extra_hist)

    floorsheet_df = ns.load_latest_floorsheet()
    floorsheet_history = load_floorsheet_history()

    download_index_missing(headless=headless, debug=debug)
    index_market, index_sector = load_index_history(all_df, sector_map)

    journal_df = load_journal()
    journal_df = evaluate_pending_entries(journal_df, extra_hist, index_market, index_sector)
    journal_df = walk_open_trades(journal_df, extra_hist)

    signal_date = latest["Business Date"].max()
    signal_date_str = str(pd.Timestamp(signal_date).date())
    already_processed = state.get("last_signal_date_processed") == signal_date_str

    results_by_version = {}
    for vid in STRATEGY_VERSIONS:
        candidates, rejects = generate_candidates(vid, signal_frame, all_df, sector_map, floorsheet_df, floorsheet_history)
        results_by_version[vid] = {"candidates": candidates, "rejects": rejects}

        if not already_processed:
            journal_df = open_new_trades(candidates, vid, signal_date, journal_df)
            if not candidates:
                if market_up_ratio < 0.45:
                    category = "poor_conditions"
                elif results_by_version.get("relaxed_shadow_v1", {}).get("candidates"):
                    category = "rules_may_be_restrictive"
                else:
                    category = "insufficient_data" if latest["DaysOfHistory"].max() < 20 else "poor_conditions"
                log_no_trade(signal_date_str, vid, category,
                             f"No symbol cleared all gates for {STRATEGY_VERSIONS[vid]['label']} today.")

    # Rule 4
    if not already_processed:
        base_candidates = results_by_version[BASELINE_ID]["candidates"]
        if not base_candidates:
            state["baseline_no_candidate_streak"] = state.get("baseline_no_candidate_streak", 0) + 1
        else:
            state["baseline_no_candidate_streak"] = 0

        if state["baseline_no_candidate_streak"] >= 5 and market_up_ratio > 0.5:
            write_relaxed_shadow_watchlist(signal_date, results_by_version[BASELINE_ID]["rejects"],
                                           results_by_version["relaxed_shadow_v1"]["candidates"])

    save_journal(journal_df)
    write_daily_review(signal_date, results_by_version, journal_df, market_up_ratio)

    iso_week = f"{signal_date.isocalendar()[0]}-W{signal_date.isocalendar()[1]:02d}"
    if state.get("last_weekly_iso_week") != iso_week:
        write_weekly_review(signal_date, journal_df, state)
        state["last_weekly_iso_week"] = iso_week

    state["last_signal_date_processed"] = signal_date_str
    save_state(state)

    ns.log("=== Adaptive cycle complete. Paper trading only -- no orders were placed. ===\n")
    return results_by_version, journal_df
