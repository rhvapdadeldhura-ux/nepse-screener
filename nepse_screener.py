"""
NEPSE Sector-Wise Screener
---------------------------
Run this any time. Each run:
  1. Checks what daily price CSVs you already have locally
  2. Downloads every missing trading day up to today (via a real browser,
     since NEPSE blocks plain requests) -- automatically detects and skips
     non-trading days (weekends and holidays), without hardcoding which
     days those are, since NEPSE's weekend schedule has changed before
     (Fri+Sat in the past, Sat+Sun currently) and may change again
  3. Downloads today's broker-wise floorsheet from nepalstock.com/floor-sheet
     (500 rows/page, paging through until Next is grayed out) and saves it
     to floorsheet_data/ -- runs automatically every time, since the site
     only ever shows the current trading day (no historical lookup), so
     this is the only way to build up floorsheet history over time. Skips
     cleanly if you already have that day's file.
  4. Refreshes the sector map (symbol -> sector) if it's stale, which
     also naturally picks up any new IPOs that started trading
  5. Rebuilds indicators (RSI, MACD, moving averages, volume trend) from
     all the history you've accumulated
  6. Writes report.html -- a sector-by-sector ranked shortlist you open
     in your browser

SETUP (run once):
    pip install -r requirements.txt
    playwright install chromium

USAGE:
    python nepse_screener.py                  # update price + floorsheet data, rebuild report
    python nepse_screener.py --show            # watch the browser while it downloads
    python nepse_screener.py --refresh-sectors # force sector map refresh
    python nepse_screener.py --skip-download   # skip price download, just rebuild report
    python nepse_screener.py --skip-floorsheet # skip today's floorsheet download
    python nepse_screener.py --leaderboard     # also run the Phase 1-4 rule-based 100-mark leaderboard
                                                # (needs floorsheet data for Rule 1/2 to score above 0 --
                                                # currently only verified end-to-end against RLFL, since
                                                # that's the only symbol with floorsheet history so far;
                                                # every other symbol will just show its Sector Marks until
                                                # more floorsheet days accumulate)

START_DATE is set to 2025-08-24 (earliest date confirmed available). If
that turns out to be wrong, edit the constant below.

LEADERBOARD MODE (--leaderboard) implements a separate, deterministic
100-mark scoring system on top of the existing report: 20 marks for sector
rotation, 40 for single-broker delivery concentration (from the floorsheet),
40 for a late-session breakout approximation (EMA20 + turnover surge +
Contract No. trade-sequence). Only stocks scoring >=60 make the leaderboard.
This is UNVERIFIED against real floorsheet column names -- the floorsheet
scraper writes whatever headers NEPSE's page actually has, and this mode
looks for columns containing "symbol", "buyer", "seller", "quantity",
"rate", and "contract" (case-insensitive). If NEPSE's real headers don't
match, --leaderboard will print which expected columns it couldn't find --
share that and the fix is a one-line tweak to _find_col's keyword lists.

FLOORSHEET: nepalstock.com/floor-sheet has no date picker and no per-symbol
filter -- it always shows exactly one trading day's data: the previous day's
until trading opens (~11 AM), and the current day's after that. There's no
way to backfill past days, so floorsheet_data/ can only ever grow one file
per run, going forward from whenever you started running this. This part
(the rows-per-page selector and the Next-button detection) is UNVERIFIED
against the live page -- if a run reports 0 rows or fails outright, re-run
once with `--debug --show` and check the screenshots saved to
floorsheet_data/debug_*.png; the fix is almost always a one-line selector
change in download_floorsheet_for_date().
"""

import argparse
import datetime as dt
import glob
import json
import os
import sys
import time
import webbrowser

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# CONFIG -- edit these as needed
# ---------------------------------------------------------------------------
START_DATE = "2025-08-24"   # NEPSE data available from this date onward (per your check)
HOME_URL = "https://www.nepalstock.com/"
TODAY_PRICE_URL = "https://www.nepalstock.com/today-price"   # go straight here for daily price
                                                                # downloads instead of the homepage --
                                                                # the homepage has been loading blank
                                                                # on some networks/CDN edges
SECTOR_SOURCE_URL = "https://merolagani.com/CompanyList.aspx"
SECTOR_REFRESH_DAYS = 7      # re-check for new IPOs / sector changes this often

# Leaderboard mode (--leaderboard) -- see module docstring above.
CIRCUIT_PCT = 0.10   # NEPSE's general equity circuit band (+/-10%). VERIFY this against the
                      # instrument category you're screening -- some categories (e.g. promoter
                      # shares, newly listed IPOs) run tighter bands, and NEPSE has changed this
                      # rule before. Used only for the Opening Day Range Limit in the risk radar.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "nepse_data")
SECTOR_CACHE = os.path.join(BASE_DIR, "sector_map.json")
SECTOR_SEED = os.path.join(BASE_DIR, "sector_seed.json")   # bundled fallback, always available
HOLIDAY_CACHE = os.path.join(BASE_DIR, "holidays.json")
REPORT_PATH = os.path.join(BASE_DIR, "report.html")

# ---- Console-quiet logging -------------------------------------------------
# Per request: the terminal should show nothing but real errors -- all
# routine progress info instead goes into report.html's "Run Log" tab. Both
# functions record to _LOG_MESSAGES so the tab always has the full picture;
# log_error() ALSO prints immediately, since a real problem needs to be seen
# right away, not just after the run finishes and the report opens. This
# does not affect anything gated behind --debug, which is opt-in verbose
# output and stays as direct print() so it appears live while debugging.
_LOG_MESSAGES = []  # list of (level, text) tuples, level is "info" or "error"


def log(msg=""):
    """Routine progress/status message -- recorded for the Run Log tab,
    not printed to the console."""
    _LOG_MESSAGES.append(("info", str(msg)))


def log_error(msg=""):
    """A real problem -- printed immediately AND recorded for the Run Log
    tab, so it's visible both in the moment and in the historical record."""
    text = str(msg)
    print(text)
    _LOG_MESSAGES.append(("error", text))


def _runlog_tab_html() -> str:
    """Renders every log()/log_error() call made during this run into the
    Run Log tab, in order, with errors visually distinguished -- this is
    where all the routine progress output that used to clutter the
    terminal now lives instead."""
    if not _LOG_MESSAGES:
        return "<div class='muted'>Nothing was logged this run.</div>"
    import html as _html
    lines = []
    for level, text in _LOG_MESSAGES:
        escaped = _html.escape(text)
        if level == "error":
            lines.append(f"<span class='runlog-err'>{escaped}</span>")
        else:
            lines.append(escaped)
    return f"<div class='runlog-box'>{chr(10).join(lines)}</div>"


def _launch_browser(p, headless: bool):
    """Prefer your REAL installed Chrome/Edge over Playwright's bundled
    Chromium. If manual browsing works but the automated browser gets
    ERR_EMPTY_RESPONSE or similar, that's usually antivirus/firewall
    software allowing traffic from a recognized browser.exe but blocking
    or interfering with an unfamiliar automation binary. Driving your real
    browser sidesteps that entirely. Falls back to bundled Chromium only if
    neither Chrome nor Edge is installed on this machine."""
    for channel in ("chrome", "msedge"):
        try:
            return p.chromium.launch(headless=headless, channel=channel, args=["--disable-http2"])
        except Exception:
            continue
    log("  (Note: couldn't find a real Chrome/Edge install -- using Playwright's bundled "
          "Chromium instead. If downloads keep failing, installing Google Chrome may help.)")
    return p.chromium.launch(headless=headless, args=["--disable-http2"])


# ---------------------------------------------------------------------------
# STEP 1: figure out which trading days we're missing
# ---------------------------------------------------------------------------
def trading_days(start: dt.date, end: dt.date):
    """Yield every calendar day in range. NEPSE's weekend has changed before
    (Fri+Sat in the past, Sat+Sun now) and may change again, so rather than
    hardcode a weekday rule, every day is attempted and the per-day
    validation step (real data for that date, or not) decides -- confirmed
    non-trading days get cached in holidays.json so repeat runs skip them
    instantly instead of re-guessing the weekend pattern."""
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def existing_dates() -> set:
    os.makedirs(DATA_DIR, exist_ok=True)
    found = set()
    for f in glob.glob(os.path.join(DATA_DIR, "nepse_*.csv")):
        name = os.path.basename(f)
        try:
            found.add(dt.datetime.strptime(name, "nepse_%Y-%m-%d.csv").date())
        except ValueError:
            continue
    return found


def load_known_holidays() -> set:
    if os.path.exists(HOLIDAY_CACHE):
        with open(HOLIDAY_CACHE) as f:
            return set(json.load(f))
    return set()


def save_known_holiday(date_str: str):
    holidays = load_known_holidays()
    holidays.add(date_str)
    with open(HOLIDAY_CACHE, "w") as f:
        json.dump(sorted(holidays), f)


MARKET_DATA_READY_TIME = dt.time(15, 15)  # NEPSE finalizes Close Price after market close (~3 PM) --
                                            # today's CSV export isn't reliably available before this


def find_missing_dates() -> list:
    start = dt.datetime.strptime(START_DATE, "%Y-%m-%d").date()
    today = dt.date.today()
    all_days = set(trading_days(start, today))
    have = existing_dates()
    known_holidays = {dt.datetime.strptime(d, "%Y-%m-%d").date() for d in load_known_holidays()}
    missing = all_days - have - known_holidays

    now = dt.datetime.now()
    if today in missing and now.time() < MARKET_DATA_READY_TIME:
        missing.discard(today)
        ready_time_str = MARKET_DATA_READY_TIME.strftime("%I:%M %p").lstrip("0")  # cross-platform equivalent
                                                                                   # of %-I (which Windows'
                                                                                   # strftime doesn't support)
        log(f"Skipping today ({today.isoformat()}) for now -- NEPSE usually doesn't finalize "
              f"the day's data until after {ready_time_str}. "
              f"Run again later today to pick it up.")

    return sorted(missing)


# ---------------------------------------------------------------------------
# STEP 2: download missing days via a real browser
# ---------------------------------------------------------------------------
def download_missing(missing: list, headless: bool = True):
    if not missing:
        log("No missing trading days -- data is already up to date.")
        return

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log_error("Playwright not installed. Run: pip install playwright  &&  playwright install chromium")
        sys.exit(1)

    log(f"Downloading {len(missing)} missing trading day(s)...")
    os.makedirs(os.path.join(BASE_DIR, "debug_failures"), exist_ok=True)
    with sync_playwright() as p:
        # --disable-http2: nepalstock.com has been hitting ERR_HTTP2_PROTOCOL_ERROR
        # on some networks -- usually antivirus/firewall SSL inspection interfering
        # with HTTP/2 framing. Forcing plain HTTP/1.1 works around it.
        browser = _launch_browser(p, headless)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()

        confirmed_holidays = []
        technical_failures = []
        for d in missing:
            target = d.isoformat()
            # Was 4 separate print()s using end=" " to build one console
            # line ("2026-09-16 ... ok") -- now that this goes to the Run
            # Log tab instead of the terminal, build one string per date
            # instead of relying on print's line-continuation behavior.
            status, reason = _download_one_day(page, target)
            if status == "ok":
                log(f"  {target} ... ok")
            elif status == "holiday":
                log(f"  {target} ... confirmed no trading data -- cached, won't retry")
                confirmed_holidays.append(target)
                save_known_holiday(target)
            else:
                log_error(f"  {target} ... FAILED ({reason}) -- will retry on next run, screenshot saved")
                technical_failures.append(target)
            time.sleep(3)  # be politer to the server -- rapid repeated hits during a
                           # long backfill may be what triggered ERR_EMPTY_RESPONSE

        browser.close()

    if confirmed_holidays:
        log(f"\n{len(confirmed_holidays)} confirmed non-trading day(s): {confirmed_holidays}")
    if technical_failures:
        log_error(f"\n{len(technical_failures)} day(s) failed for technical reasons (NOT cached as holidays -- will retry): {technical_failures}")
        log_error(f"Check debug_failures/ folder for screenshots of what the page looked like.")


def _table_matches_requested_date(page, target_date: str):
    """Checks whether the filtered Today's Price table's own date column
    actually shows the requested date, or whether it's stale data left over
    from a different date. On this site, filtering to a date with no
    trading data doesn't reliably clear the table to empty -- it can just
    silently fail to update and leave whatever was showing before (commonly
    the most recent real trading day's 20 rows), which is exactly what
    produces a table that has rows yet never triggers a download: there's
    nothing new to export for the requested date.

    Returns True if the first row's date matches the request, False if
    there are rows but none match (stale -- almost certainly a non-trading
    day), or None if the check itself couldn't be performed (caller should
    fall back to its other heuristics rather than trust this)."""
    try:
        rows = page.locator("table tbody tr")
        if rows.count() == 0:
            return None
        first_row_text = rows.first.inner_text()
    except Exception:
        return None
    y, m, d = target_date.split("-")
    candidates = {f"{m}/{d}/{y}", f"{int(m)}/{int(d)}/{y}", target_date}
    return any(c in first_row_text for c in candidates)


def _download_one_day(page, target_date: str):
    """Returns (status, reason) where status is 'ok', 'holiday', or 'error'.
    Only 'holiday' gets permanently cached -- that means we successfully
    reached the page, filtered, and the site itself showed no data for that
    date. 'error' means something technical went wrong (selector not found,
    timeout, site slow to load) and should be retried, not treated as a
    holiday."""
    out_path = os.path.join(DATA_DIR, f"nepse_{target_date}.csv")
    debug_path = os.path.join(BASE_DIR, "debug_failures", f"{target_date}.png")
    y, m, d = target_date.split("-")
    display_date = f"{m}/{d}/{y}"

    # Try the direct Today's Price URL first (faster, lighter page). If that
    # keeps getting refused (ERR_EMPTY_RESPONSE suggests the server doesn't
    # like being hit directly on a subpage without a prior visit -- some
    # anti-bot protections expect that normal-browsing pattern), fall back
    # to visiting the homepage first and clicking through, which establishes
    # a normal session before touching the Today's Price page.
    loaded = False
    load_error = None
    for attempt in range(2):
        try:
            page.goto(TODAY_PRICE_URL, wait_until="domcontentloaded", timeout=60000)
            time.sleep(2)
            loaded = True
            break
        except Exception as e:
            load_error = e
            time.sleep(3 + attempt * 2)  # backoff: 3s, then 5s

    if not loaded:
        try:
            page.goto(HOME_URL, wait_until="domcontentloaded", timeout=60000)
            time.sleep(2)
            page.locator("text=Today's Price").first.click(timeout=10000)
            time.sleep(2)
            loaded = True
        except Exception as e:
            load_error = e

    if not loaded:
        try:
            page.screenshot(path=debug_path, full_page=True)
        except Exception:
            pass  # page may genuinely have nothing renderable yet -- don't crash over a failed screenshot
        return "error", f"couldn't load today-price page (direct URL and homepage fallback both failed): {load_error}"

    try:
        # Same priority-ordered fallback list as the original working
        # standalone script -- using only "input.form-control" (as a prior
        # version of this file did) can grab the WRONG input if the page has
        # more than one .form-control element, which silently fills some
        # other field instead of the date picker.
        date_selectors = [
            "input[placeholder*='MM/DD/YYYY']",
            "input.form-control",
            "input[type='text']",
        ]
        field = None
        for sel in date_selectors:
            candidate = page.locator(sel).first
            if candidate.count() > 0:
                field = candidate
                break
        if field is None:
            page.screenshot(path=debug_path, full_page=True)
            return "error", "no date field found on page"

        field.click()
        field.fill("")
        field.fill(display_date)
        time.sleep(0.5)
    except Exception as e:
        page.screenshot(path=debug_path, full_page=True)
        return "error", f"couldn't set date field: {e}"

    try:
        page.locator("text=Filter").first.click(timeout=10000)
        time.sleep(2.5)
    except Exception as e:
        page.screenshot(path=debug_path, full_page=True)
        return "error", f"couldn't click Filter: {e}"

    # Check whether the filtered table actually has rows before even trying
    # to download -- this is the real signal for "holiday", not a download
    # timeout (which could just as easily mean slow page/selector drift).
    try:
        table_rows = page.locator("table tbody tr").count()
    except Exception:
        table_rows = None

    # A non-empty table isn't by itself proof this date has real data --
    # the site can leave a *stale* table on screen (silently failing to
    # update for a date with nothing to show) instead of clearing it to
    # empty. Checking whether the table's own date matches what was
    # requested tells the two cases apart without hardcoding any
    # weekday/weekend assumption.
    date_match = _table_matches_requested_date(page, target_date)

    download_btn = page.locator("text=Download as CSV").first
    if download_btn.count() == 0:
        page.screenshot(path=debug_path, full_page=True)
        if table_rows == 0:
            return "holiday", "table empty, no download button -- genuine non-trading day"
        if date_match is False:
            return "holiday", "table shows a different date's data (filter didn't update) -- likely a non-trading day"
        return "error", "Download as CSV button not found (but table may have data -- check screenshot)"

    try:
        with page.expect_download(timeout=15000) as download_info:
            download_btn.click()
        download_info.value.save_as(out_path)
    except Exception as e:
        page.screenshot(path=debug_path, full_page=True)
        if table_rows == 0:
            return "holiday", "table empty -- genuine non-trading day"
        if date_match is False:
            return "holiday", "table shows stale data from a different date (filter didn't update) -- likely a non-trading day"
        return "error", f"download didn't trigger even though table has {table_rows} row(s): {e}"

    ok = _validate_price_csv(out_path, target_date)
    if ok:
        return "ok", ""
    page.screenshot(path=debug_path, full_page=True)
    return "error", "downloaded file didn't match the requested date -- see screenshot"


def _validate_price_csv(path: str, target_date: str) -> bool:
    """A holiday can sometimes leave a stale/empty export behind even when the
    click 'succeeds' -- confirm the file actually has rows for the requested
    date before trusting it, and delete it if not (so it's retried, not
    silently treated as done)."""
    try:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        if df.empty or "Business Date" not in df.columns:
            os.remove(path)
            return False
        dates_in_file = set(df["Business Date"].astype(str).str[:10])
        if target_date not in dates_in_file:
            os.remove(path)
            return False
        return True
    except Exception:
        if os.path.exists(path):
            os.remove(path)
        return False


# ---------------------------------------------------------------------------
# FLOORSHEET (broker-wise transactions) -- opt-in, since it's page-by-page
# scraping rather than a single CSV download, and each trading day can run
# 100+ pages. Not run by default -- use --floorsheet to include it.
#
# NOT YET VERIFIED against the live page structure (I don't have a way to
# load nepalstock.com from where this was written). Run once with
# --floorsheet --debug --show on a single recent date first and tell me
# what happens -- this almost certainly needs a selector fix or two.
# ---------------------------------------------------------------------------
FLOORSHEET_DIR = os.path.join(BASE_DIR, "floorsheet_data")
COMPANY_DIR = os.path.join(BASE_DIR, "company_data")

# ---------------------------------------------------------------------------
# COMPANY PROFILE (per-symbol deep dive: Financials, Dividend, AGM, Corporate
# Actions) -- on-demand for ONE symbol at a time via --company SYMBOL, since
# doing this for all ~343 stocks would be very slow. Once this is confirmed
# working against the real site for one stock, --company-all runs it for
# everything (meant to be left running in the background).
#
# UNVERIFIED against the live page beyond the one screenshot you shared --
# the search box and tab-click flow especially may need a selector fix.
# Run with --debug --show on RLFL first and tell me what breaks.
# ---------------------------------------------------------------------------
COMPANY_TABS = ["Financials", "Dividend", "Corporate Actions"]  # AGM dropped per your request

KEY_STAT_LABELS = [
    "Instrument Type", "Listing Date", "Last Traded Price", "Total Traded Quantity",
    "Total Trades", "Previous Day Close Price", "High Price / Low Price",
    "52 Week High / 52 Week Low", "Open Price", "Close Price",
    "Total Listed Shares", "Total Paid up Value", "Market Capitalization",
]


def _search_and_open_company(page, symbol: str) -> bool:
    page.goto(HOME_URL, wait_until="domcontentloaded", timeout=60000)
    time.sleep(2)

    search_selectors = [
        "input[placeholder*='Search by Symbol' i]",
        "input[placeholder*='Search' i]",
    ]
    box = None
    for sel in search_selectors:
        candidate = page.locator(sel).first
        if candidate.count() > 0:
            box = candidate
            break
    if box is None:
        return False

    box.click()
    box.fill(symbol)
    time.sleep(1.5)  # let the autocomplete dropdown populate

    # Try clicking a dropdown result matching the symbol; fall back to Enter
    result_selectors = [f"text={symbol}", "li >> visible=true", ".dropdown-item"]
    clicked = False
    for sel in result_selectors:
        try:
            opt = page.locator(sel).first
            if opt.count() > 0:
                opt.click(timeout=4000)
                clicked = True
                break
        except Exception:
            continue
    if not clicked:
        page.keyboard.press("Enter")

    time.sleep(2)
    return True


def _scrape_key_stats(page) -> dict:
    """The left-side info box on the company page is a fixed set of labeled
    rows (Instrument Type, Listing Date, LTP, 52-week range, etc.) -- extract
    each by finding the label text and reading the row it's in."""
    stats = {}
    for label in KEY_STAT_LABELS:
        try:
            row = page.locator(f"text='{label}'").first.locator("..")
            text = row.inner_text()
            value = text.replace(label, "", 1).strip()
            if value:
                stats[label] = value
        except Exception:
            continue
    return stats


def _scrape_tab_table(page, tab_name: str):
    """Click a tab and read whatever table(s) appear -- returns list of
    (headers, rows) tuples, one per table found (a tab could have more than
    one, e.g. a filter row plus a data table)."""
    try:
        page.locator(f"text={tab_name}").first.click(timeout=10000)
        time.sleep(2)
    except Exception:
        return []

    # Same fix as the floorsheet scraper: read every table on the tab (headers
    # + all rows + all cells) in one JS round-trip instead of one Playwright
    # call per cell. --company-all calls this 3x per symbol across ~343
    # symbols, so this matters even though each individual table is small.
    ALL_TABLES_EXTRACT_JS = """
        () => Array.from(document.querySelectorAll('table')).map(table => {
            const headers = Array.from(table.querySelectorAll('thead tr th'))
                .map(th => th.innerText.trim());
            const rows = Array.from(table.querySelectorAll('tbody tr')).map(tr =>
                Array.from(tr.querySelectorAll('td')).map(td => td.innerText.trim())
            ).filter(row => row.length > 0);
            return {headers, rows};
        })
    """
    extracted = page.evaluate(ALL_TABLES_EXTRACT_JS)
    return [(t["headers"], t["rows"]) for t in extracted if t["headers"] or t["rows"]]


def fetch_company_profile(symbol: str, headless: bool = True, debug: bool = False) -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log_error("Playwright not installed.")
        sys.exit(1)

    out_dir = os.path.join(COMPANY_DIR, symbol)
    os.makedirs(out_dir, exist_ok=True)

    log(f"Fetching company profile for {symbol}...")
    with sync_playwright() as p:
        browser = _launch_browser(p, headless)
        page = browser.new_context().new_page()

        if not _search_and_open_company(page, symbol):
            log_error("  Could not find the search box -- check debug_failures/ if --debug was set.")
            if debug:
                os.makedirs(os.path.join(BASE_DIR, "debug_failures"), exist_ok=True)
                page.screenshot(path=os.path.join(BASE_DIR, "debug_failures", f"company_{symbol}_search.png"), full_page=True)
            browser.close()
            return False

        if debug:
            os.makedirs(os.path.join(BASE_DIR, "debug_failures"), exist_ok=True)
            page.screenshot(path=os.path.join(BASE_DIR, "debug_failures", f"company_{symbol}_landed.png"), full_page=True)

        log("  Reading key stats...")
        stats = _scrape_key_stats(page)
        with open(os.path.join(out_dir, "key_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        log(f"    got {len(stats)}/{len(KEY_STAT_LABELS)} fields")

        tab_data = {}
        for tab in COMPANY_TABS:
            log(f"  Reading {tab} tab...")
            tables = _scrape_tab_table(page, tab)
            if debug:
                page.screenshot(path=os.path.join(BASE_DIR, "debug_failures", f"company_{symbol}_{tab.replace(' ', '_')}.png"), full_page=True)
            tab_data[tab] = tables
            for idx, (headers, rows) in enumerate(tables):
                if not rows:
                    continue
                csv_path = os.path.join(out_dir, f"{tab.replace(' ', '_').lower()}_{idx}.csv")
                with open(csv_path, "w", encoding="utf-8") as f:
                    import csv as csv_module
                    writer = csv_module.writer(f)
                    if headers:
                        writer.writerow(headers)
                    writer.writerows(rows)
            log(f"    found {len(tables)} table(s)")

        browser.close()

    _build_company_report(symbol, stats, tab_data, out_dir)
    return True


def _build_company_report(symbol: str, stats: dict, tab_data: dict, out_dir: str):
    stat_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in stats.items())

    tab_sections = []
    for tab, tables in tab_data.items():
        if not tables:
            tab_sections.append(f'<div class="tabsec"><h2>{tab}</h2><div class="empty">No data found -- selectors may need adjusting for this tab.</div></div>')
            continue
        for headers, rows in tables:
            if not rows:
                continue
            head_html = "".join(f"<th>{h}</th>" for h in headers)
            body_html = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
            tab_sections.append(f'''
              <div class="tabsec">
                <h2>{tab}</h2>
                <table><thead><tr>{head_html}</tr></thead><tbody>{body_html}</tbody></table>
              </div>''')

    bonus_note = _summarize_bonus_pattern(tab_data.get("Dividend", []))

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>{symbol} Profile</title>
<style>
body{{background:#0d1117;color:#e6edf3;font-family:Segoe UI,sans-serif;padding:24px;max-width:900px;margin:0 auto;}}
h1{{margin-bottom:4px;}} h2{{font-size:15px;color:#5fb4c9;margin:0 0 8px;}}
.disclaimer{{background:#2a1f14;border:1px solid #4a3a1f;color:#e8c37a;font-size:12.5px;padding:10px 14px;border-radius:8px;margin-bottom:18px;}}
.bonus-note{{background:#132a2f;border:1px solid #1f4a4f;color:#7fd4d9;font-size:13px;padding:10px 14px;border-radius:8px;margin-bottom:18px;}}
.tabsec{{background:#141b24;border:1px solid #26313d;border-radius:10px;padding:14px 18px;margin-bottom:16px;}}
table{{width:100%;border-collapse:collapse;font-size:13px;}}
th{{text-align:left;color:#8b98a5;font-size:11px;text-transform:uppercase;padding:5px 8px;border-bottom:1px solid #26313d;}}
td{{padding:6px 8px;border-bottom:1px solid #1e2731;}}
.empty{{color:#8b98a5;font-size:13px;}}
</style></head><body>
<h1>{symbol}</h1>
<div class="disclaimer">⚠️ Historical data only -- not a prediction. Bonus/dividend patterns shown below reflect the past, not a guarantee of future declarations.</div>
<div class="tabsec"><h2>Key Stats</h2><table>{stat_rows}</table></div>
{bonus_note}
{''.join(tab_sections)}
</body></html>"""

    report_path = os.path.join(out_dir, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    log(f"\nCompany report written to: {report_path}")


def _summarize_bonus_pattern(dividend_tables) -> str:
    """Best-effort: look for a column that mentions 'bonus' and a fiscal-year
    column in whatever the Dividend tab returned, and summarize how often
    and around what time of year bonus has historically been declared.
    This is pattern-surfacing only -- never a forecast."""
    for headers, rows in dividend_tables:
        lower_headers = [h.lower() for h in headers]
        bonus_col = next((i for i, h in enumerate(lower_headers) if "bonus" in h), None)
        year_col = next((i for i, h in enumerate(lower_headers) if "year" in h or "fiscal" in h), None)
        if bonus_col is None or not rows:
            continue
        entries = []
        for row in rows:
            if bonus_col < len(row):
                year = row[year_col] if year_col is not None and year_col < len(row) else "?"
                entries.append(f"{year}: {row[bonus_col]}")
        if entries:
            return (f'<div class="bonus-note"><b>Bonus history (from Dividend tab)</b><br>'
                    + " &middot; ".join(entries[:10])
                    + '<br><span style="font-size:11.5px;">This is what has happened in the past -- not a prediction of what will happen next.</span></div>')
    return ""




FLOORSHEET_URL = "https://nepalstock.com/floor-sheet"
FLOORSHEET_READY_TIME = dt.time(11, 0)  # per your description: the floorsheet page only starts showing
                                          # TODAY's trades once trading opens (~11 AM) -- before that it's
                                          # still showing the previous trading day's sheet, and there's no
                                          # date picker on the page to ask for a specific day either way.


def shot(page, name: str, debug: bool):
    if debug:
        os.makedirs(FLOORSHEET_DIR, exist_ok=True)
        path = os.path.join(FLOORSHEET_DIR, f"debug_{name}.png")
        page.screenshot(path=path, full_page=True)
        print(f"    [debug] screenshot: {path}")


def _floorsheet_target_date() -> dt.date:
    """Cheap, best-effort guess at which trading day's floorsheet
    nepalstock.com is currently showing, used ONLY for the early "do we
    already have this file" skip-check before a browser is even launched
    (see download_floorsheet_missing) -- a clock-based heuristic: before
    ~11 AM (trading hasn't started) it's still showing the previous trading
    day, so file it under the most recent date we already have price data
    for; after 11 AM, file it under today.

    This guess is NOT trusted for the actual saved filename anymore. The
    page shows a real "As of <date>, <time>" label above the floorsheet,
    and the direct-API response carries a real businessDate on every row;
    both download paths (_try_floorsheet_via_api and
    _download_floorsheet_via_clicking, via _real_date_from_floorsheet_rows
    / _read_floorsheet_as_of_date + _correct_floorsheet_out_path) read the
    real date from the page/data itself and rename the output file to match
    if this guess turns out to be wrong."""
    now = dt.datetime.now()
    if now.time() < FLOORSHEET_READY_TIME:
        have = sorted(existing_dates())
        if have:
            return have[-1]
        return now.date() - dt.timedelta(days=1)
    return now.date()


def _real_date_from_floorsheet_rows(rows):
    """Looks for the authoritative businessDate field the floorsheet API
    actually returns on every row (confirmed via a real captured sample
    earlier), rather than trusting the clock-based guess in
    _floorsheet_target_date. Returns a dt.date, or None if no row has a
    parseable businessDate."""
    for row in rows:
        bd = row.get("businessDate")
        if bd:
            try:
                return dt.datetime.strptime(str(bd)[:10], "%Y-%m-%d").date()
            except ValueError:
                continue
    return None


def _read_floorsheet_as_of_date(page):
    """Scrapes the "As of <Month> <Day>, <Year>, <time>" label that
    nepalstock.com shows directly above the floorsheet table and returns the
    trading date it names (as a dt.date), or None if the label can't be
    found/parsed. This is the authoritative source for which day's
    floorsheet is currently showing -- confirmed present on the real page --
    and is used by the click-based fallback scraper, which (unlike the
    direct-API path) has no businessDate field to read the real date from
    the data itself."""
    import re
    try:
        text = page.locator("text=/As of/i").first.inner_text(timeout=5000)
    except Exception:
        return None
    m = re.search(r"As of\s+([A-Za-z]+)\s+(\d{1,2}),?\s*(\d{4})", text)
    if not m:
        return None
    month_str, day_str, year_str = m.groups()
    for fmt in ("%b %d %Y", "%B %d %Y"):
        try:
            return dt.datetime.strptime(f"{month_str} {day_str} {year_str}", fmt).date()
        except ValueError:
            continue
    return None


def _correct_floorsheet_out_path(out_path, real_date, target_date, debug=False):
    """If the real/confirmed trading date differs from the guessed
    target_date originally used to name out_path, returns the corrected
    path (and logs the correction). If a file under the corrected name
    already exists, returns None to signal "nothing left to do, we already
    have it" so the caller can skip re-writing it."""
    if real_date is None or real_date.isoformat() == target_date:
        return out_path
    corrected = os.path.join(os.path.dirname(out_path), f"floorsheet_{real_date.isoformat()}.csv")
    if os.path.exists(corrected):
        log(f"    The page says this floorsheet is actually for {real_date.isoformat()}, not the "
              f"guessed {target_date} -- and we already have that file, so nothing more to do.")
        return None
    log(f"    Correcting floorsheet date: guessed {target_date}, but the page itself says "
          f"{real_date.isoformat()} -- saving under the correct filename.")
    return corrected


def _rows_to_csv(rows, out_path):
    """Write a list of dicts to CSV using the union of every key seen
    across all rows, in first-seen order -- so a field that only shows up
    on a later row doesn't silently get dropped or misalign columns."""
    import csv as csv_module
    fieldnames = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv_module.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _try_floorsheet_via_api(page, target_date, out_path, debug=False, timeout_s=25, page_size=500):
    """Preferred path. A real captured response (see debug_floorsheet_
    response_sample_<date>.json) showed the true schema: this is a
    standard Spring Data pageable endpoint, NOT a single full-day payload
    as first assumed. The response shape is:
        {"floorsheets": {"content": [...], "last": bool,
                          "totalElements": N, "number": pageIndex, ...},
         "totalTrades": N, ...}
    with a default page size of 20 when no page/size query params are
    sent. So this fetches page 0 first (to learn the request URL/body
    NEPSE's own frontend uses and confirm the schema), then walks every
    page directly via HTTP through the browser's authenticated session
    (page.context.request, which shares cookies -- this is what avoids
    NEPSE's block on plain/unauthenticated requests) at a larger page
    size than the site's own default, turning what would be 1000+ pages
    at size 20 into a couple dozen requests at size 500.

    There is no real "Serial Number" in the raw data -- the site's S.N.
    column is just a display index computed on NEPSE's end. contractId is
    the actual unique per-trade identifier, and is used here to dedupe:
    since results are sorted by contractId descending and new trades keep
    getting inserted while the market is open, two page fetches taken
    seconds apart can have their offsets shift underneath them (a row
    that was on page 3 can slide onto page 4 as newer trades push in
    ahead of it). Deduping by contractId means a row that shows up twice
    across pages because of that shift is simply skipped the second time,
    rather than being written to the CSV twice.

    Because the total keeps growing while the market's open, the row
    count from a run started mid-session is a genuine, honest count as of
    that moment -- not a bug -- exactly as already confirmed with the
    click-based scraper. For a final, complete count, run after market
    close as already established.

    Returns True on success (writes out_path). Returns False (never
    raises) to tell the caller to fall back to the click-based scraper."""
    captured = []

    def _on_response(resp):
        try:
            if resp.request.method == "POST" and "nepse-data/floorsheet" in resp.url.lower():
                captured.append({
                    "body": resp.json(),
                    "url": resp.request.url,
                    "post_data": resp.request.post_data,
                    "headers": dict(resp.request.headers),
                })
        except Exception:
            pass

    page.on("response", _on_response)
    try:
        page.goto(FLOORSHEET_URL, wait_until="domcontentloaded", timeout=60000)
        time.sleep(2)

        # The "Filter" click is what actually triggers this request --
        # without it, nothing fires and we time out.
        filter_btn = page.locator("text=Filter").first
        if filter_btn.count() > 0:
            try:
                filter_btn.click(timeout=10000)
            except Exception:
                pass

        deadline = time.time() + timeout_s
        while time.time() < deadline and not captured:
            time.sleep(0.5)

        if not captured:
            if debug:
                print(f"    [api-fetch] no floorsheet API response seen within {timeout_s}s -- "
                      f"falling back to click-based scraping.")
            return False

        first = captured[0]
        first_body = first["body"]
        if debug:
            sample_path = os.path.join(FLOORSHEET_DIR, f"debug_floorsheet_response_sample_{target_date}.json")
            with open(sample_path, "w", encoding="utf-8") as sf:
                json.dump(first_body, sf, indent=2)

        fs = first_body.get("floorsheets") if isinstance(first_body, dict) else None
        if not isinstance(fs, dict) or "content" not in fs:
            if debug:
                print("    [api-fetch] response didn't match the expected {floorsheets: {content: [...]}} "
                      "shape -- falling back to click-based scraping. Check the saved sample file.")
            return False

        # Check the real date as early as possible -- from just this first
        # page's rows -- rather than waiting until the whole day is paged
        # in. If the guessed target_date was wrong AND we already have a
        # file saved under the real date, there's no point spending minutes
        # paginating data we already have; bail out immediately.
        early_real_date = _real_date_from_floorsheet_rows(fs.get("content") or [])
        if early_real_date is not None and early_real_date.isoformat() != target_date:
            already_have = os.path.join(os.path.dirname(out_path), f"floorsheet_{early_real_date.isoformat()}.csv")
            if os.path.exists(already_have):
                log(f"    The floorsheet API's businessDate says this is actually "
                      f"{early_real_date.isoformat()}, not the guessed {target_date} -- and we already "
                      f"have that file, so skipping the download entirely.")
                return True

        # Strip any existing page/size query params from the URL the site
        # itself used, so we can set our own without duplicating them.
        from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
        parts = urlsplit(first["url"])
        base_qs = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                   if k.lower() not in ("page", "size")]
        post_data_str = first["post_data"]
        request_ctx = page.context.request

        # Cookies are already shared automatically via the browser context,
        # but NEPSE's own auth flow (authenticate/prove, refresh-token seen
        # in the network log) suggests a non-cookie token header may also
        # be required. Replay the real request's headers to be safe, minus
        # a few that must be recomputed per-request rather than reused
        # verbatim.
        excluded_headers = {"cookie", "content-length", "host"}
        replay_headers = {k: v for k, v in first.get("headers", {}).items()
                           if k.lower() not in excluded_headers}
        replay_headers["Content-Type"] = "application/json"

        def _refresh_auth():
            # The captured auth header (authorization: Salter <token>) was
            # grabbed once at the start and apparently expires partway
            # through a long paginated download -- confirmed on a real run
            # that hit HTTP 401 at page 74 after 37,000 rows. Reloading the
            # floorsheet page and re-clicking Filter makes the site mint a
            # fresh token the same way it did the first time; recapture it
            # from the next matching response and hand it back so the
            # caller can update the shared replay_headers for every
            # subsequent page fetch, not just the one being retried.
            refreshed = []

            def _on_refresh_response(resp):
                try:
                    if resp.request.method == "POST" and "nepse-data/floorsheet" in resp.url.lower():
                        refreshed.append(dict(resp.request.headers))
                except Exception:
                    pass

            page.on("response", _on_refresh_response)
            try:
                page.goto(FLOORSHEET_URL, wait_until="domcontentloaded", timeout=60000)
                time.sleep(2)
                btn = page.locator("text=Filter").first
                if btn.count() > 0:
                    try:
                        btn.click(timeout=10000)
                    except Exception:
                        pass
                deadline = time.time() + 15
                while time.time() < deadline and not refreshed:
                    time.sleep(0.5)
            finally:
                try:
                    page.remove_listener("response", _on_refresh_response)
                except Exception:
                    pass

            if not refreshed:
                return None
            new_headers = {k: v for k, v in refreshed[0].items() if k.lower() not in excluded_headers}
            new_headers["Content-Type"] = "application/json"
            return new_headers

        def fetch_page(page_num, retries=3):
            nonlocal replay_headers
            qs = [("page", str(page_num)), ("size", str(page_size))] + base_qs
            url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(qs), ""))
            last_error = None
            for attempt in range(1, retries + 1):
                try:
                    resp = request_ctx.post(url, data=post_data_str, headers=replay_headers)
                except Exception as e:
                    # Transient network errors (e.g. a dropped connection --
                    # "socket hang up" -- seen on a real run after 23
                    # successful pages) shouldn't sink an otherwise-working
                    # run. Retry a few times with a short pause before
                    # giving up on this page.
                    last_error = e
                    if debug:
                        print(f"    [api-fetch] page {page_num} attempt {attempt}/{retries} "
                              f"hit a network error ({e}) -- retrying." if attempt < retries
                              else f"    [api-fetch] page {page_num} attempt {attempt}/{retries} "
                                   f"hit a network error ({e}) -- giving up on this page.")
                    time.sleep(2)
                    continue
                if resp.status == 401 and attempt < retries:
                    # The token's expired mid-download -- re-auth and retry
                    # this same page with the fresh token before giving up.
                    if debug:
                        print(f"    [api-fetch] page {page_num} got HTTP 401 (token expired) -- "
                              f"re-authenticating and retrying.")
                    fresh = _refresh_auth()
                    if fresh:
                        replay_headers = fresh
                    else:
                        if debug:
                            print(f"    [api-fetch] re-auth didn't capture a fresh token -- "
                                  f"retrying with the old one anyway.")
                    time.sleep(1)
                    continue
                if not resp.ok:
                    if debug:
                        try:
                            body_preview = resp.text()[:300]
                        except Exception:
                            body_preview = "<couldn't read body>"
                        print(f"    [api-fetch] page {page_num} request failed: HTTP {resp.status} "
                              f"{resp.status_text} -- {body_preview}")
                    return None
                try:
                    return resp.json()
                except Exception:
                    return None
            return None

        all_rows = []
        seen_contract_ids = set()
        last_total_elements = fs.get("totalElements")
        page_num = 0
        max_pages_guard = 5000  # sanity ceiling only -- normal days stop via the "last" flag long before this

        while page_num <= max_pages_guard:
            body = fetch_page(page_num)
            if body is None:
                if debug:
                    print(f"    [api-fetch] request for page {page_num} failed -- stopping with "
                          f"{len(all_rows)} row(s) collected so far.")
                break
            fs2 = body.get("floorsheets") or {}
            content = fs2.get("content") or []
            if not content:
                break

            new_this_page = 0
            for row in content:
                cid = row.get("contractId")
                if cid is not None:
                    if cid in seen_contract_ids:
                        continue  # already have this trade -- offset shifted under us, not a new row
                    seen_contract_ids.add(cid)
                all_rows.append(row)
                new_this_page += 1

            last_total_elements = fs2.get("totalElements", last_total_elements)
            is_last = fs2.get("last", False)
            page_num += 1

            if is_last:
                break
            if new_this_page == 0:
                if debug:
                    print(f"    [api-fetch] page {page_num} had no rows we hadn't already seen, "
                          f"and wasn't flagged as the last page -- stopping here to be safe.")
                break
            time.sleep(0.3)  # brief pacing between requests -- cheap insurance against
                              # tripping a rate limit or connection reset on a long run

        if not all_rows:
            if debug:
                print("    [api-fetch] first page came back empty -- falling back to click-based scraping.")
            return False

        # No real serial number exists in the raw data (see docstring) --
        # add our own sequential reference column rather than pretending
        # to reproduce the site's display-only S.N.
        for i, row in enumerate(all_rows, start=1):
            row["rowNumber"] = i

        # The caller's target_date is only ever a guess (see
        # _floorsheet_target_date) -- every row carries the real,
        # authoritative businessDate straight from NEPSE's own API, so use
        # that to file the CSV under the correct date instead of trusting
        # the guess.
        real_date = _real_date_from_floorsheet_rows(all_rows)
        corrected_path = _correct_floorsheet_out_path(out_path, real_date, target_date, debug=debug)
        if corrected_path is None:
            return True
        out_path = corrected_path

        _rows_to_csv(all_rows, out_path)
        log(f"    saved {len(all_rows)} row(s) via direct API paging -> {os.path.basename(out_path)}")
        if last_total_elements and len(all_rows) < last_total_elements:
            log(f"    NOTE: the server's last-seen total was {last_total_elements} trades, "
                  f"{last_total_elements - len(all_rows)} more than what got saved -- normal if the "
                  f"market was still live during this run (new trades arrived mid-download); "
                  f"re-run after market close for the final, complete count.")
        return True

    except Exception as e:
        if debug:
            print(f"    [api-fetch] error during direct capture: {e} -- falling back to click-based scraping.")
        return False
    finally:
        try:
            page.remove_listener("response", _on_response)
        except Exception:
            pass



def download_floorsheet_for_date(page, target_date: str, debug: bool = False, max_pages: int = 400) -> bool:
    """Downloads the current day's floorsheet. Tries the direct API capture
    first (see _try_floorsheet_via_api) since that gets the whole day in
    one shot with nothing that can stop early. Only falls back to clicking
    through the UI (see _download_floorsheet_via_clicking) if that doesn't
    pan out for some reason."""
    os.makedirs(FLOORSHEET_DIR, exist_ok=True)
    out_path = os.path.join(FLOORSHEET_DIR, f"floorsheet_{target_date}.csv")
    if os.path.exists(out_path):
        log(f"  Already have floorsheet for {target_date}, skipping.")
        return True

    if _try_floorsheet_via_api(page, target_date, out_path, debug=debug):
        return True

    log("    Direct API capture didn't pan out this run -- falling back to the click-based scraper.")
    return _download_floorsheet_via_clicking(page, target_date, out_path, debug=debug, max_pages=max_pages)


def _download_floorsheet_via_clicking(page, target_date: str, out_path: str, debug: bool = False, max_pages: int = 400) -> bool:
    """FALLBACK ONLY -- see download_floorsheet_for_date. Scrapes
    nepalstock.com/floor-sheet by clicking through the UI: set rows-per-page
    to 500, apply the filter, then read the table and click Next repeatedly
    until the Next control is disabled/grayed out (that's the expected end
    of data, not an error). There's no per-symbol or per-date filtering on
    this page -- it always shows whichever single trading day's floorsheet
    is currently published, which is why target_date is only used for the
    output filename, not for anything typed into the page.

    UNVERIFIED against the live page -- the rows-per-page control and the
    Next-button selectors below are best guesses at nepalstock.com's actual
    markup. Run once with --debug --show and check the screenshots in
    floorsheet_data/ if this doesn't work; the fix is almost always a
    one-line selector change.

    Termination does NOT rely solely on the Next button's disabled state
    (that check turned out to be unreliable, and was the root cause of a
    bug where an undetected stuck/wrapping paginator produced ~100,000
    duplicated rows against a real ~99-page, ~49,500-row floorsheet).

    The actual stop condition is the row-level Serial Number column
    (S.N., always 1, 2, 3, ... with no gaps or repeats per your
    description). Each page's serial numbers are checked against every
    serial number collected so far; if a page's serials have already been
    seen, that page is a genuine repeat (stuck/wrapped paginator or a
    true last page) and is discarded without stopping the run short --
    the loop only stops once the *next* page turns out to be a repeat, or
    the Next button reports disabled, or max_pages is hit. An earlier
    version used a whole-row content signature (row count + first/last
    row) instead of the serial number; that produced false positives on
    real 99+ page floorsheets whenever two different pages happened to
    share the same first/last row content (common with tied
    price/quantity values), stopping the download dozens of pages early.
    If the serial column can't be found or parsed on a given page, this
    falls back to the old content-signature check for that page only, and
    logs a warning so it's visible in --debug output.

    IMPORTANT: on two separate real runs, this stopped dozens of pages
    early on a "repeated_page_detected_via_serial" verdict against a day
    that, in fact, had many more pages -- because the site was still mid-
    session and briefly slow/stalled between clicks, which looked
    identical to a genuine repeat within the old retry budget. A later run
    on the SAME (by-then closed) session correctly went all the way via
    next_button_disabled. So next_button_disabled has proven reliable;
    repeated-serial has not, on its own, been reliable enough to trust
    quickly. This version gives a repeated-serial verdict much more
    benefit of the doubt before accepting it (higher retry budget, longer
    per-attempt wait, longer cooldown between retries) than it did in the
    runs that failed -- next_button_disabled can still end the loop
    immediately, since that signal has held up in practice."""
    # --- API SNIFFER (debug only) --------------------------------------
    # The click-and-poll loop below can't reliably tell "genuinely repeated
    # page" apart from a site quirk that merely LOOKS like a repeat once
    # you're dozens of pages in -- that's almost certainly why real 100+
    # page floorsheets have been stopping around page 20-something. The
    # real fix is to stop clicking the UI and call whatever JSON endpoint
    # the page itself calls when you click Next. This listener captures
    # every XHR/fetch response that looks floorsheet-related so you don't
    # have to go digging in DevTools by hand -- it writes them to
    # floorsheet_data/debug_api_calls_<date>.json. After a --debug --show
    # run, open that file and share its contents; that's what's needed to
    # rewrite this function to hit the API directly instead of clicking.
    #
    # SPECIAL CASE: the actual floorsheet data POST
    # (nepse-data/floorsheet) is tracked separately, WITHOUT dedup, and
    # its response body's shape is captured once -- because the working
    # hypothesis is that this single call returns the ENTIRE day's
    # floorsheet already (no page/size params in the request at all), and
    # the "pages" the UI shows afterward are just it re-slicing that array
    # client-side. If that's right, the fix isn't a better repeat-detector
    # -- it's deleting the click loop and parsing this response directly.
    captured_calls = []
    floorsheet_call_log = []       # every hit on the floorsheet endpoint, no dedup
    floorsheet_sample_saved = [False]  # mutable flag so the closure can set it
    if debug:
        def _on_response(resp):
            try:
                url = resp.url.lower()
                rtype = getattr(resp.request, "resource_type", "")

                if "nepse-data/floorsheet" in url:
                    entry = {
                        "seq": len(floorsheet_call_log) + 1,
                        "method": resp.request.method,
                        "url": resp.url,
                        "post_data": resp.request.post_data,
                        "status": resp.status,
                    }
                    if not floorsheet_sample_saved[0]:
                        try:
                            body = resp.json()
                            if isinstance(body, dict):
                                entry["response_top_level_keys"] = sorted(body.keys())
                                # look one level down for the actual row array --
                                # common shapes: {"content": [...]}, {"floorsheets":
                                # {"content": [...]}}, or a bare top-level list.
                                for k, v in body.items():
                                    if isinstance(v, list):
                                        entry[f"response.{k}_length"] = len(v)
                                    elif isinstance(v, dict):
                                        for k2, v2 in v.items():
                                            if isinstance(v2, list):
                                                entry[f"response.{k}.{k2}_length"] = len(v2)
                            elif isinstance(body, list):
                                entry["response_is_bare_list_length"] = len(body)
                            sample_path = os.path.join(
                                FLOORSHEET_DIR, f"debug_floorsheet_response_sample_{target_date}.json"
                            )
                            with open(sample_path, "w", encoding="utf-8") as sf:
                                json.dump(body, sf, indent=2)
                            entry["full_response_saved_to"] = os.path.basename(sample_path)
                            floorsheet_sample_saved[0] = True
                        except Exception as parse_err:
                            entry["response_parse_error"] = str(parse_err)
                    floorsheet_call_log.append(entry)
                    return  # don't also fall into the generic bucket below

                if rtype in ("xhr", "fetch") and any(
                    s in url for s in ("floorsheet", "floor-sheet", "nots", "api")
                ):
                    captured_calls.append({
                        "method": resp.request.method,
                        "url": resp.url,
                        "post_data": resp.request.post_data,
                        "status": resp.status,
                        "resource_type": rtype,
                    })
            except Exception:
                pass
        page.on("response", _on_response)

    def _flush_captured_calls():
        # Pulled out so it can run on EVERY exit path (success, empty
        # table, or exception) -- not just the happy path. A failed run is
        # exactly when you most need to see what the network actually did.
        if not debug:
            return
        if floorsheet_call_log:
            fs_log_path = os.path.join(FLOORSHEET_DIR, f"debug_floorsheet_calls_{target_date}.json")
            with open(fs_log_path, "w", encoding="utf-8") as f:
                json.dump(floorsheet_call_log, f, indent=2)
            print(f"    [debug] the floorsheet data endpoint was hit {len(floorsheet_call_log)} time(s) "
                  f"this run -> details in {os.path.basename(fs_log_path)}. If that number is 1 no matter "
                  f"how many pages were clicked, the whole day's data came back in that one call and the "
                  f"click loop is unnecessary.")
        else:
            print("    [debug] the floorsheet data endpoint (nepse-data/floorsheet) was never hit -- "
                  "check debug_*.png to see what the page actually loaded.")
        if not captured_calls:
            print("    [debug] no other XHR/fetch calls matched the floorsheet/api keyword filter.")
            return
        seen_keys = set()
        deduped = []
        for c in captured_calls:
            key = (c["method"], c["url"].split("?")[0])
            if key not in seen_keys:
                seen_keys.add(key)
                deduped.append(c)
        api_log_path = os.path.join(FLOORSHEET_DIR, f"debug_api_calls_{target_date}.json")
        with open(api_log_path, "w", encoding="utf-8") as f:
            json.dump(deduped, f, indent=2)
        print(f"    [debug] captured {len(deduped)} other distinct-looking API call(s) -> "
              f"{os.path.basename(api_log_path)}.")
    # ---------------------------------------------------------------------

    try:
        page.goto(FLOORSHEET_URL, wait_until="domcontentloaded", timeout=60000)
        time.sleep(2)
        shot(page, f"floorsheet_{target_date}_01_page", debug)

        # Set items-per-page to 500 so far fewer pages need to be clicked
        # through. Tries a native <select> first (most common), then a
        # custom dropdown that opens on click and lists options as text.
        per_page_set = False
        for sel in ["select[name*='per' i]", "select[name*='page' i]", "select"]:
            candidate = page.locator(sel).first
            if candidate.count() > 0:
                try:
                    candidate.select_option("500")
                    per_page_set = True
                    time.sleep(0.5)
                    break
                except Exception:
                    continue
        if not per_page_set:
            for sel in ["text=Per Page", "text=Items per page", "div.dropdown:has-text('page')"]:
                candidate = page.locator(sel).first
                if candidate.count() > 0:
                    try:
                        candidate.click(timeout=3000)
                        time.sleep(0.5)
                        page.locator("text=500").first.click(timeout=3000)
                        per_page_set = True
                        time.sleep(0.5)
                        break
                    except Exception:
                        continue
        if debug:
            print(f"    [debug] items-per-page set to 500: {per_page_set}")
        shot(page, f"floorsheet_{target_date}_02_perpage", debug)

        filter_btn = page.locator("text=Filter").first
        if filter_btn.count() > 0:
            try:
                filter_btn.click(timeout=10000)
                time.sleep(2)
            except Exception:
                pass
        shot(page, f"floorsheet_{target_date}_03_filtered", debug)

        # The page itself shows an authoritative "As of <date>, <time>"
        # label above the floorsheet table -- read that now (before the
        # click-and-page loop below, while it's cheap) so the final save
        # can be filed under the real date rather than the guessed one.
        real_date = _read_floorsheet_as_of_date(page)
        if real_date is not None:
            corrected = _correct_floorsheet_out_path(out_path, real_date, target_date, debug=debug)
            if corrected is None:
                _flush_captured_calls()
                return True
            out_path = corrected
        elif debug:
            print("    [debug] couldn't find/parse the 'As of' date label on the page -- "
                  "keeping the guessed filename.")

        # Read the ENTIRE floorsheet -- every page, every row, every cell --
        # in a single JS round-trip to the browser. Previously each page
        # still cost 2 round-trips (one to read the table, one to click Next
        # and poll for the reload), which is ~200 round-trips for a 100-page
        # floorsheet. This async function drives the whole click-read-click
        # loop itself, inside the browser, and only returns to Python once
        # everything is collected -- so the entire download is 1 call.
        FULL_FLOORSHEET_JS = """
            async (maxPages) => {
                // Pick the table that actually looks like a floorsheet
                // (has floorsheet-ish column headers), not just the first
                // <table> on the page -- a wrong pick here was one way the
                // old loop could silently "extract" a table that never
                // changes between pages.
                function extractTable() {
                    const tables = Array.from(document.querySelectorAll('table'));
                    if (tables.length === 0) return null;

                    const floorsheetHints = ['symbol', 'buyer', 'seller', 'contract', 'quantity', 'rate'];
                    let best = null, bestScore = -1;
                    for (const table of tables) {
                        const headerText = Array.from(table.querySelectorAll('thead tr th'))
                            .map(th => th.innerText.trim().toLowerCase());
                        const score = floorsheetHints.filter(h => headerText.some(t => t.includes(h))).length;
                        const bodyRowCount = table.querySelectorAll('tbody tr').length;
                        // Prefer a real header-keyword match; break ties by row count so we
                        // don't accidentally lock onto a near-empty decoy table.
                        if (score > bestScore || (score === bestScore && bodyRowCount > 0 && best &&
                            bodyRowCount > best.querySelectorAll('tbody tr').length)) {
                            best = table;
                            bestScore = score;
                        }
                    }
                    if (!best) best = tables[0];

                    const headers = Array.from(best.querySelectorAll('thead tr th')).map(th => th.innerText.trim());
                    const rows = Array.from(best.querySelectorAll('tbody tr')).map(tr =>
                        Array.from(tr.querySelectorAll('td')).map(td => td.innerText.trim())
                    ).filter(row => row.length > 0);
                    return {headers, rows};
                }

                function findNextButton() {
                    const selectors = [
                        "a[aria-label='Next Page']", "button[aria-label='Next Page']",
                        ".p-paginator-next", "li.pagination-next a",
                    ];
                    for (const sel of selectors) {
                        const el = document.querySelector(sel);
                        if (el) return el;
                    }
                    // last resort: any link/button whose text is exactly "Next"
                    const generic = Array.from(document.querySelectorAll("a, button"));
                    return generic.find(el => el.textContent.trim() === "Next") || null;
                }

                function isDisabled(btn) {
                    if (!btn) return true;
                    if (btn.disabled) return true;
                    // check the button's own class AND its immediate parent's --
                    // some paginator libraries put the disabled marker (e.g.
                    // "p-disabled") on the wrapping <li>/<span>, not the <a>/<button>
                    // itself, which the old single-element check would miss.
                    const cls = ((btn.className || "") + " " + ((btn.parentElement && btn.parentElement.className) || "")).toLowerCase();
                    if (cls.includes("disabled")) return true;
                    if (btn.getAttribute("aria-disabled") === "true") return true;
                    if (btn.parentElement && btn.parentElement.getAttribute("aria-disabled") === "true") return true;
                    return false;
                }

                // Cheap-but-exact content signature for a page. Kept ONLY as
                // a fallback for the (rare) page where the Serial Number
                // column can't be found or parsed -- see findSerialCol
                // below, which is the PRIMARY dedup/stop signal now.
                function signature(extracted) {
                    return extracted.rows.length + "|" + JSON.stringify(extracted.rows[0] || []) + "|"
                        + JSON.stringify(extracted.rows[extracted.rows.length - 1] || []);
                }

                // Find the Serial Number column: NEPSE labels it things like
                // "S.N.", "S.No", "SN", "Sl. No.". Normalize by stripping
                // everything but letters and comparing against known forms,
                // rather than an exact string match, since punctuation/
                // spacing in the header can vary.
                function findSerialColIndex(headers) {
                    const norm = (h) => h.toLowerCase().replace(/[^a-z]/g, "");
                    const candidates = ["sn", "sno", "slno", "serial", "serialno"];
                    for (let i = 0; i < headers.length; i++) {
                        if (candidates.includes(norm(headers[i]))) return i;
                    }
                    return -1;
                }

                // Pull integer serial numbers out of a page's rows using the
                // given column index. Returns null (not an empty array) if
                // ANY row fails to parse cleanly, so the caller can tell
                // "this page's serials are unusable, fall back to content
                // signature" apart from "this page legitimately has zero
                // rows" -- collapsing those two cases was a latent trap.
                function extractSerials(rows, colIdx) {
                    if (colIdx < 0) return null;
                    const out = [];
                    for (const row of rows) {
                        const n = parseInt((row[colIdx] || "").replace(/[^0-9]/g, ""), 10);
                        if (Number.isNaN(n)) return null;
                        out.push(n);
                    }
                    return out;
                }

                const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

                const allRows = [];
                let headers = null;
                let serialColIdx = null;   // resolved once headers are known
                let pageNum = 1;
                const log = [];  // checkpoint every 10 pages, for debug printing back in Python
                const seenSerials = new Set();     // primary dedup: every S.N. value collected so far
                const seenSignatures = new Set();  // fallback dedup, only used when serials are unusable
                let usedFallbackDedup = false;
                let stopReason = "max_pages_reached";

                // Given a freshly-read table, return only the rows that are
                // genuinely new (not already in seenSerials / seenSignatures).
                // This is used BOTH to decide whether a post-click poll has
                // found the real next page, and to decide the page is
                // actually done -- there's only one "is this new?" check in
                // the whole function, so the two can't disagree with each
                // other the way the old wait-loop and dedup-check could.
                function newRowsOf(extracted) {
                    if (headers === null) {
                        headers = extracted.headers;
                        serialColIdx = findSerialColIndex(headers);
                    }
                    const serials = extractSerials(extracted.rows, serialColIdx);
                    if (serials !== null) {
                        const keepIdx = [];
                        for (let i = 0; i < serials.length; i++) {
                            if (!seenSerials.has(serials[i])) keepIdx.push(i);
                        }
                        return {rows: keepIdx.map((i) => extracted.rows[i]),
                                 serials: keepIdx.map((i) => serials[i]), viaSerial: true};
                    }
                    usedFallbackDedup = true;
                    const sig = signature(extracted);
                    if (seenSignatures.has(sig)) return {rows: [], serials: [], viaSerial: false, sig};
                    return {rows: extracted.rows, serials: [], viaSerial: false, sig};
                }

                function commit(fresh) {
                    if (fresh.viaSerial) {
                        for (const s of fresh.serials) seenSerials.add(s);
                    } else if (fresh.sig) {
                        seenSignatures.add(fresh.sig);
                    }
                    allRows.push(...fresh.rows);
                }

                // Page 1: no click needed, just read what's already on screen.
                const firstPage = extractTable();
                if (!firstPage || firstPage.rows.length === 0) {
                    return {headers: [], rows: [], pages: 0, log, stopReason: "empty_table", usedFallbackDedup};
                }
                commit(newRowsOf(firstPage));
                let lastExtracted = firstPage;   // last page we actually landed on, for the
                                                  // still-loading check below
                let repeatRetries = 0;
                const MAX_REPEAT_RETRIES = 15;   // raised from 4 after two real runs stopped
                                                  // 60-100+ pages early on this exact verdict --
                                                  // a genuine repeat now gets many more "maybe it
                                                  // was just a glitch/throttle" second chances
                                                  // before we trust it and stop

                // From here on: click Next, then poll for a page that
                // contains at least one row we haven't seen before -- NOT
                // just "any change from the last page". Two different
                // things can fool a naive "did it change?" check:
                //
                //   1. A transient loading/re-render frame that hasn't
                //      settled yet -- e.g. the grid mid-animation, showing
                //      a partial or reordered SUBSET of the OLD page's own
                //      rows while the new page is still being fetched.
                //      Sampled at the wrong instant, that looks exactly
                //      like "changed, but zero new rows" -- i.e. a repeat
                //      -- even though the real next page hasn't arrived
                //      yet. This is what caused a real 99-page floorsheet
                //      to falsely stop at page 3: a single glimpse of one
                //      such in-between frame was enough to trip the repeat
                //      check.
                //   2. NEPSE briefly re-serving an old page (throttling
                //      rapid Next clicks, or a one-off backend hiccup)
                //      before recovering on a retry.
                //
                // Fix for (1): a "changed" reading is only trusted once it
                // has been STABLE (identical signature) across two
                // consecutive 200ms polls -- an in-between animation frame
                // won't hold still like that, but a settled page (new or
                // repeated) will.
                // Fix for (2): a stable-but-repeated reading doesn't stop
                // the run immediately. It's retried, with a cooldown pause,
                // up to MAX_REPEAT_RETRIES times -- only stopping for real
                // once every retry comes back a repeat too.
                //
                // Stop conditions, checked in this order:
                //   1. Next button is already disabled -- fast path, no
                //      need to click or wait at all.
                //   2. A stable repeat survives every retry.
                //   3. Nothing ever changes before the poll deadline, on
                //      every retry -- the real "reached the end" signal
                //      for sites that never grey out Next at the true
                //      last page.
                //   4. max_pages safety cap.
                while (pageNum <= maxPages) {
                    const nextBtn = findNextButton();
                    if (isDisabled(nextBtn)) { stopReason = "next_button_disabled"; break; }

                    const beforeSig = signature(lastExtracted);

                    try {
                        nextBtn.click();
                    } catch (e) {
                        stopReason = "next_button_click_failed";
                        break;
                    }

                    const deadline = Date.now() + 15000;  // raised from 8000 -- a mid-session
                                                           // slow response is normal, a stuck one isn't
                    let gotNew = false;
                    let repeated = false;
                    let stableSig = null;
                    let stableCount = 0;
                    while (Date.now() < deadline) {
                        await sleep(200);
                        const again = extractTable();
                        if (!again || again.rows.length === 0) { stableSig = null; stableCount = 0; continue; }
                        const curSig = signature(again);
                        if (curSig === beforeSig) {
                            // Still the pre-click page -- genuinely still
                            // loading. Not a "changed" reading at all.
                            stableSig = null;
                            stableCount = 0;
                            continue;
                        }
                        if (curSig === stableSig) {
                            stableCount += 1;
                        } else {
                            stableSig = curSig;
                            stableCount = 1;
                        }
                        if (stableCount < 2) {
                            // Changed from before-click, but only seen once
                            // so far -- could still be a mid-transition
                            // frame. Wait for it to hold still on a second
                            // poll before trusting it either way.
                            continue;
                        }
                        // Settled: two consecutive polls agree on this
                        // content. NOW decide new-vs-repeat by Serial
                        // Number, not by the fact that it changed.
                        const fresh = newRowsOf(again);
                        if (fresh.rows.length > 0) {
                            commit(fresh);
                            lastExtracted = again;
                            gotNew = true;
                            pageNum += 1;
                            if (pageNum % 10 === 0) {
                                log.push(`page ${pageNum}: ${allRows.length} rows so far`
                                    + (fresh.viaSerial ? ` (serials up to ${Math.max(...fresh.serials)})` : " (fallback dedup)"));
                            }
                            break;
                        }
                        repeated = true;
                        break;
                    }

                    if (gotNew) {
                        repeatRetries = 0;   // any real progress resets the retry budget
                        continue;
                    }

                    if (repeated) {
                        repeatRetries += 1;
                        if (repeatRetries <= MAX_REPEAT_RETRIES) {
                            log.push(`page ${pageNum + 1}: looked like a repeat (retry `
                                + `${repeatRetries}/${MAX_REPEAT_RETRIES}) -- pausing and trying `
                                + `Next again in case this was a throttle/glitch, not the real end`);
                            await sleep(4000);   // raised from 2500 -- cooldown before re-clicking
                                                  // Next on the same page
                            continue;
                        }
                        stopReason = usedFallbackDedup ? "repeated_page_detected_via_signature_fallback"
                                                        : "repeated_page_detected_via_serial";
                        break;
                    }

                    // Neither new content nor a settled repeat showed up at
                    // all before the deadline -- genuinely stuck, not just
                    // a slow-but-real repeat. Also worth a retry in case
                    // it's a one-off network stall.
                    repeatRetries += 1;
                    if (repeatRetries <= MAX_REPEAT_RETRIES) {
                        log.push(`page ${pageNum + 1}: no response before the poll deadline (retry `
                            + `${repeatRetries}/${MAX_REPEAT_RETRIES}) -- pausing and trying Next again`);
                        await sleep(4000);   // raised from 2500, to match the other cooldown
                        continue;
                    }
                    stopReason = usedFallbackDedup ? "no_new_rows_after_next_signature_fallback"
                                                    : "no_new_rows_after_next_via_serial";
                    break;
                }

                return {headers: headers || [], rows: allRows, pages: pageNum, log, stopReason, usedFallbackDedup};
            }
        """

        result = page.evaluate(FULL_FLOORSHEET_JS, max_pages)
        headers = result["headers"]
        all_rows = result["rows"]
        page_num = result["pages"]
        stop_reason = result.get("stopReason", "unknown")
        used_fallback_dedup = result.get("usedFallbackDedup", False)

        if debug:
            for line in result["log"]:
                print(f"    [debug] {line}")
            print(f"    [debug] floorsheet {target_date}: {page_num} page(s), {len(all_rows)} row(s), "
                  f"stopped because: {stop_reason}")
            shot(page, f"floorsheet_{target_date}_04_last_page", debug)

        if not all_rows:
            _flush_captured_calls()
            return False

        if used_fallback_dedup:
            # The S.N. column couldn't be found/parsed on at least one page,
            # so that page fell back to the old whole-row content signature.
            # Rare, but worth flagging since the signature check is the one
            # known to false-positive on real data (see function docstring).
            log_error(f"    NOTE: the Serial Number column wasn't usable on at least one page, so the less "
                  f"reliable content-signature check filled in for it. Re-run with --debug --show and check "
                  f"floorsheet_data/debug_*.png if the row count below looks short.")

        if stop_reason == "max_pages_reached":
            # The loop only stops here if it never once saw a disabled Next
            # button OR a repeated page within max_pages tries -- i.e. real,
            # non-duplicated pages the whole way. Legitimate on a very heavy
            # trading day, but worth a flag since it's the one case the
            # duplicate-page safety net can't distinguish from something new
            # going wrong.
            log_error(f"    NOTE: hit the {max_pages}-page safety cap without the site ever signaling 'last page' "
                  f"or repeating a page -- {len(all_rows)} rows saved, but if that's far more than you'd "
                  f"expect for one trading day, re-run with --debug --show and check "
                  f"floorsheet_data/debug_*.png.")
        elif stop_reason == "repeated_page_detected_via_serial":
            log(f"    (stopped at page {page_num} after a page's Serial Numbers were all already seen -- "
                  f"this is the expected, correct way this now ends, not an error.)")
        elif stop_reason == "repeated_page_detected_via_signature_fallback":
            log(f"    (stopped at page {page_num} after detecting a repeated page via the fallback content "
                  f"check -- likely correct, but double-check the row count since this check is less reliable "
                  f"than Serial Number matching.)")

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            import csv as csv_module
            writer = csv_module.writer(f)
            writer.writerow(headers or [f"col{i}" for i in range(len(all_rows[0]))])
            writer.writerows(all_rows)
        log(f"    saved {len(all_rows)} row(s) across {page_num} page(s) -> {os.path.basename(out_path)}")
        _flush_captured_calls()
        return True

    except Exception as e:
        log_error(f"    floorsheet error: {e}")
        _flush_captured_calls()
        return False


def download_floorsheet_missing(headless: bool = True, debug: bool = False):
    """Runs on every normal invocation of the script (not opt-in) so
    floorsheet_data/ accumulates one file per trading day over time --
    the site itself never gives you more than the current day, so this is
    the only way to build history for --leaderboard's Rule 1/2. Skips
    cleanly if today's (or the currently-showing day's) file is already
    saved, so re-running the same day is a no-op.

    NOTE: `target` here is only the cheap pre-browser guess (see
    _floorsheet_target_date) used for that skip-check. If it's wrong, this
    check simply won't find an existing file and will redo the download --
    mildly wasteful but not incorrect, since the actual save (inside
    _try_floorsheet_via_api / _download_floorsheet_via_clicking) always
    re-files the result under the real date read from the page/data."""
    target = _floorsheet_target_date().isoformat()
    out_path = os.path.join(FLOORSHEET_DIR, f"floorsheet_{target}.csv")
    if os.path.exists(out_path):
        log(f"Already have floorsheet for {target}.")
        return

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log_error("Playwright not installed -- skipping floorsheet (run: pip install playwright && playwright install chromium)")
        return

    log(f"Downloading floorsheet for {target} (500 rows/page, paging until the last page or a repeated "
          f"page is detected)...")
    with sync_playwright() as p:
        browser = _launch_browser(p, headless)
        page = browser.new_context().new_page()
        ok = download_floorsheet_for_date(page, target, debug=debug)
        if not ok:
            log_error("  Floorsheet download failed -- market may be closed, or a selector needs adjusting "
                  "(re-run with --debug --show and check floorsheet_data/debug_*.png).")
        browser.close()


# ---------------------------------------------------------------------------
# STEP 3: sector map (also catches new IPOs as they appear)
# ---------------------------------------------------------------------------
def _scrape_sector_map_from_merolagani() -> dict:
    """merolagani.com's Company List page is plain server-rendered HTML (unlike
    NEPSE's own site), grouped into collapsible sections per sector -- each
    with a heading link like <a href="#collapse_0">Commercial Banks</a>
    pointing at a container <div id="collapse_0"> that holds the table.

    Matching by POSITION (zipping links and tables in order) is fragile --
    any extra heading or non-tabular section anywhere on the page shifts
    everything after it, which is what produced sectors like "Capital"
    showing up with no (or wrong) stocks. Matching by the actual id
    reference is exact regardless of what else is on the page."""
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    }
    resp = requests.get(SECTOR_SOURCE_URL, headers=headers, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    sector_links = soup.select("a[href^='#collapse_']")
    if not sector_links:
        raise ValueError("Page structure didn't match what's expected -- site may have changed.")

    # id -> sector name, from the heading links themselves
    id_to_name = {}
    for link in sector_links:
        container_id = link.get("href", "").lstrip("#")
        name = link.get_text(strip=True)
        if container_id and name:
            id_to_name[container_id] = name

    mapping = {}
    for container_id, sector_name in id_to_name.items():
        container = soup.find(id=container_id)
        if container is None:
            continue
        table = container.find("table") if container.name != "table" else container
        if table is None:
            continue
        found_any = False
        for row in table.find_all("tr")[1:]:
            cells = row.find_all("td")
            if not cells:
                continue
            symbol = cells[0].get_text(strip=True)
            if symbol:
                mapping[symbol] = sector_name
                found_any = True
        if not found_any:
            log(f"    (note: '{sector_name}' section had no stock rows -- skipped)")

    if not mapping:
        raise ValueError("Matched sector headings but found no stock rows in any of them.")
    return mapping


def fetch_sector_map(force: bool = False) -> dict:
    if not force and os.path.exists(SECTOR_CACHE):
        with open(SECTOR_CACHE) as f:
            cache = json.load(f)
        fetched = dt.datetime.fromisoformat(cache.get("fetched_at", "2000-01-01"))
        if (dt.datetime.now() - fetched).days < SECTOR_REFRESH_DAYS:
            return cache["map"]

    log("Refreshing sector map (this also picks up any new IPOs)...")
    try:
        mapping = _scrape_sector_map_from_merolagani()
        if mapping:
            with open(SECTOR_CACHE, "w") as f:
                json.dump({"fetched_at": dt.datetime.now().isoformat(), "map": mapping}, f)
            log(f"  Sector map updated: {len(mapping)} symbols.")
            return mapping
    except Exception as e:
        log_error(f"  Live sector fetch failed ({e}).")

    # Fall back, in order: stale cache -> bundled seed file (343 symbols as of Aug 2026)
    if os.path.exists(SECTOR_CACHE):
        log_error("  Using stale cached sector map.")
        with open(SECTOR_CACHE) as f:
            return json.load(f)["map"]
    if os.path.exists(SECTOR_SEED):
        log_error("  Using bundled seed sector map (won't include IPOs listed after Aug 2026).")
        with open(SECTOR_SEED) as f:
            return json.load(f)

    log_error("  No sector data available -- stocks will show as 'Uncategorized'.")
    return {}


# ---------------------------------------------------------------------------
# STEP 4: load history + compute indicators
# ---------------------------------------------------------------------------
def load_history() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(DATA_DIR, "nepse_*.csv")))
    if not files:
        log_error("No data files found. Run without --skip-download first.")
        sys.exit(1)

    frames = []
    for f in files:
        try:
            # on_bad_lines='skip': a small number of NEPSE rows (e.g. debentures
            # with an unescaped comma in the name) break strict CSV parsing --
            # skip just those rows rather than losing the whole day's file.
            df = pd.read_csv(f, engine="python", on_bad_lines="skip")
            frames.append(df)
        except Exception as e:
            log_error(f"  Skipping unreadable file {f}: {e}")

    if not frames:
        log_error("No readable data files found.")
        sys.exit(1)

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.dropna(subset=["Symbol", "Close Price", "Total Traded Quantity"])
    all_df["Business Date"] = pd.to_datetime(all_df["Business Date"])
    all_df = all_df.sort_values(["Symbol", "Business Date"])
    return all_df


def _find_col(df: pd.DataFrame, keywords: list) -> str:
    """Case-insensitive best-effort column lookup for data sources whose exact
    header text we haven't verified (floorsheet, and any 'turnover'/'high
    price'-style column in the price CSV that isn't hardcoded elsewhere).
    Returns the first column whose name contains any of the given keyword
    phrases, checked in the order given, or None if nothing matches."""
    for kw in keywords:
        for c in df.columns:
            if kw in str(c).lower():
                return c
    return None


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def compute_indicators(all_df: pd.DataFrame) -> pd.DataFrame:
    results = []

    # Detected once, up front, since these are the same across every symbol's
    # slice -- used by --leaderboard's Rule 2 (EMA20 + turnover-surge check).
    # Turnover isn't hardcoded anywhere else in this script, so if NEPSE's CSV
    # doesn't have an explicit turnover/traded-value column we fall back to
    # Close Price * Total Traded Quantity as an approximation.
    turnover_col = _find_col(all_df, ["turnover", "traded value"])
    high_col = _find_col(all_df, ["high price"])
    all_df = all_df.copy()
    if turnover_col:
        all_df["_Turnover"] = pd.to_numeric(
            all_df[turnover_col].astype(str).str.replace(",", ""), errors="coerce")
    else:
        all_df["_Turnover"] = all_df["Close Price"] * all_df["Total Traded Quantity"]

    for symbol, g in all_df.groupby("Symbol"):
        g = g.sort_values("Business Date").reset_index(drop=True)
        close = g["Close Price"]

        g["RSI14"] = rsi(close, 14)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        g["MACD"] = ema12 - ema26
        g["MACD_signal"] = g["MACD"].ewm(span=9, adjust=False).mean()
        g["MA20"] = close.rolling(20, min_periods=5).mean()
        g["MA50"] = close.rolling(50, min_periods=10).mean()
        g["EMA20"] = close.ewm(span=20, adjust=False).mean()  # used by --leaderboard Rule 2, Condition A
        g["VolAvg20"] = g["Total Traded Quantity"].rolling(20, min_periods=5).mean()
        g["VolRatio"] = g["Total Traded Quantity"] / g["VolAvg20"]
        g["PctChange"] = close.pct_change() * 100
        g["UpDayStreak10"] = (g["PctChange"] > 0).rolling(10, min_periods=3).sum()
        g["DaysOfHistory"] = range(1, len(g) + 1)
        g["Turnover"] = g["_Turnover"]
        g["TurnoverAvg10"] = g["Turnover"].rolling(10, min_periods=3).mean()  # --leaderboard Rule 2, Condition B
        g["HighPrice"] = g[high_col] if high_col else pd.NA  # --leaderboard Rule 2 close-to-high check

        results.append(g.iloc[[-1]])  # latest row per symbol, with indicators attached

    latest = pd.concat(results, ignore_index=True)
    latest["NearCircuit"] = latest["PctChange"].abs() >= 9

    # -----------------------------------------------------------------
    # LAYER 1: TECHNICAL (always available -- from price/volume history)
    # -----------------------------------------------------------------
    def technical_score(r):
        has_hist = r["DaysOfHistory"] >= 15
        momentum = max(-1, min(1, (r["PctChange"] or 0) / 8))
        vol_score = max(-1, min(1, r["VolRatio"] - 1)) if pd.notna(r.get("VolRatio")) else 0
        rsi_score = max(-1, min(1, (50 - abs(r["RSI14"] - 50)) / 50)) if pd.notna(r.get("RSI14")) else 0
        macd_score = 1 if pd.notna(r.get("MACD")) and r["MACD"] > r.get("MACD_signal", 0) else -0.3

        weights = {"momentum": 0.25, "volume": 0.30, "rsi": 0.20, "macd": 0.25} if has_hist \
            else {"momentum": 0.6, "volume": 0.4, "rsi": 0, "macd": 0}
        raw = (momentum * weights["momentum"] + vol_score * weights["volume"]
               + rsi_score * weights["rsi"] + macd_score * weights["macd"])
        return round(((raw + 1) / 2) * 100)

    latest["TechScore"] = latest.apply(technical_score, axis=1)
    return latest


# ---------------------------------------------------------------------------
# STEP 5: attach sectors, then finalize sentiment + fundamental scoring
# (sentiment needs Sector attached first, which is why this is split from
# compute_indicators rather than done in one pass)
# ---------------------------------------------------------------------------
def attach_sectors(latest: pd.DataFrame, sector_map: dict) -> pd.DataFrame:
    latest["Sector"] = latest["Symbol"].map(sector_map).fillna("Uncategorized")
    return latest


def finalize_scores(latest: pd.DataFrame) -> pd.DataFrame:
    # -----------------------------------------------------------------
    # LAYER 2: SENTIMENT -- market breadth (is today a broadly bullish or
    # bearish day across NEPSE?) blended with sector relative strength (is
    # THIS stock's sector outperforming the overall market today?). NEPSE
    # research consistently flags sector rotation and retail-driven
    # sentiment as more influential short-term than in bigger markets, so
    # this gets real weight, not just a footnote.
    # -----------------------------------------------------------------
    market_avg_pct = latest["PctChange"].mean()
    market_up_ratio = (latest["PctChange"] > 0).mean()  # advance/decline breadth, 0-1
    market_breadth_score = round(market_up_ratio * 100)  # 50 = balanced, >50 = broadly bullish day

    latest["_sector_avg_pct"] = latest.groupby("Sector")["PctChange"].transform("mean")

    def sentiment_score(r):
        sector_avg = r.get("_sector_avg_pct", market_avg_pct)
        sector_avg = market_avg_pct if pd.isna(sector_avg) else sector_avg
        relative = sector_avg - market_avg_pct  # positive = sector beating the market today
        sector_strength = 50 + max(-30, min(30, relative * 10))
        return round(0.4 * market_breadth_score + 0.6 * sector_strength)

    latest["SentimentScore"] = latest.apply(sentiment_score, axis=1)
    latest.attrs["market_breadth_score"] = market_breadth_score
    latest.attrs["market_up_ratio"] = market_up_ratio
    latest.attrs["market_avg_pct"] = market_avg_pct

    # -----------------------------------------------------------------
    # LAYER 3: FUNDAMENTALS -- only populated when you've run --company for
    # that symbol (reads company_data/{symbol}/financials_*.csv and
    # dividend_*.csv). Benchmarks below come from typical NEPSE ranges
    # (P/E 15-30 normal, >40 stretched; EPS >20 considered strong for
    # banks -- scaled down a bit as a general-sector rule of thumb since
    # not every sector runs as high as banking).
    # -----------------------------------------------------------------
    def fundamental_score(symbol):
        fin_path_candidates = glob.glob(os.path.join(COMPANY_DIR, symbol, "financials_*.csv"))
        div_path_candidates = glob.glob(os.path.join(COMPANY_DIR, symbol, "dividend_*.csv"))
        if not fin_path_candidates:
            return None, {}

        try:
            fin = pd.read_csv(fin_path_candidates[0])
        except Exception:
            return None, {}
        if fin.empty:
            return None, {}

        pe_col = next((c for c in fin.columns if c.strip().upper() == "P.E" or "P/E" in c.upper()), None)
        eps_col = next((c for c in fin.columns if "EPS" in c.upper()), None)

        pe_val = pd.to_numeric(fin[pe_col].iloc[0], errors="coerce") if pe_col else None
        eps_val = pd.to_numeric(fin[eps_col].iloc[0], errors="coerce") if eps_col else None

        pe_pts, eps_pts, bonus_pts = 50, 50, 50  # neutral default per sub-factor if unavailable

        if pe_val is not None and pd.notna(pe_val):
            if pe_val <= 0:
                pe_pts = 30  # negative earnings -- caution, not automatically bad but flag it
            elif pe_val <= 20:
                pe_pts = 80
            elif pe_val <= 30:
                pe_pts = 60
            elif pe_val <= 40:
                pe_pts = 40
            else:
                pe_pts = 20  # stretched valuation

        if eps_val is not None and pd.notna(eps_val):
            if eps_val <= 0:
                eps_pts = 15
            elif eps_val < 10:
                eps_pts = 45
            elif eps_val < 20:
                eps_pts = 65
            else:
                eps_pts = 85

        bonus_years_found = 0
        bonus_years_total = 0
        if div_path_candidates:
            try:
                div = pd.read_csv(div_path_candidates[0])
                bonus_col = next((c for c in div.columns if "bonus" in c.lower()), None)
                if bonus_col is not None:
                    bonus_years_total = len(div)
                    bonus_years_found = div[bonus_col].astype(str).str.strip().apply(
                        lambda v: v not in ("", "0", "0.00", "-", "nan", "None")
                    ).sum()
                    if bonus_years_total > 0:
                        bonus_pts = round(30 + 60 * (bonus_years_found / bonus_years_total))
            except Exception:
                pass

        combined = round(0.4 * pe_pts + 0.35 * eps_pts + 0.25 * bonus_pts)
        detail = {
            "pe": pe_val, "pe_pts": pe_pts,
            "eps": eps_val, "eps_pts": eps_pts,
            "bonus_years": f"{bonus_years_found}/{bonus_years_total}" if bonus_years_total else "n/a",
            "bonus_pts": bonus_pts,
        }
        return combined, detail

    fund_results = latest["Symbol"].apply(fundamental_score)
    latest["FundScore"] = fund_results.apply(lambda x: x[0])
    latest["FundDetail"] = fund_results.apply(lambda x: x[1])

    # -----------------------------------------------------------------
    # FINAL SCORE: weighted blend of the three layers. If a stock has no
    # fundamental data (most won't, until you run --company for them), its
    # 25% weight is redistributed proportionally to Technical + Sentiment
    # rather than defaulting to a fake neutral 50 -- that would be
    # pretending we know something we don't.
    # -----------------------------------------------------------------
    BASE_WEIGHTS = {"tech": 0.50, "sentiment": 0.25, "fundamental": 0.25}

    def blend(r):
        parts = {"tech": r["TechScore"], "sentiment": r["SentimentScore"], "fundamental": r["FundScore"]}
        available = {k: v for k, v in parts.items() if pd.notna(v)}
        total_w = sum(BASE_WEIGHTS[k] for k in available)
        blended = sum(available[k] * BASE_WEIGHTS[k] for k in available) / total_w
        return round(blended)

    latest["Score"] = latest.apply(blend, axis=1)
    latest["ScoreWeights"] = latest.apply(
        lambda r: {k: BASE_WEIGHTS[k] for k in ("tech", "sentiment", "fundamental")
                   if pd.notna(r[{"tech": "TechScore", "sentiment": "SentimentScore", "fundamental": "FundScore"}[k]])},
        axis=1,
    )
    return latest


# ---------------------------------------------------------------------------
# LEADERBOARD MODE (--leaderboard) -- a separate, deterministic 100-mark
# rule-based scoring system, independent of the Score/TechScore/Sentiment
# system above. Implements exactly the four phases you specced:
#   Phase 1: Sector Rotation            -- up to 20 marks
#   Phase 2: Rule 1, Broker Concentration -- up to 40 marks
#   Phase 3: Rule 2, Late-Session Breakout Approximation -- up to 40 marks
#   Phase 4: Leaderboard + Risk Radar output
#
# Rule 1 and Rule 2 both need floorsheet_data/ (run with --floorsheet first).
# Floorsheet is only ever "today's" file (NEPSE doesn't offer historical
# floorsheet lookup -- see module docstring), so both rules are evaluated
# against whichever floorsheet file is most recently downloaded, matched
# against that same day's row in the price history.
#
# UNVERIFIED against real data: this has only been checked against the
# column-detection logic, not a live floorsheet file. Currently you only
# have floorsheet data for RLFL, so run with --leaderboard on that first --
# every other symbol will show Rule 1 Marks = 0 and Rule 2 Marks = 0 (capped
# at its Sector Marks) until you download floorsheet data covering it too.
# ---------------------------------------------------------------------------
def load_latest_floorsheet():
    """Loads the most recently downloaded floorsheet_data/floorsheet_*.csv.
    Returns None (with an explanation printed) if none exists yet or if the
    file's columns don't look like a floorsheet -- run --floorsheet first,
    or check the printed column list if this keeps failing."""
    files = sorted(glob.glob(os.path.join(FLOORSHEET_DIR, "floorsheet_*.csv")))
    if not files:
        log_error("No floorsheet data found -- Rule 1 & Rule 2 marks will all be 0. "
              "Run with --floorsheet to download today's floorsheet first.")
        return None

    path = files[-1]
    try:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception as e:
        log_error(f"Could not read floorsheet {path}: {e}")
        return None
    if df.empty:
        log_error(f"Floorsheet {os.path.basename(path)} is empty.")
        return None

    sym_col = _find_col(df, ["symbol"])
    # Prefer the actual broker NAME column over a numeric member-id column
    # when both exist (confirmed real format: buyerMemberId/sellerMemberId
    # are numeric IDs, buyerBrokerName/sellerBrokerName are human-readable
    # names -- both work for grouping, but names read far better in any
    # report output). Falls back to a plain "buyer"/"seller" match for the
    # older click-scraped format, which only ever had one such column.
    buyer_col = _find_col(df, ["buyerbrokername", "buyer broker", "buyer"])
    seller_col = _find_col(df, ["sellerbrokername", "seller broker", "seller"])
    qty_col = _find_col(df, ["quantity", "qty"])
    rate_col = _find_col(df, ["rate", "price"])
    contract_col = _find_col(df, ["contract"])
    # Optional -- older click-scraped files won't have this. When present
    # (confirmed real field: tradeTime, down to the millisecond), it lets
    # rule2_late_session_breakout and _risk_radar use a real time window
    # for "late session" instead of guessing from Contract No. order.
    time_col = _find_col(df, ["tradetime", "trade time", "time"])

    required = [("Symbol", sym_col), ("Buyer", buyer_col), ("Seller", seller_col),
                ("Quantity", qty_col), ("Rate", rate_col), ("Contract No.", contract_col)]
    missing = [name for name, col in required if col is None]
    if missing:
        log_error(f"Floorsheet {os.path.basename(path)} is missing expected column(s): {', '.join(missing)}.\n"
              f"  Actual columns found: {list(df.columns)}\n"
              f"  Fix: adjust the keyword lists in _find_col() calls inside load_latest_floorsheet() "
              f"to match these real header names.")
        return None

    out = pd.DataFrame({
        "Symbol": df[sym_col].astype(str).str.strip().str.upper(),
        "Buyer": df[buyer_col].astype(str).str.strip(),
        "Seller": df[seller_col].astype(str).str.strip(),
        "Quantity": pd.to_numeric(df[qty_col].astype(str).str.replace(",", ""), errors="coerce"),
        "Rate": pd.to_numeric(df[rate_col].astype(str).str.replace(",", ""), errors="coerce"),
        "ContractNo": pd.to_numeric(df[contract_col].astype(str).str.replace(",", ""), errors="coerce"),
        "TradeTime": pd.to_datetime(df[time_col], errors="coerce") if time_col else pd.NaT,
    })
    out = out.dropna(subset=["Symbol", "Quantity", "Rate", "ContractNo"])
    out.attrs["source_file"] = os.path.basename(path)
    if out.empty:
        log_error(f"Floorsheet {os.path.basename(path)} had columns but no usable rows after parsing.")
        return None
    log(f"Loaded floorsheet: {os.path.basename(path)} ({len(out)} rows, "
          f"{out['Symbol'].nunique()} symbol(s): {sorted(out['Symbol'].unique())[:10]}"
          f"{'...' if out['Symbol'].nunique() > 10 else ''})")
    return out


def sector_rotation_marks(all_df: pd.DataFrame, sector_map: dict):
    """PHASE 1 (20 marks): rank sectors by combined 3-session turnover +
    price momentum. #1 sector's stocks get +20, #2 gets +10, everyone else 0.
    Returns (marks_by_sector: dict, top2: list of stat dicts) for reporting."""
    df = all_df.copy()
    df["Sector"] = df["Symbol"].map(sector_map).fillna("Uncategorized")
    df["PctChange"] = df.groupby("Symbol")["Close Price"].pct_change() * 100

    turnover_col = _find_col(df, ["turnover", "traded value"])
    if turnover_col:
        df["Turnover"] = pd.to_numeric(df[turnover_col].astype(str).str.replace(",", ""), errors="coerce")
    else:
        df["Turnover"] = df["Close Price"] * df["Total Traded Quantity"]

    dates = sorted(df["Business Date"].dropna().unique())
    if len(dates) < 2:
        return {}, []
    last3 = dates[-3:]
    recent = df[df["Business Date"].isin(last3)]

    sector_stats = recent.groupby("Sector").agg(
        Turnover=("Turnover", "sum"),
        AvgPctChange=("PctChange", "mean"),
    ).reset_index()
    if sector_stats.empty:
        return {}, []

    sector_stats["TurnoverRank"] = sector_stats["Turnover"].rank(ascending=False)
    sector_stats["MomentumRank"] = sector_stats["AvgPctChange"].rank(ascending=False)
    sector_stats["CombinedRank"] = sector_stats["TurnoverRank"] + sector_stats["MomentumRank"]
    sector_stats = sector_stats.sort_values("CombinedRank")

    top2 = sector_stats.head(2).to_dict("records")
    marks = {}
    if len(top2) >= 1:
        marks[top2[0]["Sector"]] = 20
    if len(top2) >= 2:
        marks[top2[1]["Sector"]] = 10
    return marks, top2


def rule1_broker_concentration(symbol: str, floorsheet_df):
    """PHASE 2 (40 marks): does a single Broker ID dominate today's delivery
    for this stock, and are they net buying it? A broker's "% of daily
    volume" is (their buy qty + sell qty) / (2 x total traded qty), since
    total buy qty == total sell qty == total traded qty across all brokers."""
    if floorsheet_df is None:
        return 0, "No floorsheet data available."

    rows = floorsheet_df[floorsheet_df["Symbol"] == symbol.upper()]
    if rows.empty:
        source = floorsheet_df.attrs.get("source_file", "today's floorsheet")
        return 0, f"No floorsheet rows for {symbol} in {source}."

    total_qty = rows["Quantity"].sum()
    if total_qty <= 0:
        return 0, "Zero traded quantity in floorsheet rows."

    buy_by_broker = rows.groupby("Buyer")["Quantity"].sum()
    sell_by_broker = rows.groupby("Seller")["Quantity"].sum()
    brokers = set(buy_by_broker.index) | set(sell_by_broker.index)

    best_marks, best_detail = 0, None
    for b in brokers:
        buy = buy_by_broker.get(b, 0)
        sell = sell_by_broker.get(b, 0)
        if buy <= 0:
            continue
        pct_of_volume = (buy + sell) / (2 * total_qty)
        ratio = (buy / sell) if sell > 0 else float("inf")

        if pct_of_volume >= 0.30 and ratio >= 3:
            m = 40
        elif pct_of_volume >= 0.20 and ratio >= 2:
            m = 20
        else:
            m = 0

        if m > best_marks:
            best_marks, best_detail = m, {"broker": b, "pct": pct_of_volume, "ratio": ratio, "buy": buy, "sell": sell}

    if best_marks == 0:
        return 0, "No single broker met the concentration + buy/sell ratio thresholds (fragmented buying)."

    d = best_detail
    ratio_str = "inf" if d["ratio"] == float("inf") else f"{d['ratio']:.1f}:1"
    return best_marks, (f"Broker {d['broker']} handled {d['pct']*100:.1f}% of today's volume "
                         f"(buy {d['buy']:,.0f} / sell {d['sell']:,.0f}, ratio {ratio_str}).")


def _late_session_split(rows: pd.DataFrame, late_minutes: int = 15, min_late_rows: int = 3):
    """Splits a symbol's floorsheet rows into (early, late, method_note).
    Prefers a real time window -- the final `late_minutes` minutes before
    this symbol's own last trade of the day -- over the old proxy of
    "final 10% of trades by Contract No. order", now that real trade
    timestamps exist in the data. Falls back to the Contract-No.-based
    method automatically if TradeTime is missing (older click-scraped
    files) or if the time window ends up with too few trades to be
    meaningful (e.g. a thinly-traded symbol whose last 15 minutes only
    saw one or two prints, even though its full-day count was fine) --
    in that case the count-based slice is likely to be more stable.
    Reference point is THIS symbol's own last trade, not the market's
    overall close, so an illiquid stock that stops trading early still
    gets a sensible "late session" window for itself."""
    rows = rows.sort_values("ContractNo")
    has_time = "TradeTime" in rows.columns and rows["TradeTime"].notna().sum() >= max(min_late_rows, 2)

    if has_time:
        timed = rows.dropna(subset=["TradeTime"])
        cutoff = timed["TradeTime"].max() - pd.Timedelta(minutes=late_minutes)
        late = timed[timed["TradeTime"] >= cutoff]
        early = timed[timed["TradeTime"] < cutoff]
        if len(late) >= min_late_rows and len(early) >= 1:
            return early, late, f"last {late_minutes} min by trade time"

    # Fallback: original proxy, since no real timestamp was usable.
    cut = max(1, int(len(rows) * 0.9))
    return rows.iloc[:cut], rows.iloc[cut:], "final 10% of trades by Contract No. order (no usable trade time)"


def rule2_late_session_breakout(symbol: str, price_row, floorsheet_df):
    """PHASE 3 (40 marks): Condition A (close > 20-day EMA) and Condition B
    (turnover >= 2x its 10-day average) must both hold before the
    floorsheet's late-session trades are even checked. If they hold, the
    late-session window (see _late_session_split) is compared to the rest
    of the day's trades."""
    close = price_row.get("Close Price")
    ema20 = price_row.get("EMA20")
    turnover = price_row.get("Turnover")
    turnover_avg10 = price_row.get("TurnoverAvg10")
    high = price_row.get("HighPrice")

    cond_a = pd.notna(close) and pd.notna(ema20) and close > ema20
    cond_b = pd.notna(turnover) and pd.notna(turnover_avg10) and turnover_avg10 > 0 and turnover >= 2 * turnover_avg10
    if not (cond_a and cond_b):
        missing = []
        if not cond_a:
            missing.append("close not above 20-day EMA")
        if not cond_b:
            missing.append("turnover not >=2x its 10-day average")
        return 0, f"Conditions not met ({'; '.join(missing)})."

    if floorsheet_df is None:
        return 0, "Conditions A & B met, but no floorsheet data to confirm the trade sequence."

    rows = floorsheet_df[floorsheet_df["Symbol"] == symbol.upper()]
    if len(rows) < 10:
        return 0, "Conditions A & B met, but too few floorsheet trades today to sample the late session."

    first90, last10, method = _late_session_split(rows)
    avg_first, avg_last = first90["Rate"].mean(), last10["Rate"].mean()

    close_to_high_pct = None
    if pd.notna(high) and high and high > 0:
        close_to_high_pct = (high - close) / high * 100

    high_note = f"{close_to_high_pct:.2f}% off day's high" if close_to_high_pct is not None else "day's high unavailable"
    detail = (f"Late session ({method}): {len(last10)}/{len(rows)} trades averaged {avg_last:.2f} vs "
              f"{avg_first:.2f} for the rest of the day ({high_note}).")

    if avg_last > avg_first and close_to_high_pct is not None and close_to_high_pct <= 0.5:
        return 40, detail
    elif close_to_high_pct is not None and close_to_high_pct <= 1.5:
        return 20, detail
    else:
        return 0, detail


def _risk_radar(symbol: str, price_row, floorsheet_df) -> dict:
    """Next-day price triggers for a ranked pick: opening range (NEPSE's
    circuit band off today's close), a buy trigger zone derived from where
    the late-session trades actually happened (see _late_session_split;
    falls back to close-to-ceiling if floorsheet data isn't available),
    and a 5% hard stop-loss off the midpoint of that buy zone."""
    close = price_row["Close Price"]
    ceiling = close * (1 + CIRCUIT_PCT)
    floor_ = close * (1 - CIRCUIT_PCT)

    buy_min, buy_max = close, ceiling
    rows = None
    if floorsheet_df is not None:
        rows = floorsheet_df[floorsheet_df["Symbol"] == symbol.upper()]
    if rows is not None and len(rows) >= 10:
        _, last10, _ = _late_session_split(rows)
        buy_min = max(close, last10["Rate"].min())
        buy_max = min(ceiling, last10["Rate"].max() * 1.01)
        if buy_max < buy_min:
            buy_min, buy_max = close, ceiling  # last10 pricing was below close; fall back

    entry_est = (buy_min + buy_max) / 2
    stop_loss = entry_est * 0.95

    return {"opening_range": (floor_, ceiling), "buy_trigger": (buy_min, buy_max),
            "entry_est": entry_est, "stop_loss": stop_loss}


def build_leaderboard(latest: pd.DataFrame, all_df: pd.DataFrame, sector_map: dict, floorsheet_df):
    """PHASE 4: combine Phases 1-3 into the ranked leaderboard, filter to
    Total >= 60, write leaderboard.md, and print it. Returns
    (board, top_sectors) -- board is the qualifying records (for tests /
    reuse / the web report), top_sectors is the Phase 1 sector-rotation
    ranking, also needed by the web report."""
    sector_marks, top_sectors = sector_rotation_marks(all_df, sector_map)

    records = []
    for _, r in latest.iterrows():
        symbol = r["Symbol"]
        sec_marks = sector_marks.get(r["Sector"], 0)
        r1_marks, r1_detail = rule1_broker_concentration(symbol, floorsheet_df)
        r2_marks, r2_detail = rule2_late_session_breakout(symbol, r, floorsheet_df)
        total = sec_marks + r1_marks + r2_marks
        records.append({
            "Symbol": symbol, "Sector": r["Sector"], "SectorMarks": sec_marks,
            "Rule1Marks": r1_marks, "Rule2Marks": r2_marks, "Total": total,
            "Rule1Detail": r1_detail, "Rule2Detail": r2_detail, "row": r,
        })

    board = sorted([rec for rec in records if rec["Total"] >= 60], key=lambda x: x["Total"], reverse=True)

    lines = ["# NEPSE Leaderboard -- Phase 1-4 Rule-Based Scoring\n",
             f"_Generated {dt.date.today().isoformat()}. Floorsheet: "
             f"{floorsheet_df.attrs.get('source_file') if floorsheet_df is not None else 'none loaded'}._\n",
             "## 🔥 TOP NEPSE SECTOR ROTATION\n"]

    if top_sectors:
        for i, s in enumerate(top_sectors, 1):
            lines.append(f"{i}. **{s['Sector']}** -- 3-session turnover {s['Turnover']:,.0f}, "
                         f"avg daily move {s['AvgPctChange']:+.2f}%")
    else:
        lines.append("Not enough sector-level history yet to rank rotation (need at least a couple of trading days).")
    lines.append("")

    lines.append("## 🏆 TOP PICKS FOR TOMORROW (Ranked by Total Marks)\n")
    if not board:
        if floorsheet_df is None:
            lines.append("No stock scored 60+ marks. No floorsheet data is loaded, so Rule 1 and Rule 2 "
                          "can't score anyone above 0 yet -- run with `--floorsheet` first, then "
                          "`--leaderboard` again.")
        else:
            covered = sorted(floorsheet_df["Symbol"].unique())
            lines.append(f"No stock scored 60+ marks today. Floorsheet data currently only covers "
                         f"{', '.join(covered)}, so Rule 1/2 can only score above 0 for {'that symbol' if len(covered) == 1 else 'those symbols'} "
                         f"-- everyone else is capped at their Sector Marks (max 20/100). Download more "
                         f"floorsheet days to widen coverage.")
    else:
        lines.append("| Ticker | Sector | Sector Marks (20) | Rule 1 Marks (40) | Rule 2 Marks (40) "
                     "| Total Marks (100) | Core Justification |")
        lines.append("|---|---|---|---|---|---|---|")
        for rec in board:
            justification = f"{rec['Rule1Detail']} {rec['Rule2Detail']}".strip()
            lines.append(f"| {rec['Symbol']} | {rec['Sector']} | {rec['SectorMarks']} | {rec['Rule1Marks']} "
                         f"| {rec['Rule2Marks']} | {rec['Total']} | {justification} |")
    lines.append("")

    lines.append("## 🚨 DISCIPLINED RISK RADAR\n")
    if not board:
        lines.append("No stocks qualified, so no triggers to show.")
    else:
        for rec in board[:3]:
            radar = _risk_radar(rec["Symbol"], rec["row"], floorsheet_df)
            lines.append(f"**{rec['Symbol']}**")
            lines.append(f"- Opening day range limit ({CIRCUIT_PCT*100:.0f}% circuit off today's close): "
                         f"{radar['opening_range'][0]:.2f} - {radar['opening_range'][1]:.2f}")
            lines.append(f"- Buy trigger price range: {radar['buy_trigger'][0]:.2f} - {radar['buy_trigger'][1]:.2f}")
            lines.append(f"- 5% hard stop-loss (off est. entry {radar['entry_est']:.2f}): {radar['stop_loss']:.2f}")
            lines.append("")

    report_md = "\n".join(lines)
    md_path = os.path.join(BASE_DIR, "leaderboard.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    log("\n" + report_md)
    log(f"\nLeaderboard written to: {md_path}")
    return board, top_sectors


def _leaderboard_html(board: list, top_sectors: list, floorsheet_df) -> str:
    """Renders the Phase 1-4 leaderboard (same data as leaderboard.md /
    the console printout) as HTML for the Leaderboard tab of report.html."""
    parts = ['<div class="lb-section"><h3>🔥 Top Sector Rotation</h3>']
    if top_sectors:
        items = "".join(
            f"<li><b>{s['Sector']}</b> &mdash; 3-session turnover {s['Turnover']:,.0f}, "
            f"avg daily move {s['AvgPctChange']:+.2f}%</li>"
            for s in top_sectors
        )
        parts.append(f"<ol class='lb-list'>{items}</ol>")
    else:
        parts.append("<div class='muted'>Not enough sector-level history yet to rank rotation "
                      "(need at least a couple of trading days).</div>")
    parts.append("</div>")

    parts.append('<div class="lb-section"><h3>🏆 Top Picks (Ranked by Total Marks)</h3>')
    if not board:
        if floorsheet_df is None:
            parts.append("<div class='muted'>No stock scored 60+ marks. No floorsheet data is loaded, "
                          "so Rule 1 and Rule 2 can't score anyone above 0 yet -- floorsheet data downloads "
                          "automatically each run, so just run again once you have a couple of days of it.</div>")
        else:
            covered = sorted(floorsheet_df["Symbol"].unique())
            plural = "that symbol" if len(covered) == 1 else "those symbols"
            parts.append(f"<div class='muted'>No stock scored 60+ marks today. Floorsheet data currently "
                         f"only covers {', '.join(covered)}, so Rule 1/2 can only score above 0 for {plural} "
                         f"-- everyone else is capped at their Sector Marks (max 20/100). Download more "
                         f"floorsheet days to widen coverage.</div>")
    else:
        rows = ""
        for rec in board:
            justification = f"{rec['Rule1Detail']} {rec['Rule2Detail']}".strip()
            rows += f"""
              <tr>
                <td><b>{rec['Symbol']}</b></td>
                <td>{rec['Sector']}</td>
                <td>{rec['SectorMarks']}</td>
                <td>{rec['Rule1Marks']}</td>
                <td>{rec['Rule2Marks']}</td>
                <td><b>{rec['Total']}</b></td>
                <td class="signal">{justification}</td>
              </tr>"""
        parts.append(f"""
            <table class="lb-table">
              <thead><tr><th>Ticker</th><th>Sector</th><th>Sector (20)</th><th>Rule 1 (40)</th>
                <th>Rule 2 (40)</th><th>Total (100)</th><th>Core Justification</th></tr></thead>
              <tbody>{rows}</tbody>
            </table>""")
    parts.append("</div>")

    parts.append('<div class="lb-section"><h3>🚨 Disciplined Risk Radar</h3>')
    if not board:
        parts.append("<div class='muted'>No stocks qualified, so no triggers to show.</div>")
    else:
        for rec in board[:3]:
            radar = _risk_radar(rec["Symbol"], rec["row"], floorsheet_df)
            parts.append(f"""
              <div class="lb-radar">
                <b>{rec['Symbol']}</b>
                <ul>
                  <li>Opening day range limit ({CIRCUIT_PCT*100:.0f}% circuit off today's close):
                      {radar['opening_range'][0]:.2f} &ndash; {radar['opening_range'][1]:.2f}</li>
                  <li>Buy trigger price range: {radar['buy_trigger'][0]:.2f} &ndash; {radar['buy_trigger'][1]:.2f}</li>
                  <li>5% hard stop-loss (off est. entry {radar['entry_est']:.2f}): {radar['stop_loss']:.2f}</li>
                </ul>
              </div>""")
    parts.append("</div>")

    return "".join(parts)


def _tomorrow_tab_html(board: list, floorsheet_df) -> str:
    """Dedicated 'Tomorrow' tab: the same Disciplined Risk Radar data shown
    in the Leaderboard tab's bottom section, but as its own top-level tab
    and covering every qualifying pick (not just the top 3), since it has
    the whole tab to itself now instead of sharing space."""
    if not board:
        return ("<div class='muted'>No stock scored 60+ marks today, so there are no next-day "
                "triggers to show. Check the Leaderboard tab for why.</div>")

    parts = ['<div class="lb-section"><h3>🚨 Disciplined Risk Radar -- Next Session Triggers</h3>']
    parts.append("<div class='muted' style='margin-bottom:10px;'>Derived from today's close and, where "
                 "available, the late-session portion of today's floorsheet. Not a guarantee -- a hard "
                 "circuit-band cap and a 5% stop-loss are built in on purpose.</div>")
    for rec in board:
        radar = _risk_radar(rec["Symbol"], rec["row"], floorsheet_df)
        parts.append(f"""
          <div class="lb-radar">
            <b>{rec['Symbol']}</b> <span class="muted">({rec['Sector']}, Total {rec['Total']}/100)</span>
            <ul>
              <li>Opening day range limit ({CIRCUIT_PCT*100:.0f}% circuit off today's close):
                  {radar['opening_range'][0]:.2f} &ndash; {radar['opening_range'][1]:.2f}</li>
              <li>Buy trigger price range: {radar['buy_trigger'][0]:.2f} &ndash; {radar['buy_trigger'][1]:.2f}</li>
              <li>5% hard stop-loss (off est. entry {radar['entry_est']:.2f}): {radar['stop_loss']:.2f}</li>
            </ul>
          </div>""")
    parts.append("</div>")
    return "".join(parts)


def _daily_review_md_to_html(md_text: str) -> str:
    """Tiny, purpose-built converter for the specific markdown subset
    adaptive_engine.write_daily_review() actually generates (#/## headers,
    **bold**, - bullets, plain paragraphs) -- avoids adding a general
    markdown-parsing dependency just to redisplay one file we already
    generated ourselves."""
    import re
    html_lines = []
    in_list = False
    for raw_line in md_text.split("\n"):
        line = raw_line.rstrip()
        if line.startswith("## "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h4>{line[3:]}</h4>")
        elif line.startswith("# "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h3>{line[2:]}</h3>")
        elif line.startswith("- "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{line[2:]}</li>")
        elif line.strip() == "":
            if in_list:
                html_lines.append("</ul>")
                in_list = False
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<p>{line}</p>")
    if in_list:
        html_lines.append("</ul>")
    out = "\n".join(html_lines)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<i>\1</i>", out)
    out = re.sub(r"`(.+?)`", r"<code>\1</code>", out)
    return out


def _adaptive_tab_html() -> str:
    """Renders the Adaptive Engine tab: the latest daily review (today's
    paper-trading recommendations, meant to be acted on -- on paper --
    tomorrow) plus a compact strategy comparison table, if
    adaptive_engine.py has been run at least once. Reads its output files
    directly from disk rather than importing adaptive_engine, since
    adaptive_engine already imports THIS module -- importing it back would
    be circular."""
    adaptive_dir = os.path.join(BASE_DIR, "adaptive")
    if not os.path.isdir(adaptive_dir):
        return ("<div class='muted'>No adaptive engine data yet -- run "
                "<code>python nepse_screener.py --adaptive</code> at least once to populate this tab "
                "with tomorrow's paper-trading recommendations and the ongoing strategy comparison.</div>")

    parts = []
    review_files = sorted(glob.glob(os.path.join(adaptive_dir, "daily_review_*.md")))
    if review_files:
        latest_review = review_files[-1]
        review_date = os.path.basename(latest_review).replace("daily_review_", "").replace(".md", "")
        try:
            with open(latest_review, encoding="utf-8") as f:
                review_md = f.read()
            parts.append(f'<div class="lb-section"><h3>🔮 Latest Adaptive Recommendations ({review_date})</h3>')
            parts.append(_daily_review_md_to_html(review_md))
            parts.append('</div>')
        except Exception as e:
            parts.append(f"<div class='muted'>Could not read {os.path.basename(latest_review)}: {e}</div>")
    else:
        parts.append("<div class='muted'>Adaptive engine directory exists, but no daily review file was "
                     "found yet -- run with <code>--adaptive</code> to generate one.</div>")

    comparison_path = os.path.join(adaptive_dir, "strategy_comparison.csv")
    if os.path.exists(comparison_path):
        try:
            comp_df = pd.read_csv(comparison_path)
            parts.append('<div class="lb-section"><h3>📊 Strategy Comparison (Baseline vs Shadow)</h3>')
            parts.append(comp_df.to_html(index=False, classes="lb-table", border=0))
            parts.append('</div>')
        except Exception:
            pass

    parts.append("<div class='muted' style='margin-top:10px;'>Paper-trading research only -- no real orders "
                 "are placed. Figures reflect however many trades have closed so far and will be noisy until "
                 "there's a real sample size.</div>")
    return "".join(parts)


def _plain_signal(r) -> str:
    """Translate the raw indicator values into a short, human-readable tag --
    this is the whole point of the legend: you shouldn't need to know what
    RSI or MACD mean to understand why a stock is ranked where it is."""
    tags = []
    rsi = r.get("RSI14")
    if pd.notna(rsi):
        if rsi < 30:
            tags.append("oversold")
        elif rsi > 70:
            tags.append("overbought")
    if pd.notna(r.get("MACD")) and pd.notna(r.get("MACD_signal")):
        tags.append("bullish MACD" if r["MACD"] > r["MACD_signal"] else "bearish MACD")
    vr = r.get("VolRatio")
    if pd.notna(vr):
        if vr >= 2:
            tags.append("volume spike")
        elif vr >= 1.3:
            tags.append("above-avg volume")
        elif vr < 0.6:
            tags.append("quiet volume")
    if r.get("NearCircuit"):
        tags.append("⚠️ near circuit")
    if pd.notna(r.get("FundScore")):
        tags.append("has fundamentals")
    return ", ".join(tags) if tags else "limited signal (still building history)"


def _company_deepdive_html(symbol: str) -> str:
    """If you've run --company for this symbol, embed a compact summary of
    what was found (key stats, latest financials, bonus history) right in
    the main dashboard instead of making you open a separate file."""
    company_folder = os.path.join(COMPANY_DIR, symbol)
    if not os.path.isdir(company_folder):
        return ""

    parts = [f'<div class="deepdive"><h3>{symbol} -- company data on file</h3>']

    stats_path = os.path.join(company_folder, "key_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in stats.items())
        parts.append(f'<table class="mini">{rows}</table>')

    div_files = glob.glob(os.path.join(company_folder, "dividend_*.csv"))
    if div_files:
        try:
            div = pd.read_csv(div_files[0])
            parts.append('<div class="subhead">Dividend / bonus history</div>')
            parts.append(div.to_html(index=False, classes="mini", border=0))
        except Exception:
            pass

    ca_files = glob.glob(os.path.join(company_folder, "corporate_actions_*.csv"))
    if ca_files:
        try:
            ca = pd.read_csv(ca_files[0])
            if not ca.empty:
                parts.append('<div class="subhead">Corporate actions</div>')
                parts.append(ca.to_html(index=False, classes="mini", border=0))
        except Exception:
            pass

    parts.append('<div class="muted" style="margin-top:6px;">Historical record only -- not a prediction of future bonus, dividend, or price.</div>')
    parts.append('</div>')
    return "".join(parts)


def build_report(latest: pd.DataFrame, leaderboard=None):
    """leaderboard, if provided, is (board, top_sectors, floorsheet_df) from
    build_leaderboard() -- it's rendered as a second 'Leaderboard' tab on the
    page. If None (i.e. the script was run without --leaderboard), that tab
    still appears but just explains how to populate it, so the tab is
    always in the same place run to run."""
    as_of = latest["Business Date"].max()
    sectors = sorted(latest["Sector"].unique())

    market_up_ratio = latest.attrs.get("market_up_ratio", 0.5)
    market_avg_pct = latest.attrs.get("market_avg_pct", 0)
    mood = "broadly bullish" if market_up_ratio > 0.55 else ("broadly bearish" if market_up_ratio < 0.45 else "mixed / balanced")

    companies_with_data = sorted({s for s in latest["Symbol"] if os.path.isdir(os.path.join(COMPANY_DIR, s))})

    rows_html = []
    for sector in sectors:
        sub = latest[latest["Sector"] == sector].sort_values("Score", ascending=False)
        table_rows = ""
        for _, r in sub.iterrows():
            hist_note = "" if r["DaysOfHistory"] >= 15 else f" <span class='muted'>({int(r['DaysOfHistory'])}d hist)</span>"
            pct = r["PctChange"] if pd.notna(r["PctChange"]) else 0
            rsi_val = f"{r['RSI14']:.0f}" if pd.notna(r.get("RSI14")) else "—"
            vol_ratio_val = f"{r['VolRatio']:.1f}x" if pd.notna(r.get("VolRatio")) else "—"
            company_name = r.get("Security Name", "")
            has_company_data = r["Symbol"] in companies_with_data

            weights = r.get("ScoreWeights", {}) or {}
            breakdown_lines = [f"Technical: {r['TechScore']:.0f} &times; {weights.get('tech', 0)*100:.0f}%"]
            breakdown_lines.append(f"Sentiment: {r['SentimentScore']:.0f} &times; {weights.get('sentiment', 0)*100:.0f}%")
            if pd.notna(r.get("FundScore")):
                fd = r.get("FundDetail", {}) or {}
                pe_str = f"{fd.get('pe'):.1f}" if fd.get("pe") is not None and pd.notna(fd.get("pe")) else "n/a"
                eps_str = f"{fd.get('eps'):.1f}" if fd.get("eps") is not None and pd.notna(fd.get("eps")) else "n/a"
                breakdown_lines.append(
                    f"Fundamentals: {r['FundScore']:.0f} &times; {weights.get('fundamental', 0)*100:.0f}% "
                    f"<span class='muted'>(P/E {pe_str}, EPS {eps_str}, bonus {fd.get('bonus_years','n/a')} yrs)</span>"
                )
            else:
                breakdown_lines.append('<span class="muted">Fundamentals: not fetched -- run --company ' + r["Symbol"] + '</span>')
            breakdown_html = "<br>".join(breakdown_lines)

            # Symbol itself is now clickable and expands its profile inline,
            # right here, rather than linking down to a section at the
            # bottom of the page.
            if has_company_data:
                symbol_cell = f'''<details class="symbol-detail">
                    <summary><b>{r['Symbol']}</b></summary>
                    <span class="company-name">{company_name}</span>{hist_note}
                    <div class="inline-profile">{_company_deepdive_html(r['Symbol'])}</div>
                  </details>'''
            else:
                symbol_cell = f'<b>{r["Symbol"]}</b><br><span class="company-name">{company_name}</span>{hist_note}'

            table_rows += f"""
              <tr>
                <td>{symbol_cell}</td>
                <td>{r['Close Price']:.2f}</td>
                <td class="{'up' if pct>=0 else 'down'}">{pct:+.2f}%</td>
                <td>{rsi_val}</td>
                <td>{vol_ratio_val}</td>
                <td class="signal">{_plain_signal(r)}</td>
                <td>
                  <details class="score-detail">
                    <summary><b>{r['Score']}</b></summary>
                    <div class="breakdown">{breakdown_html}</div>
                  </details>
                </td>
              </tr>"""
        # Sectors are now collapsible -- closed by default, click the
        # sector name to expand and see its stocks.
        rows_html.append(f"""
          <details class="sector">
            <summary><h2>{sector} <span class="muted">({len(sub)} stocks)</span></h2></summary>
            <table>
              <thead><tr><th>Symbol / Company</th><th>Close</th><th>Chg</th><th>RSI</th><th>Vol</th><th>Signal</th><th>Score (click)</th></tr></thead>
              <tbody>{table_rows}</tbody>
            </table>
          </details>""")

    legend = f"""
      <details class="legend">
        <summary>How is Score calculated? (click to expand)</summary>
        <div class="legend-body">
          <div class="legend-item"><b>Score is a weighted blend of three layers</b>, each 0&ndash;100, click any stock's score to see its own breakdown:</div>
          <div class="legend-item">1. <b>Technical (50% weight)</b> &mdash; momentum, volume vs. its own average, RSI, MACD. Reflects short-term price/volume behavior.</div>
          <div class="legend-item">2. <b>Sentiment (25% weight)</b> &mdash; today's market breadth (% of NEPSE advancing) blended with whether this stock's sector is outperforming the overall market today. NEPSE is retail-driven, so sector rotation and day-to-day mood genuinely move prices here.</div>
          <div class="legend-item">3. <b>Fundamentals (25% weight, only when available)</b> &mdash; P/E (15&ndash;30 is typical on NEPSE, above 40 is stretched), EPS, and bonus-share consistency over recent years. Only populated once you run <code>--company SYMBOL</code> for that stock -- until then its weight shifts proportionally onto Technical + Sentiment rather than being guessed.</div>
          <div class="legend-item"><b>RSI (0&ndash;100)</b> &mdash; <span class="good">below 30</span> = oversold (may bounce), <span class="bad">above 70</span> = overbought (may pull back).</div>
          <div class="legend-item"><b>Vol (volume ratio)</b> &mdash; <span class="good">above 1.3x</span> = notably more interest than usual; <span class="bad">below 0.6x</span> = quiet.</div>
          <div class="legend-item">None of this is a prediction or financial advice -- it's a way to prioritize your own research, not a buy signal.</div>
        </div>
      </details>"""

    # ---- Leaderboard tab content -----------------------------------------
    # `leaderboard` is (board, top_sectors, floorsheet_df) when the script
    # was run with --leaderboard; otherwise this tab still exists (so its
    # position never moves run to run) but just explains how to fill it in.
    if leaderboard is not None:
        board, top_sectors, floorsheet_df = leaderboard
        leaderboard_body = _leaderboard_html(board, top_sectors, floorsheet_df)
        lb_badge = f" <span class='tab-badge'>{len(board)}</span>" if board else ""
        tomorrow_body = _tomorrow_tab_html(board, floorsheet_df)
        tomorrow_badge = f" <span class='tab-badge'>{len(board)}</span>" if board else ""
    else:
        leaderboard_body = (
            "<div class='lb-section'><div class='muted'>This tab is empty because the script was run "
            "without <code>--leaderboard</code>. Run <code>python nepse_screener.py --leaderboard</code> "
            "to populate it with the Phase 1-4 rule-based scoring (sector rotation + broker delivery "
            "concentration + late-session breakout).</div></div>"
        )
        lb_badge = ""
        tomorrow_body = (
            "<div class='muted'>This tab is empty because the script was run without "
            "<code>--leaderboard</code>. Run <code>python nepse_screener.py --leaderboard</code> to "
            "populate it with next-session price triggers for every qualifying pick.</div>"
        )
        tomorrow_badge = ""

    adaptive_body = _adaptive_tab_html()

    runlog_body = _runlog_tab_html()
    _error_count = sum(1 for level, _ in _LOG_MESSAGES if level == "error")
    runlog_badge = f" <span class='tab-badge'>{_error_count} error(s)</span>" if _error_count else ""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>NEPSE Sector Screener</title>
<style>
body{{background:#0d1117;color:#e6edf3;font-family:Segoe UI,sans-serif;padding:24px;max-width:1050px;margin:0 auto;}}
h1{{margin-bottom:2px;}} .asof{{color:#8b98a5;font-size:13px;margin-bottom:18px;}}
.sentiment-banner{{background:#132a2f;border:1px solid #1f4a4f;color:#7fd4d9;font-size:13px;padding:10px 14px;border-radius:8px;margin-bottom:14px;}}
.disclaimer{{background:#2a1f14;border:1px solid #4a3a1f;color:#e8c37a;font-size:12.5px;padding:10px 14px;border-radius:8px;margin-bottom:14px;}}
.legend{{background:#141b24;border:1px solid #26313d;border-radius:8px;padding:10px 16px;margin-bottom:22px;font-size:13px;}}
.legend summary{{cursor:pointer;color:#5fb4c9;font-weight:600;}}
.legend-body{{margin-top:10px;line-height:1.7;}}
.legend-item{{margin-bottom:8px;}}
.good{{color:#3fb950;}} .bad{{color:#f85149;}}
.muted{{color:#8b98a5;font-size:11px;}}
.company-name{{color:#8b98a5;font-size:11.5px;}}
.sector{{background:#141b24;border:1px solid #26313d;border-radius:10px;padding:14px 18px;margin-bottom:16px;}}
.sector summary{{cursor:pointer;list-style:none;}}
.sector summary::-webkit-details-marker{{display:none;}}
.sector h2{{font-size:15px;margin:0;color:#5fb4c9;display:inline;}}
.sector summary::before{{content:"▸ ";color:#5fb4c9;}}
.sector[open] summary::before{{content:"▾ ";}}
.sector table{{margin-top:12px;}}
table{{width:100%;border-collapse:collapse;font-size:13px;}}
th{{text-align:left;color:#8b98a5;font-size:11px;text-transform:uppercase;padding:5px 8px;border-bottom:1px solid #26313d;}}
td{{padding:6px 8px;border-bottom:1px solid #1e2731;vertical-align:top;}}
.up{{color:#3fb950;}} .down{{color:#f85149;}} .signal{{color:#c9d1d9;font-size:12px;}}
.score-detail summary{{cursor:pointer;list-style:none;}}
.score-detail summary::-webkit-details-marker{{display:none;}}
.breakdown{{font-size:11.5px;color:#c9d1d9;margin-top:6px;line-height:1.6;background:#0d1117;padding:8px;border-radius:6px;}}
.symbol-detail summary{{cursor:pointer;list-style:none;}}
.symbol-detail summary::-webkit-details-marker{{display:none;}}
.symbol-detail summary b{{text-decoration:underline dotted;}}
.inline-profile{{margin-top:8px;}}
.deepdive{{background:#0d1117;border:1px solid #26313d;border-radius:8px;padding:12px 14px;margin-top:4px;}}
.deepdive h3{{font-size:13px;color:#5fb4c9;margin:0 0 8px;}}
.subhead{{font-size:11px;color:#8b98a5;text-transform:uppercase;margin:10px 0 5px;}}
table.mini{{font-size:12px;}}
table.mini td, table.mini th{{padding:4px 8px;}}
.tab-input{{display:none;}}
.tab-bar{{display:flex;gap:4px;margin-bottom:20px;border-bottom:1px solid #26313d;}}
.tab-label{{cursor:pointer;padding:9px 18px;color:#8b98a5;font-size:13.5px;font-weight:600;border-bottom:2px solid transparent;margin-bottom:-1px;user-select:none;}}
.tab-label:hover{{color:#c9d1d9;}}
.tab-badge{{background:#1f4a4f;color:#7fd4d9;border-radius:10px;padding:1px 7px;font-size:11px;margin-left:4px;}}
.tab-panel{{display:none;}}
#tab-screener:checked ~ .tab-bar .tab-label-screener,
#tab-leaderboard:checked ~ .tab-bar .tab-label-leaderboard,
#tab-tomorrow:checked ~ .tab-bar .tab-label-tomorrow,
#tab-adaptive:checked ~ .tab-bar .tab-label-adaptive,
#tab-runlog:checked ~ .tab-bar .tab-label-runlog{{color:#5fb4c9;border-bottom-color:#5fb4c9;}}
#tab-screener:checked ~ .tab-panel-screener{{display:block;}}
#tab-leaderboard:checked ~ .tab-panel-leaderboard{{display:block;}}
#tab-tomorrow:checked ~ .tab-panel-tomorrow{{display:block;}}
#tab-adaptive:checked ~ .tab-panel-adaptive{{display:block;}}
#tab-runlog:checked ~ .tab-panel-runlog{{display:block;}}
.runlog-box{{background:#0d1117;border:1px solid #26313d;border-radius:8px;padding:12px 16px;
  font-family:Consolas,Menlo,monospace;font-size:12.5px;line-height:1.7;white-space:pre-wrap;
  color:#c9d1d9;max-height:70vh;overflow-y:auto;}}
.runlog-err{{color:#ff7b72;}}
.lb-section{{background:#141b24;border:1px solid #26313d;border-radius:10px;padding:14px 18px;margin-bottom:16px;}}
.lb-section h3{{font-size:14px;color:#5fb4c9;margin:0 0 10px;}}
.lb-list{{margin:0;padding-left:20px;line-height:1.8;font-size:13px;}}
table.lb-table{{font-size:13px;}}
.lb-radar{{background:#0d1117;border:1px solid #26313d;border-radius:8px;padding:10px 14px;margin-top:8px;}}
.lb-radar ul{{margin:6px 0 0;padding-left:18px;line-height:1.7;font-size:12.5px;color:#c9d1d9;}}
</style></head><body>
<h1>NEPSE Sector-Wise Screener</h1>
<div class="asof">As of {as_of.date()} &middot; {len(latest)} symbols across {len(sectors)} sectors &middot; generated {dt.date.today().isoformat()}</div>
<div class="sentiment-banner">📊 Today's market mood: <b>{mood}</b> &mdash; {market_up_ratio*100:.0f}% of stocks advancing, average move {market_avg_pct:+.2f}%. This feeds into every stock's Sentiment score below.</div>
<div class="disclaimer">⚠️ Research shortlist only, not financial advice. Nothing here is a prediction of future performance -- always verify current prices and do your own research before trading.</div>

<input type="radio" name="tabs" id="tab-screener" class="tab-input" checked>
<input type="radio" name="tabs" id="tab-leaderboard" class="tab-input">
<input type="radio" name="tabs" id="tab-tomorrow" class="tab-input">
<input type="radio" name="tabs" id="tab-adaptive" class="tab-input">
<input type="radio" name="tabs" id="tab-runlog" class="tab-input">
<div class="tab-bar">
  <label for="tab-screener" class="tab-label tab-label-screener">📋 Screener</label>
  <label for="tab-leaderboard" class="tab-label tab-label-leaderboard">🏆 Leaderboard{lb_badge}</label>
  <label for="tab-tomorrow" class="tab-label tab-label-tomorrow">🔮 Tomorrow{tomorrow_badge}</label>
  <label for="tab-adaptive" class="tab-label tab-label-adaptive">🧪 Adaptive</label>
  <label for="tab-runlog" class="tab-label tab-label-runlog">🪵 Run Log{runlog_badge}</label>
</div>

<div class="tab-panel tab-panel-screener">
  <div class="muted" style="margin-bottom:16px;">Click a sector name to expand its stocks. Click a symbol (underlined) to expand its full profile inline, if you've fetched it with --company.</div>
  {legend}
  {''.join(rows_html)}
</div>

<div class="tab-panel tab-panel-leaderboard">
  {leaderboard_body}
</div>

<div class="tab-panel tab-panel-tomorrow">
  {tomorrow_body}
</div>

<div class="tab-panel tab-panel-adaptive">
  {adaptive_body}
</div>

<div class="tab-panel tab-panel-runlog">
  <div class="muted" style="margin-bottom:10px;">Routine progress messages from this run -- moved here instead of the terminal. Real errors are highlighted and also still print to the console immediately when they happen.</div>
  {runlog_body}
</div>
</body></html>"""

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    log(f"\nReport written to: {REPORT_PATH}")
    log("Open it in your browser to see sector-wise rankings.")


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="NEPSE sector-wise screener")
    parser.add_argument("--show", action="store_true",
                         help="deprecated / no longer needed -- the browser is now visible by default, "
                              "since nepalstock.com appears to block headless (invisible) browsing.")
    parser.add_argument("--headless", action="store_true",
                         help="run with the browser hidden (NOT recommended -- nepalstock.com has been "
                              "silently failing in headless mode; only use this if you've confirmed it "
                              "still works for you)")
    parser.add_argument("--skip-download", action="store_true", help="only rebuild the report from existing data")
    parser.add_argument("--refresh-sectors", action="store_true", help="force sector map refresh")
    parser.add_argument("--floorsheet", action="store_true",
                         help="deprecated / no longer needed -- floorsheet download now runs automatically "
                              "every time the script runs. Kept as a harmless flag for old habits.")
    parser.add_argument("--skip-floorsheet", action="store_true",
                         help="skip today's floorsheet download (it otherwise runs automatically every time)")
    parser.add_argument("--debug", action="store_true", help="save screenshots during floorsheet/company scraping")
    parser.add_argument("--company", metavar="SYMBOL", help="fetch a deep-dive profile (Financials, Dividend, AGM, Corporate Actions) for one stock, e.g. --company RLFL")
    parser.add_argument("--company-all", action="store_true", help="run --company for every symbol in your data (slow -- meant for background use once --company is confirmed working)")
    parser.add_argument("--leaderboard", action="store_true",
                         help="run the Phase 1-4 rule-based 100-mark leaderboard (sector rotation + broker "
                              "concentration + late-session breakout). Needs --floorsheet data for Rule 1/2 "
                              "to score above 0 -- currently only meaningful for RLFL.")
    parser.add_argument("--adaptive", action="store_true",
                         help="run the adaptive evaluation engine (paper trading / research only, no real "
                              "orders): baseline + relaxed + strict shadow strategies, daily/weekly learning "
                              "reports, and a proposed_strategy_change.md if a shadow strategy earns "
                              "promotion. Writes everything to adaptive/. See adaptive_engine.py's module "
                              "docstring for the trade-management and cost-model assumptions it uses.")
    parser.add_argument("--no-open", action="store_true",
                         help="don't automatically open report.html in your browser when the run finishes "
                              "(it opens by default -- this is for the double-click launcher / unattended "
                              "runs where popping open a browser tab isn't wanted).")
    args = parser.parse_args()

    if args.company:
        os.makedirs(COMPANY_DIR, exist_ok=True)
        fetch_company_profile(args.company.upper(), headless=args.headless, debug=args.debug)
        return

    if args.company_all:
        history = load_history()
        symbols = sorted(history["Symbol"].unique())
        os.makedirs(COMPANY_DIR, exist_ok=True)
        log(f"Fetching company profiles for all {len(symbols)} symbols -- this will take a long time.")
        for i, sym in enumerate(symbols, 1):
            log(f"[{i}/{len(symbols)}] {sym}")
            report_exists = os.path.exists(os.path.join(COMPANY_DIR, sym, "report.html"))
            if report_exists:
                log("  already have this one, skipping.")
                continue
            fetch_company_profile(sym, headless=args.headless, debug=False)
            time.sleep(2)  # be polite to the server across hundreds of requests
        return

    if not args.skip_download:
        missing = find_missing_dates()
        download_missing(missing, headless=args.headless)

    if not args.skip_floorsheet:
        download_floorsheet_missing(headless=args.headless, debug=args.debug)

    sector_map = fetch_sector_map(force=args.refresh_sectors)
    history = load_history()
    latest = compute_indicators(history)
    latest = attach_sectors(latest, sector_map)
    latest = finalize_scores(latest)

    leaderboard_data = None
    if args.leaderboard:
        floorsheet_df = load_latest_floorsheet()
        board, top_sectors = build_leaderboard(latest, history, sector_map, floorsheet_df)
        leaderboard_data = (board, top_sectors, floorsheet_df)

    if args.adaptive:
        # This module aliasing is required for "one execution updates
        # everything" to actually work. Without it: when THIS file runs as
        # the main script, Python's sys.modules key for it is "__main__",
        # not "nepse_screener" -- so adaptive_engine.py's own
        # "import nepse_screener as ns" doesn't find this already-running
        # module, it loads a SECOND, entirely separate copy of this file
        # under the name "nepse_screener". That copy has its own,
        # independent _LOG_MESSAGES list, so anything adaptive_engine logs
        # via ns.log()/ns.log_error() would silently go into a log this
        # script never reads -- the Run Log tab would look complete but
        # actually be missing everything the adaptive cycle did. Registering
        # this running module under the name adaptive_engine.py expects
        # means "import nepse_screener as ns" resolves to THIS exact
        # module object instead of loading a fresh copy.
        sys.modules["nepse_screener"] = sys.modules["__main__"]
        import adaptive_engine
        adaptive_engine.run_adaptive_cycle(headless=args.headless, debug=args.debug)
        # Adaptive runs BEFORE build_report (not after, as before) so its
        # daily_review/strategy_comparison files -- and its log messages,
        # now that they land in the same list -- are both ready in time to
        # appear in this same report.html instead of only showing up on
        # the NEXT run.

    build_report(latest, leaderboard_data)

    if not args.no_open:
        # The whole point of the double-click launcher (run_nepse_screener.bat)
        # is that you never have to go find report.html yourself -- the run
        # finishes and your browser just shows you the result.
        try:
            webbrowser.open("file://" + os.path.abspath(REPORT_PATH))
        except Exception as e:
            log_error(f"Couldn't auto-open the report in your browser ({e}) -- open {REPORT_PATH} manually.")


if __name__ == "__main__":
    main()
