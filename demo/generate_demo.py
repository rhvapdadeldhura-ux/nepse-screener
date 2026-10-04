"""
Generates a realistic-looking but entirely FICTIONAL dataset and runs the
real nepse_screener.py / adaptive_engine.py pipeline against it, so the
resulting report.html is produced by the actual, unmodified scoring and
reporting code -- not a hand-faked screenshot. All tickers, sector names,
prices and broker names below are invented for demo purposes only.
"""
import os, sys, json, random, datetime as dt
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import nepse_screener as ns

# ---------------------------------------------------------------------------
# 1. Fictional universe: 8 sectors x 5 symbols
# ---------------------------------------------------------------------------
SECTORS = {
    "Banking":            ["HMBL", "GRKB", "SNFB", "PCNB", "UDBL"],
    "Hydropower":         ["BLHY", "TRHP", "KRNH", "SNHY", "MRGH"],
    "Microfinance":       ["GRMF", "ANMF", "LXMF", "SHMF", "VKMF"],
    "Insurance":          ["HMIN", "NBIC", "SGIL", "PRIN", "CTIN"],
    "Hotels & Tourism":   ["HMTR", "SGTV", "ANHT", "PKHT", "RLHT"],
    "Manufacturing":      ["GRMN", "SNMF", "LXMN", "BHMN", "KTMN"],
    "Finance":            ["NBFC", "PCFN", "UDFC", "SHFC", "TRFC"],
    "Investment":         ["HMIV", "GRIV", "SNIV", "ANIV", "VKIV"],
}
SECTOR_MAP = {sym: sec for sec, syms in SECTORS.items() for sym in syms}
ALL_SYMBOLS = sorted(SECTOR_MAP)

BROKERS = [f"Sample Securities {n}" for n in ["Alpha", "Beta", "Gamma", "Delta",
                                               "Epsilon", "Zeta", "Eta", "Theta"]]

# ---------------------------------------------------------------------------
# 2. 45 trading days of price history (random walk, one CSV per day --
#    matches the real daily export: all symbols in one file per date)
# ---------------------------------------------------------------------------
END_DATE = dt.date(2026, 10, 4)
trading_dates = []
d = END_DATE
while len(trading_dates) < 45:
    if d.weekday() != 5:  # skip Saturday (NEPSE's weekly holiday)
        trading_dates.append(d)
    d -= dt.timedelta(days=1)
trading_dates = sorted(trading_dates)

os.makedirs(ns.DATA_DIR, exist_ok=True)
prices = {sym: 200 + random.random() * 1800 for sym in ALL_SYMBOLS}

for day_i, day in enumerate(trading_dates):
    rows = []
    is_last_day = (day == trading_dates[-1])
    for sym in ALL_SYMBOLS:
        drift = 0.0015 if sym in ("SNHY", "GRMF", "HMBL") else 0.0  # a few consistent "winners"
        # On the final day, give a handful of symbols a clear breakout candle
        # (big volume + strong up-close) so the leaderboard has something to
        # actually surface -- otherwise every demo run looks flat.
        breakout = is_last_day and sym in ("SNHY", "GRMF", "BLHY")
        shock = np.random.normal(drift, 0.018) + (0.06 if breakout else 0)
        prices[sym] = max(5, prices[sym] * (1 + shock))
        close = round(prices[sym], 2)
        open_ = round(close * (1 - np.random.uniform(-0.01, 0.01)), 2)
        high = round(max(open_, close) * (1 + abs(np.random.uniform(0, 0.012))), 2)
        low = round(min(open_, close) * (1 - abs(np.random.uniform(0, 0.012))), 2)
        base_qty = np.random.randint(2_000, 40_000)
        qty = int(base_qty * (3.2 if breakout else 1))
        turnover = round(close * qty, 2)
        rows.append({
            "Business Date": day.isoformat(),
            "Symbol": sym,
            "Open Price": open_,
            "High Price": high,
            "Low Price": low,
            "Close Price": close,
            "Total Traded Quantity": qty,
            "Total Trades": max(5, qty // 150),
            "Total Turnover": turnover,
        })
    df = pd.DataFrame(rows)
    out = os.path.join(ns.DATA_DIR, f"nepse_{day.isoformat()}.csv")
    df.to_csv(out, index=False)

print(f"Wrote {len(trading_dates)} daily price files -> {ns.DATA_DIR}")

# ---------------------------------------------------------------------------
# 3. Floorsheet for the latest day -- concentrate SNHY's buys on one broker
#    (broker-concentration signal) and cluster GRMF's biggest trades late in
#    the session (late-session-breakout signal), so Rule 1 / Rule 2 actually
#    have something to find.
# ---------------------------------------------------------------------------
os.makedirs(ns.FLOORSHEET_DIR, exist_ok=True)
fs_rows = []
cid = 1_000_000
last_day = trading_dates[-1]


def add_trade(symbol, buyer, seller, qty, rate, hour, minute):
    global cid
    cid += 1
    fs_rows.append({
        "contractId": cid,
        "stockSymbol": symbol,
        "buyerMemberId": BROKERS.index(buyer) + 1,
        "buyerBrokerName": buyer,
        "sellerMemberId": BROKERS.index(seller) + 1,
        "sellerBrokerName": seller,
        "contractQuantity": qty,
        "contractRate": rate,
        "contractAmount": round(qty * rate, 2),
        "businessDate": last_day.isoformat(),
        "tradeBookId": 1,
        "stockId": ALL_SYMBOLS.index(symbol) + 1,
        "tradeTime": f"{last_day.isoformat()}T{hour:02d}:{minute:02d}:00.000",
        "securityName": symbol,
    })


for sym in ALL_SYMBOLS:
    n_trades = np.random.randint(8, 20)
    for i in range(n_trades):
        hour = np.random.choice([11, 12, 13, 14])
        minute = np.random.randint(0, 59)
        buyer, seller = np.random.choice(BROKERS, size=2, replace=False)
        qty = int(np.random.randint(50, 2000))
        rate = round(prices[sym] * (1 + np.random.uniform(-0.01, 0.01)), 2)
        add_trade(sym, buyer, seller, qty, rate, hour, minute)

# Broker concentration: "Sample Securities Alpha" does most of SNHY's buying
for i in range(14):
    minute = np.random.randint(0, 59)
    add_trade("SNHY", "Sample Securities Alpha", np.random.choice(BROKERS),
               int(np.random.randint(500, 3000)), round(prices["SNHY"] * 1.01, 2), 13, minute)

# Late-session breakout: GRMF's largest trades cluster in the last 15 minutes
for i in range(10):
    minute = np.random.randint(45, 59)
    add_trade("GRMF", np.random.choice(BROKERS), np.random.choice(BROKERS),
               int(np.random.randint(3000, 8000)), round(prices["GRMF"] * 1.015, 2), 14, minute)

fs_df = pd.DataFrame(fs_rows).sort_values("contractId", ascending=False).reset_index(drop=True)
fs_df.insert(0, "rowNumber", range(1, len(fs_df) + 1))
fs_path = os.path.join(ns.FLOORSHEET_DIR, f"floorsheet_{last_day.isoformat()}.csv")
fs_df.to_csv(fs_path, index=False)
print(f"Wrote floorsheet -> {fs_path} ({len(fs_df)} rows)")

# ---------------------------------------------------------------------------
# 4. Sector-summary "scrape" cache (pre-seeded so the real network scraper
#    never runs for this demo) + sector map cache
# ---------------------------------------------------------------------------
import adaptive_engine as ae
os.makedirs(ae.INDEX_DIR, exist_ok=True)
idx_rows = [{"Sector Name": "NEPSE Index", "Percent Change": round(np.random.uniform(-0.6, 0.9), 2)}]
for sec in SECTORS:
    idx_rows.append({"Sector Name": sec, "Percent Change": round(np.random.uniform(-1.5, 2.2), 2)})
pd.DataFrame(idx_rows).to_csv(os.path.join(ae.INDEX_DIR, f"index_{dt.date.today().isoformat()}.csv"), index=False)

with open(ns.SECTOR_CACHE, "w") as f:
    json.dump({"fetched_at": dt.datetime.now().isoformat(), "map": SECTOR_MAP}, f)

print("Sample dataset generation complete.")
