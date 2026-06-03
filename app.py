"""Stock ticker technical analysis web app.

Enter one or more stock tickers and get same-day (intraday) technical
analysis: price action, moving averages, RSI, MACD, Bollinger Bands, VWAP,
ATR and a set of derived bullish/bearish signals.
"""

import datetime as dt
import io
import os
import urllib.request
import zlib

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

import indicators as ta

app = Flask(__name__)

# Intraday resolution used for "same day" analysis.
INTERVAL = "5m"
PERIOD = "1d"
# Daily history window for the 50- and 200-day moving averages.
DAILY_PERIOD = "1y"

# Demo mode serves realistic synthetic data instead of calling Yahoo Finance,
# so the UI is fully usable offline / outside market hours. Toggle with the
# DEMO=1 env var or the --demo command-line flag.
DEMO = os.environ.get("DEMO", "").lower() in ("1", "true", "yes")


def _demo_rng(ticker: str, salt: int = 0):
    """Deterministic RNG seeded by the ticker (stable across runs)."""
    seed = zlib.crc32(ticker.encode()) + salt
    return np.random.default_rng(seed)


def _demo_ohlcv(ticker: str, n: int, freq: str, end: pd.Timestamp, vol: float, salt: int):
    """Build a synthetic OHLCV DataFrame via a gentle random walk."""
    rng = _demo_rng(ticker, salt)
    base = 50 + zlib.crc32(ticker.encode()) % 350  # per-ticker price level
    steps = rng.normal(0, vol, n)
    close = base + np.cumsum(steps)
    close = np.maximum(close, 1.0)
    opens = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, vol, n)) + vol * 0.5
    high = np.maximum(opens, close) + spread
    low = np.minimum(opens, close) - spread
    idx = pd.date_range(end=end, periods=n, freq=freq, tz="America/New_York")
    return pd.DataFrame(
        {
            "Open": opens,
            "High": high,
            "Low": np.maximum(low, 0.5),
            "Close": close,
            "Volume": rng.integers(1_000, 50_000, n),
        },
        index=idx,
    )


def _make_ticker(ticker: str):
    """Create a yfinance Ticker (or None in demo mode)."""
    if DEMO:
        return None
    import yfinance as yf

    return yf.Ticker(ticker)


def _fetch_intraday(tk, ticker: str):
    """Fetch same-day intraday (5-minute) OHLCV data.

    In demo mode returns synthetic data; otherwise pulls from Yahoo Finance.
    """
    if DEMO:
        # Anchor the synthetic session to a realistic market close (16:00 ET).
        close_time = pd.Timestamp.now(tz="America/New_York").normalize() + pd.Timedelta(hours=15, minutes=55)
        return _demo_ohlcv(ticker, 78, "5min", close_time, vol=0.4, salt=1)
    return tk.history(period=PERIOD, interval=INTERVAL)


def _fetch_daily(tk, ticker: str):
    """Fetch ~1 year of daily OHLCV data (for daily analysis + 50/200-day MAs)."""
    if DEMO:
        today = pd.Timestamp.now(tz="America/New_York").normalize()
        return _demo_ohlcv(ticker, 260, "1D", today, vol=1.2, salt=2)
    return tk.history(period=DAILY_PERIOD, interval="1d")


def _fetch_stooq_daily(ticker: str):
    """Fetch daily OHLCV from Stooq (free, no API key, cloud-host friendly).

    Used as a fallback when yfinance/Yahoo is unavailable (e.g. Yahoo throttles
    datacenter IPs, so it often fails on cloud hosts). Returns a DataFrame with
    a DatetimeIndex and Open/High/Low/Close/Volume columns, or None.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    # US listings use a ".us" suffix on Stooq; try that first, then bare symbol.
    for sym in (f"{ticker.lower()}.us", ticker.lower()):
        url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                text = resp.read().decode("utf-8", "replace")
        except Exception:
            continue
        if not text or text.lstrip().startswith("<") or "No data" in text:
            continue
        try:
            df = pd.read_csv(io.StringIO(text))
        except Exception:
            continue
        if df.empty or "Close" not in df.columns or "Date" not in df.columns:
            continue
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
        keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
        df = df[keep]
        if "Volume" not in df.columns:
            df["Volume"] = 0
        if df["Close"].notna().sum() >= 1:
            return df
    return None


def _safe(fn, *args):
    """Call a fetch function, returning None on any failure."""
    try:
        return fn(*args)
    except Exception:
        return None



def _build_result(ticker, df, mode, daily_df, source="Yahoo Finance"):
    """Compute the full analysis from an OHLCV frame.

    ``mode`` is "intraday" (5-min, same-day) or "daily" (fallback when the
    market is closed). ``daily_df`` provides the 50/200-day moving averages.
    ``source`` names the data provider used.
    """
    close = df["Close"]
    intraday = mode == "intraday"
    last = -1

    sma20 = ta.sma(close, 20)
    ema20 = ta.ema(close, 20)
    rsi14 = ta.rsi(close, 14)
    macd_line, macd_signal, macd_hist = ta.macd(close)
    bb_mid, bb_upper, bb_lower = ta.bollinger_bands(close, 20)
    atr14 = ta.atr(df, 14)
    lux_osc, lux_signal = ta.ultimate_rsi(close, 14, 14)

    # VWAP is a same-session concept, so only meaningful for intraday data.
    vwap_series = ta.vwap(df) if intraday else None

    # 50/200-day MAs always come from the daily history.
    sma50 = sma200 = None
    if daily_df is not None and not daily_df.empty:
        dclose = daily_df["Close"]
        if len(dclose) >= 50:
            sma50 = ta.sma(dclose, 50).iloc[last]
        if len(dclose) >= 200:
            sma200 = ta.sma(dclose, 200).iloc[last]

    price = float(close.iloc[last])
    if intraday:
        baseline = float(df["Open"].iloc[0])  # change vs session open
    else:
        baseline = float(close.iloc[-2]) if len(close) > 1 else price  # vs prev close
    change = price - baseline
    change_pct = (change / baseline * 100) if baseline else 0.0

    latest = {
        "price": ta._round(price),
        "session_open": ta._round(baseline),
        "day_high": ta._round(df["High"].iloc[last] if not intraday else df["High"].max()),
        "day_low": ta._round(df["Low"].iloc[last] if not intraday else df["Low"].min()),
        "volume": int(df["Volume"].iloc[last] if not intraday else df["Volume"].sum()),
        "change": ta._round(change),
        "change_pct": ta._round(change_pct),
        "sma20": ta._round(sma20.iloc[last]),
        "ema20": ta._round(ema20.iloc[last]),
        "rsi": ta._round(rsi14.iloc[last]),
        "macd": ta._round(macd_line.iloc[last], 4),
        "macd_signal": ta._round(macd_signal.iloc[last], 4),
        "macd_hist": ta._round(macd_hist.iloc[last], 4),
        "bb_upper": ta._round(bb_upper.iloc[last]),
        "bb_middle": ta._round(bb_mid.iloc[last]),
        "bb_lower": ta._round(bb_lower.iloc[last]),
        "vwap": ta._round(vwap_series.iloc[last]) if intraday else None,
        "atr": ta._round(atr14.iloc[last]),
        "lux_osc": ta._round(lux_osc.iloc[last]),
        "lux_signal": ta._round(lux_signal.iloc[last]),
        "sma50": ta._round(sma50),
        "sma200": ta._round(sma200),
    }

    signals = ta.build_signals(latest)

    # Chart series. Intraday shows the whole session; daily shows the last ~60
    # trading days for a readable trend.
    view = df if intraday else df.iloc[-60:]
    vclose = view["Close"]
    vlux = lux_osc.iloc[-len(view):]
    vsig = lux_signal.iloc[-len(view):]
    time_fmt = "%H:%M" if intraday else "%m-%d"
    chart = {
        "times": [t.strftime(time_fmt) for t in view.index],
        "close": [ta._round(v) for v in vclose.tolist()],
        "vwap": [ta._round(v) for v in vwap_series.tolist()] if intraday else [None] * len(view),
        "lux_osc": [ta._round(v) for v in vlux.tolist()],
        "lux_signal": [ta._round(v) for v in vsig.tolist()],
    }

    note = None if intraday else (
        "Showing latest daily analysis (no same-day intraday data available)."
    )

    return {
        "ticker": ticker,
        "mode": mode,
        "interval": INTERVAL if intraday else "1d",
        "as_of": view.index[-1].strftime("%Y-%m-%d %H:%M %Z").strip(),
        "source": source,
        "metrics": latest,
        "signals": signals,
        "overall": ta.overall_sentiment(signals),
        "chart": chart,
        "note": note,
        "error": None,
    }


def analyze(ticker: str) -> dict:
    """Run the full technical analysis for a single ticker.

    Uses same-day intraday data when available, and falls back to daily
    analysis (still including the 50/200-day MAs) when the market is closed.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        return {"ticker": ticker, "error": "Empty ticker symbol."}

    try:
        tk = _make_ticker(ticker)
    except Exception as exc:  # yfinance import / construction failure
        return {"ticker": ticker, "error": f"Failed to initialise data source: {exc}"}

    intraday = _safe(_fetch_intraday, tk, ticker)
    daily = _safe(_fetch_daily, tk, ticker)
    daily_source = "Yahoo Finance"

    # Yahoo often throttles cloud/datacenter IPs, so fall back to Stooq (which
    # does not) for the daily series when Yahoo returns nothing.
    if (daily is None or daily.empty) and not DEMO:
        stooq = _safe(_fetch_stooq_daily, ticker)
        if stooq is not None and not stooq.empty:
            daily, daily_source = stooq, "Stooq"

    if intraday is not None and not intraday.empty:
        return _build_result(ticker, intraday, "intraday", daily, "Yahoo Finance")
    if daily is not None and not daily.empty:
        return _build_result(ticker, daily, "daily", daily, daily_source)

    return {
        "ticker": ticker,
        "error": "No data available — the symbol may be invalid, or the data "
        "provider is temporarily unavailable. Please try again.",
    }


@app.route("/")
def index():
    return render_template("index.html", demo=DEMO)


@app.route("/api/analyze")
def api_analyze():
    """Analyze one or more comma/space separated tickers."""
    raw = request.args.get("tickers", "") or request.args.get("ticker", "")
    symbols = [s for s in raw.replace(",", " ").split() if s]
    if not symbols:
        return jsonify({"error": "No ticker(s) provided."}), 400

    # Cap to avoid abuse / long requests.
    symbols = symbols[:10]
    results = [analyze(sym) for sym in symbols]
    return jsonify({"generated_at": dt.datetime.now().isoformat(timespec="seconds"),
                    "results": results})


if __name__ == "__main__":
    import sys

    if "--demo" in sys.argv:
        DEMO = True
        print("Running in DEMO mode: serving synthetic data (no live market data).")

    app.run(host="0.0.0.0", port=5000, debug=True)
