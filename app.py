"""Stock ticker technical analysis web app.

Enter one or more stock tickers and get same-day (intraday) technical
analysis: price action, moving averages, RSI, MACD, Bollinger Bands, VWAP,
ATR and a set of derived bullish/bearish signals.
"""

import datetime as dt
import os
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


def _fetch_history(ticker: str):
    """Fetch same-day intraday OHLCV data for a ticker.

    In demo mode returns synthetic data; otherwise pulls from Yahoo Finance via
    yfinance (imported lazily so the module loads even if it is unavailable).
    """
    if DEMO:
        # Anchor the synthetic session to a realistic market close (16:00 ET).
        close_time = pd.Timestamp.now(tz="America/New_York").normalize() + pd.Timedelta(hours=15, minutes=55)
        return None, _demo_ohlcv(ticker, 78, "5min", close_time, vol=0.4, salt=1)

    import yfinance as yf

    tk = yf.Ticker(ticker)
    df = tk.history(period=PERIOD, interval=INTERVAL)
    return tk, df


def _fetch_daily(tk, ticker: str):
    """Fetch ~1 year of daily OHLCV data (for the 50/200-day MAs)."""
    if DEMO:
        today = pd.Timestamp.now(tz="America/New_York").normalize()
        return _demo_ohlcv(ticker, 260, "1D", today, vol=1.2, salt=2)
    return tk.history(period=DAILY_PERIOD, interval="1d")


def analyze(ticker: str) -> dict:
    """Run the full technical analysis for a single ticker."""
    ticker = ticker.strip().upper()
    if not ticker:
        return {"ticker": ticker, "error": "Empty ticker symbol."}

    try:
        tk, df = _fetch_history(ticker)
    except Exception as exc:  # network / library failure
        return {"ticker": ticker, "error": f"Failed to fetch data: {exc}"}

    if df is None or df.empty:
        return {
            "ticker": ticker,
            "error": "No intraday data available (market may be closed, "
            "or the symbol is invalid).",
        }

    close = df["Close"]

    # Indicator series.
    sma20 = ta.sma(close, 20)
    ema20 = ta.ema(close, 20)
    rsi14 = ta.rsi(close, 14)
    macd_line, macd_signal, macd_hist = ta.macd(close)
    bb_mid, bb_upper, bb_lower = ta.bollinger_bands(close, 20)
    vwap_series = ta.vwap(df)
    atr14 = ta.atr(df, 14)
    lux_osc, lux_signal = ta.ultimate_rsi(close, 14, 14)

    # 50- and 200-day simple moving averages require daily data over a longer
    # window than the intraday session, so fetch it separately.
    sma50 = sma200 = None
    try:
        daily = _fetch_daily(tk, ticker)
        if daily is not None and not daily.empty:
            dclose = daily["Close"]
            if len(dclose) >= 50:
                sma50 = ta.sma(dclose, 50).iloc[-1]
            if len(dclose) >= 200:
                sma200 = ta.sma(dclose, 200).iloc[-1]
    except Exception:
        pass  # daily MAs are best-effort; intraday analysis still works.

    last = -1
    price = float(close.iloc[last])
    session_open = float(df["Open"].iloc[0])
    change = price - session_open
    change_pct = (change / session_open * 100) if session_open else 0.0

    latest = {
        "price": ta._round(price),
        "session_open": ta._round(session_open),
        "day_high": ta._round(df["High"].max()),
        "day_low": ta._round(df["Low"].min()),
        "volume": int(df["Volume"].sum()),
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
        "vwap": ta._round(vwap_series.iloc[last]),
        "atr": ta._round(atr14.iloc[last]),
        "lux_osc": ta._round(lux_osc.iloc[last]),
        "lux_signal": ta._round(lux_signal.iloc[last]),
        "sma50": ta._round(sma50),
        "sma200": ta._round(sma200),
    }

    signals = ta.build_signals(latest)

    # Build a compact price + VWAP series for charting on the frontend.
    chart = {
        "times": [t.strftime("%H:%M") for t in df.index],
        "close": [ta._round(v) for v in close.tolist()],
        "vwap": [ta._round(v) for v in vwap_series.tolist()],
        "lux_osc": [ta._round(v) for v in lux_osc.tolist()],
        "lux_signal": [ta._round(v) for v in lux_signal.tolist()],
    }

    return {
        "ticker": ticker,
        "interval": INTERVAL,
        "as_of": df.index[-1].strftime("%Y-%m-%d %H:%M %Z").strip(),
        "metrics": latest,
        "signals": signals,
        "overall": ta.overall_sentiment(signals),
        "chart": chart,
        "error": None,
    }


@app.route("/")
def index():
    return render_template("index.html")


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
