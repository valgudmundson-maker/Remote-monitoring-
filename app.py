"""Stock ticker technical analysis web app.

Enter one or more stock tickers and get same-day (intraday) technical
analysis: price action, moving averages, RSI, MACD, Bollinger Bands, VWAP,
ATR and a set of derived bullish/bearish signals.
"""

import datetime as dt

from flask import Flask, jsonify, render_template, request

import indicators as ta

app = Flask(__name__)

# Intraday resolution used for "same day" analysis.
INTERVAL = "5m"
PERIOD = "1d"


def _fetch_history(ticker: str):
    """Fetch same-day intraday OHLCV data for a ticker via yfinance.

    Imported lazily so the module loads even if the network/yfinance is
    unavailable at import time.
    """
    import yfinance as yf

    tk = yf.Ticker(ticker)
    df = tk.history(period=PERIOD, interval=INTERVAL)
    return tk, df


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
    }

    signals = ta.build_signals(latest)

    # Build a compact price + VWAP series for charting on the frontend.
    chart = {
        "times": [t.strftime("%H:%M") for t in df.index],
        "close": [ta._round(v) for v in close.tolist()],
        "vwap": [ta._round(v) for v in vwap_series.tolist()],
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
    app.run(host="0.0.0.0", port=5000, debug=True)
