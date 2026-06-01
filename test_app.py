"""Smoke tests for the technical analysis app.

These run without network access by mocking yfinance, so they are safe for CI.
Run with: pytest -q
"""

import sys
import types

import numpy as np
import pandas as pd

import indicators as ta


def _make_df(n, freq, start, seed):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq, tz="America/New_York")
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    return pd.DataFrame(
        {
            "Open": np.r_[close[0], close[:-1]],
            "High": close + 0.4,
            "Low": close - 0.4,
            "Close": close,
            "Volume": rng.integers(1000, 9000, n),
        },
        index=idx,
    )


def _install_fake_yfinance():
    """Register a fake yfinance module so app._fetch_* work offline."""

    class FakeTicker:
        def __init__(self, ticker):
            self.ticker = ticker

        def history(self, period, interval):
            if interval == "1d":
                return _make_df(260, "1D", "2025-06-01", seed=1)  # ~1y daily
            return _make_df(78, "5min", "2026-06-01 09:30", seed=2)  # intraday

    fake = types.ModuleType("yfinance")
    fake.Ticker = FakeTicker
    sys.modules["yfinance"] = fake


# ---- Indicator math ----------------------------------------------------------

def test_ultimate_rsi_bounded_and_no_nan_after_warmup():
    rng = np.random.default_rng(7)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.4, 120)))
    osc, signal = ta.ultimate_rsi(close, 14, 14)
    valid = osc.dropna()
    assert valid.min() >= 0 and valid.max() <= 100
    assert not osc.iloc[20:].isna().any()
    assert not signal.iloc[20:].isna().any()


def test_rsi_extremes():
    rising = pd.Series(np.arange(1, 50, dtype=float))
    falling = pd.Series(np.arange(50, 1, -1, dtype=float))
    assert ta.rsi(rising, 14).iloc[-1] > 95
    assert ta.rsi(falling, 14).iloc[-1] < 5


def test_round_handles_nan_and_none():
    assert ta._round(None) is None
    assert ta._round(float("nan")) is None
    assert ta._round(1.23456, 2) == 1.23


def test_build_signals_includes_new_indicators():
    latest = {
        "price": 105.0, "rsi": 65.0, "macd": 0.1, "macd_signal": 0.05,
        "sma20": 104.0, "vwap": 103.0, "bb_upper": 110.0, "bb_lower": 100.0,
        "lux_osc": 85.0, "lux_signal": 70.0, "sma50": 102.0, "sma200": 108.0,
    }
    names = {s["name"] for s in ta.build_signals(latest)}
    assert {"LuxAlgo", "SMA(50)", "SMA(200)", "Trend"} <= names


def test_overall_sentiment():
    bull = [{"sentiment": "bullish"}, {"sentiment": "bullish"}, {"sentiment": "bearish"}]
    bear = [{"sentiment": "bearish"}, {"sentiment": "neutral"}]
    assert ta.overall_sentiment(bull) == "bullish"
    assert ta.overall_sentiment(bear) == "bearish"
    assert ta.overall_sentiment([]) == "neutral"


# ---- Full flow (mocked data) -------------------------------------------------

def test_analyze_full_flow():
    _install_fake_yfinance()
    import app

    res = app.analyze("AAPL")
    assert res["error"] is None
    m = res["metrics"]
    for key in ("price", "vwap", "rsi", "lux_osc", "lux_signal", "sma50", "sma200"):
        assert key in m
    assert len(res["chart"]["lux_osc"]) == len(res["chart"]["times"])
    assert res["overall"] in ("bullish", "bearish", "neutral")


def test_api_endpoint():
    _install_fake_yfinance()
    import app

    client = app.app.test_client()

    # No tickers -> 400
    assert client.get("/api/analyze").status_code == 400

    # Multiple tickers -> one result each
    payload = client.get("/api/analyze?tickers=AAPL,MSFT").get_json()
    assert len(payload["results"]) == 2
    assert "sma200" in payload["results"][0]["metrics"]

    # Index page renders
    assert client.get("/").status_code == 200
