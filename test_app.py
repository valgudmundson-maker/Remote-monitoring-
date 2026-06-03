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


def test_demo_mode_offline(monkeypatch):
    """Demo mode must produce a full result with no network access."""
    import app

    # Ensure any real yfinance import would fail, proving demo uses no network.
    monkeypatch.setitem(sys.modules, "yfinance", None)
    monkeypatch.setattr(app, "DEMO", True)

    res = app.analyze("AAPL")
    assert res["error"] is None
    m = res["metrics"]
    assert m["price"] is not None and m["sma50"] is not None and m["sma200"] is not None

    # Deterministic per ticker, distinct across tickers.
    assert app.analyze("AAPL")["metrics"]["price"] == m["price"]
    assert app.analyze("TSLA")["metrics"]["price"] != m["price"]


def _install_yf(intraday_df, daily_df):
    """Install a fake yfinance returning the given intraday/daily frames."""
    class FakeTicker:
        def __init__(self, ticker):
            self.ticker = ticker

        def history(self, period, interval):
            return daily_df if interval == "1d" else intraday_df

    fake = types.ModuleType("yfinance")
    fake.Ticker = FakeTicker
    sys.modules["yfinance"] = fake


def test_daily_fallback_when_intraday_empty(monkeypatch):
    """When intraday data is empty (market closed), fall back to daily."""
    empty = pd.DataFrame()
    daily = _make_df(260, "1D", "2025-06-01", seed=4)
    _install_yf(empty, daily)

    import app
    monkeypatch.setattr(app, "DEMO", False)

    res = app.analyze("AAPL")
    assert res["error"] is None
    assert res["mode"] == "daily"
    assert res["note"]  # explanatory banner present
    assert res["interval"] == "1d"
    m = res["metrics"]
    assert m["vwap"] is None  # VWAP is intraday-only
    assert m["sma50"] is not None and m["sma200"] is not None
    assert len(res["chart"]["times"]) <= 60


def test_error_when_no_data(monkeypatch):
    """Both intraday and daily empty -> a clear error, no crash."""
    empty = pd.DataFrame()
    _install_yf(empty, empty)

    import app
    monkeypatch.setattr(app, "DEMO", False)

    res = app.analyze("BADTICKER")
    assert res.get("error")
    assert "mode" not in res or res.get("metrics") is None


def test_intraday_preferred_when_available(monkeypatch):
    """Intraday data, when present, is used over the daily fallback."""
    intraday = _make_df(78, "5min", "2026-06-01 09:30", seed=5)
    daily = _make_df(260, "1D", "2025-06-01", seed=6)
    _install_yf(intraday, daily)

    import app
    monkeypatch.setattr(app, "DEMO", False)

    res = app.analyze("AAPL")
    assert res["mode"] == "intraday"
    assert res["metrics"]["vwap"] is not None


def _fake_stooq_csv(n=300):
    """Build a Stooq-style daily CSV string."""
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    rows = ["Date,Open,High,Low,Close,Volume"]
    price = 100.0
    for d in dates:
        price += 0.1
        rows.append(f"{d.date()},{price:.2f},{price+1:.2f},{price-1:.2f},{price:.2f},10000")
    return "\n".join(rows) + "\n"


def test_stooq_parsing(monkeypatch):
    """_fetch_stooq_daily parses a CSV response into a clean OHLCV frame."""
    import contextlib
    import app

    csv = _fake_stooq_csv(250)

    class FakeResp:
        def read(self):
            return csv.encode()

    @contextlib.contextmanager
    def fake_urlopen(req, timeout=0):
        yield FakeResp()

    monkeypatch.setattr(app.urllib.request, "urlopen", fake_urlopen)
    df = app._fetch_stooq_daily("AAPL")
    assert df is not None and not df.empty
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df) == 250


def test_stooq_fallback_when_yahoo_empty(monkeypatch):
    """When Yahoo returns nothing, analyze() falls back to Stooq (daily)."""
    _install_yf(pd.DataFrame(), pd.DataFrame())  # Yahoo intraday + daily empty

    import app
    monkeypatch.setattr(app, "DEMO", False)
    monkeypatch.setattr(app, "_fetch_stooq_daily",
                        lambda ticker: _make_df(260, "1D", "2025-06-01", seed=9))

    res = app.analyze("AAPL")
    assert res["error"] is None
    assert res["mode"] == "daily"
    assert res["source"] == "Stooq"
    assert res["metrics"]["sma200"] is not None


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
