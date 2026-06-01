"""Technical analysis indicator calculations.

All functions operate on pandas Series/DataFrames and use only pandas/numpy
so the project has no native (C library) dependencies like TA-Lib.
"""

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period, min_periods=1).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    # When there are no losses RSI is 100; when no gains it is 0.
    out = out.where(avg_loss != 0, 100)
    out = out.where(avg_gain != 0, out.where(avg_loss != 0, 0))
    return out


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Moving Average Convergence Divergence.

    Returns (macd_line, signal_line, histogram).
    """
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0):
    """Bollinger Bands. Returns (middle, upper, lower)."""
    middle = sma(series, period)
    std = series.rolling(window=period, min_periods=1).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return middle, upper, lower


def ultimate_rsi(src: pd.Series, length: int = 14, smooth: int = 14):
    """LuxAlgo "Ultimate RSI" oscillator (open-source LuxAlgo indicator).

    A range-aware variant of RSI: instead of raw price changes it measures
    change relative to the rolling high/low range, then normalises to 0-100.
    Returns (oscillator, signal_line). Overbought ~80, oversold ~20, midline 50.
    """
    upper = src.rolling(window=length, min_periods=1).max()
    lower = src.rolling(window=length, min_periods=1).min()
    r = upper - lower
    d = src.diff()

    # Use the full range when the channel expands, otherwise the bar change.
    diff = np.where(
        upper > upper.shift(1), r,
        np.where(lower < lower.shift(1), -r, d),
    )
    diff = pd.Series(diff, index=src.index).fillna(0.0)

    # Wilder's (RMA) smoothing, matching the LuxAlgo default.
    num = diff.ewm(alpha=1 / length, adjust=False).mean()
    den = diff.abs().ewm(alpha=1 / length, adjust=False).mean()
    arsi = (num / den.replace(0, np.nan)) * 50 + 50

    # Signal line: EMA of the oscillator (LuxAlgo default smoothing).
    signal = arsi.ewm(span=smooth, adjust=False).mean()
    return arsi, signal


def vwap(df: pd.DataFrame) -> pd.Series:
    """Volume Weighted Average Price (cumulative over the session)."""
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_vol = df["Volume"].cumsum().replace(0, np.nan)
    return (typical_price * df["Volume"]).cumsum() / cum_vol


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range, a volatility measure."""
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def _round(value, digits=2):
    """Round a possibly-NaN numeric value to a JSON-friendly float or None."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(f) or np.isinf(f):
        return None
    return round(f, digits)


def build_signals(latest: dict) -> list:
    """Derive simple human-readable bullish/bearish signals from indicators."""
    signals = []

    rsi_val = latest.get("rsi")
    if rsi_val is not None:
        if rsi_val >= 70:
            signals.append(("RSI", "bearish", f"Overbought (RSI {rsi_val})"))
        elif rsi_val <= 30:
            signals.append(("RSI", "bullish", f"Oversold (RSI {rsi_val})"))
        else:
            signals.append(("RSI", "neutral", f"Neutral (RSI {rsi_val})"))

    macd_line = latest.get("macd")
    signal_line = latest.get("macd_signal")
    if macd_line is not None and signal_line is not None:
        if macd_line > signal_line:
            signals.append(("MACD", "bullish", "MACD above signal line"))
        else:
            signals.append(("MACD", "bearish", "MACD below signal line"))

    price = latest.get("price")
    sma20 = latest.get("sma20")
    if price is not None and sma20 is not None:
        if price > sma20:
            signals.append(("SMA(20)", "bullish", "Price above SMA(20)"))
        else:
            signals.append(("SMA(20)", "bearish", "Price below SMA(20)"))

    vwap_val = latest.get("vwap")
    if price is not None and vwap_val is not None:
        if price > vwap_val:
            signals.append(("VWAP", "bullish", "Price above VWAP"))
        else:
            signals.append(("VWAP", "bearish", "Price below VWAP"))

    upper = latest.get("bb_upper")
    lower = latest.get("bb_lower")
    if price is not None and upper is not None and lower is not None:
        if price >= upper:
            signals.append(("Bollinger", "bearish", "Price at/above upper band"))
        elif price <= lower:
            signals.append(("Bollinger", "bullish", "Price at/below lower band"))
        else:
            signals.append(("Bollinger", "neutral", "Price within bands"))

    lux = latest.get("lux_osc")
    lux_sig = latest.get("lux_signal")
    if lux is not None:
        if lux >= 80:
            signals.append(("LuxAlgo", "bearish", f"Overbought (osc {lux})"))
        elif lux <= 20:
            signals.append(("LuxAlgo", "bullish", f"Oversold (osc {lux})"))
        elif lux_sig is not None and lux > lux_sig:
            signals.append(("LuxAlgo", "bullish", "Oscillator above signal line"))
        elif lux_sig is not None:
            signals.append(("LuxAlgo", "bearish", "Oscillator below signal line"))
        else:
            signals.append(("LuxAlgo", "neutral", f"Oscillator {lux}"))

    sma50 = latest.get("sma50")
    if price is not None and sma50 is not None:
        if price > sma50:
            signals.append(("SMA(50)", "bullish", "Price above 50-day MA"))
        else:
            signals.append(("SMA(50)", "bearish", "Price below 50-day MA"))

    sma200 = latest.get("sma200")
    if price is not None and sma200 is not None:
        if price > sma200:
            signals.append(("SMA(200)", "bullish", "Price above 200-day MA"))
        else:
            signals.append(("SMA(200)", "bearish", "Price below 200-day MA"))

    if sma50 is not None and sma200 is not None:
        if sma50 > sma200:
            signals.append(("Trend", "bullish", "Golden cross (50d > 200d)"))
        else:
            signals.append(("Trend", "bearish", "Death cross (50d < 200d)"))

    return [{"name": n, "sentiment": s, "detail": d} for n, s, d in signals]


def overall_sentiment(signals: list) -> str:
    """Aggregate individual signals into an overall verdict."""
    bull = sum(1 for s in signals if s["sentiment"] == "bullish")
    bear = sum(1 for s in signals if s["sentiment"] == "bearish")
    if bull > bear:
        return "bullish"
    if bear > bull:
        return "bearish"
    return "neutral"
