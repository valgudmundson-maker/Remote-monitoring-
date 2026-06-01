# 📈 Same-Day Stock Technical Analysis

A simple web app to enter stock tickers and get **intraday (same-day) technical
analysis** — price action, moving averages, momentum and volatility indicators,
plus derived bullish/bearish signals and a live price/VWAP chart.

## Features

- Enter one or many tickers (e.g. `AAPL, MSFT, TSLA`) — up to 10 at once.
- Same-day intraday data at 5-minute resolution (via Yahoo Finance).
- Indicators computed for each symbol:
  - Price, session open, day high/low, volume, % change
  - **VWAP** (Volume Weighted Average Price)
  - **RSI(14)** — momentum / overbought-oversold
  - **MACD** (12, 26, 9) line, signal and histogram
  - **SMA(20)** and **EMA(20)** moving averages
  - **Bollinger Bands** (20, 2σ)
  - **ATR(14)** — volatility
- Auto-generated **signals** (bullish / bearish / neutral) and an overall verdict.
- Intraday price vs. VWAP chart per ticker.

## Quick start

```bash
# 1. (optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. run the app
python app.py
```

Then open <http://localhost:5000> in your browser and enter a ticker.

## API

The frontend is powered by a small JSON API you can also call directly:

```
GET /api/analyze?tickers=AAPL,MSFT
```

Returns analysis (metrics, signals, overall sentiment and chart series) for each
ticker.

## How it works

- `app.py` — Flask server; fetches intraday OHLCV data with
  [`yfinance`](https://pypi.org/project/yfinance/) and assembles the analysis.
- `indicators.py` — pure pandas/numpy implementations of the technical
  indicators (no native dependencies like TA-Lib required).
- `templates/index.html` — single-page UI (vanilla JS + Chart.js via CDN).

## Notes & limitations

- Intraday data is only available during/around market hours. Outside trading
  hours (or for invalid symbols) you'll see a "no intraday data" message.
- Data is sourced from Yahoo Finance and may be delayed.
- **This is for educational/informational purposes only and is not financial
  advice.**
