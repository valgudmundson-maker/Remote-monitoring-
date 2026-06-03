"""Plain-English "so what?" summaries.

Turns the raw technical metrics and company fundamentals into short, decision-
oriented statements for someone looking for a reasonable entry point for a
medium-to-long-term position.
"""


def technical_summary(metrics: dict, overall: str) -> str:
    """One-paragraph "so what" for the technicals, framed for entry timing."""
    price = metrics.get("price")
    sma50 = metrics.get("sma50")
    sma200 = metrics.get("sma200")
    rsi = metrics.get("rsi")

    trend_bull = None
    bits = []
    if price is not None and sma200 is not None:
        if price > sma200:
            bits.append("trading above its 200-day average (long-term uptrend)")
            trend_bull = True
        else:
            bits.append("trading below its 200-day average (long-term downtrend)")
            trend_bull = False
    if sma50 is not None and sma200 is not None:
        bits.append(
            "with a golden cross (50-day above 200-day)"
            if sma50 > sma200
            else "with a death cross (50-day below 200-day)"
        )

    timing = None
    if rsi is not None:
        timing = "overbought" if rsi >= 70 else "oversold" if rsi <= 30 else "neutral"

    if trend_bull is True and timing == "overbought":
        verdict = (
            "The long-term trend is healthy, but momentum looks stretched. For a "
            "medium/long-term entry, consider waiting for a pullback or scaling in "
            "gradually rather than buying all at once."
        )
    elif trend_bull is True and timing == "oversold":
        verdict = (
            "The long-term trend is up and the stock has pulled back to oversold "
            "levels — often a more attractive entry for a longer-term position."
        )
    elif trend_bull is True:
        verdict = (
            "The long-term trend is constructive and momentum is neutral — a "
            "reasonable backdrop for starting a medium/long-term position."
        )
    elif trend_bull is False and timing == "oversold":
        verdict = (
            "The long-term trend is still down and the stock is oversold, so a bounce "
            "is possible, but for a longer-term entry it is usually safer to wait for "
            "price to reclaim the 200-day average before committing."
        )
    elif trend_bull is False:
        verdict = (
            "The long-term trend is weak (below the 200-day average). A medium/long-term "
            "buyer may want to wait for price to stabilize above the 200-day average "
            "rather than buy into a downtrend."
        )
    else:
        verdict = (
            "There isn't enough history to judge the long-term trend yet; lean on "
            "fundamentals and a longer data window before committing."
        )

    desc = ("The stock is " + ", ".join(bits) + ". ") if bits else ""
    if timing is not None:
        desc += f"Short-term momentum (RSI {rsi}) is {timing}. "
    return desc + verdict


def _pct(x):
    return f"{x * 100:.0f}%"


def fundamental_summary(f: dict) -> str:
    """One-paragraph "so what" for the fundamentals, for a long-term holder."""
    if not f:
        return None

    name = f.get("name") or "This company"
    pe = f.get("pe")
    peg = f.get("peg")
    margin = f.get("profit_margin")
    growth = f.get("rev_growth")
    price = f.get("price")
    target = f.get("target")
    div = f.get("dividend_yield")

    parts = []
    if pe is not None and pe > 0:
        if pe < 15:
            val = f"looks inexpensively valued (P/E {pe:.0f})"
        elif pe < 25:
            val = f"is reasonably valued (P/E {pe:.0f})"
        elif pe < 40:
            val = f"trades at an elevated valuation (P/E {pe:.0f})"
        else:
            val = f"is richly valued (P/E {pe:.0f})"
        if peg is not None and peg > 0:
            val += f", PEG {peg:.2f}"
        parts.append(val)
    elif pe is not None:
        parts.append("has negative trailing earnings (no meaningful P/E)")

    if growth is not None:
        if growth > 0.15:
            parts.append(f"strong revenue growth ({_pct(growth)} YoY)")
        elif growth > 0.05:
            parts.append(f"moderate revenue growth ({_pct(growth)} YoY)")
        elif growth >= 0:
            parts.append(f"slow revenue growth ({_pct(growth)} YoY)")
        else:
            parts.append(f"declining revenue ({_pct(growth)} YoY)")

    if margin is not None:
        if margin > 0.15:
            parts.append(f"strong profitability ({_pct(margin)} net margin)")
        elif margin > 0.05:
            parts.append(f"healthy profitability ({_pct(margin)} net margin)")
        elif margin >= 0:
            parts.append(f"thin profitability ({_pct(margin)} net margin)")
        else:
            parts.append(f"currently unprofitable ({_pct(margin)} net margin)")

    cheap = pe is not None and 0 < pe < 25
    growing = growth is not None and growth > 0.05
    profitable = margin is not None and margin > 0.05

    if cheap and growing and profitable:
        verdict = ("Fundamentally solid for a long-term holder: reasonable valuation, real "
                   "growth and genuine profitability.")
    elif growing and profitable and not cheap:
        verdict = ("A quality, growing business — but you'd be paying a premium, which is fine "
                   "for long-term holders comfortable with valuation risk.")
    elif margin is not None and margin < 0:
        verdict = ("More speculative: not yet consistently profitable, so size any long-term "
                   "position accordingly.")
    elif growth is not None and growth <= 0.05:
        verdict = "Stable but slow-growing — more of a value/income hold than a growth story."
    else:
        verdict = "Mixed fundamental picture; weigh valuation against growth for your time horizon."

    if price and target and price > 0:
        verdict += f" Analyst targets imply roughly {(target - price) / price * 100:+.0f}% vs the current price."
    if div and div > 0:
        verdict += f" Dividend yield ~{div * 100:.1f}%."

    body = f"{name} " + ("; ".join(parts) if parts else "has limited fundamental data") + "."
    return body + " " + verdict
