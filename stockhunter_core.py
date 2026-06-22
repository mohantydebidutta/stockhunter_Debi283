"""
============================================================
  stockhunter_core.py  -  shared analysis engine
  Used by BOTH the desktop (Tkinter) app and the web (Streamlit) app.
  Pure logic only - no UI. Same numbers everywhere.
============================================================
"""

import os, json
import urllib.request, json as _json
import time
import csv, datetime as _dt

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
except ImportError:
    raise SystemExit("\n[!] Install karo:  pip install yfinance pandas numpy\n")

# ---- symbol lists (optional companion files; empty if missing) ----
try:
    from nifty500_list import NIFTY500
    try:
        from nifty500_list import NIFTY1000
    except ImportError:
        NIFTY1000 = NIFTY500
except ImportError:
    NIFTY500 = []; NIFTY1000 = []
try:
    from us_stocks_list import US_STOCKS
except ImportError:
    US_STOCKS = []
try:
    from forex_list import FOREX_INSTRUMENTS
except ImportError:
    FOREX_INSTRUMENTS = []
try:
    from mf_list import MUTUAL_FUNDS
except ImportError:
    MUTUAL_FUNDS = []


def set_market(market):
    """Switch India/US. Updates the module globals the analysis fns read."""
    global MARKET, CUR
    MARKET = "US" if market == "US" else "India"
    CUR = "$" if MARKET == "US" else "Rs"
    return MARKET, CUR


# ---- Global market state (India / US). Toggled by the header button. ----
MARKET = "India"          # "India" or "US"
CUR = "Rs"                # currency label, switches to "$" for US

def active_list():
    return US_STOCKS if MARKET=="US" else NIFTY1000

def min_price():
    return 10 if MARKET=="US" else 80

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.patches import Rectangle
    HAS_MPL = True
    def plt_rect(x,y,w,h,color):
        return Rectangle((x,y),w,h,facecolor=color,edgecolor=color)
except ImportError:
    HAS_MPL = False


# ----------------------------------------------------------------------
#  PEER / SECTOR GROUPS
# ----------------------------------------------------------------------
SECTORS = {
    "IT":        ["TCS","INFY","WIPRO","HCLTECH","TECHM","LTIM","PERSISTENT","COFORGE"],
    "Pvt Bank":  ["HDFCBANK","ICICIBANK","AXISBANK","KOTAKBANK","INDUSINDBK","IDFCFIRSTB"],
    "PSU Bank":  ["SBIN","BANKBARODA","PNB","CANBK","UNIONBANK","INDIANB"],
    "Auto":      ["TATAMOTORS","M&M","MARUTI","BAJAJ-AUTO","EICHERMOT","HEROMOTOCO","TVSMOTOR"],
    "Energy":    ["RELIANCE","ONGC","IOC","BPCL","GAIL","NTPC","POWERGRID","COALINDIA"],
    "FMCG":      ["HINDUNILVR","ITC","NESTLEIND","BRITANNIA","DABUR","MARICO","TATACONSUM"],
    "Pharma":    ["SUNPHARMA","DRREDDY","CIPLA","DIVISLAB","AUROPHARMA","LUPIN","TORNTPHARM"],
    "Metals":    ["TATASTEEL","JSWSTEEL","HINDALCO","VEDL","SAIL","NMDC","JINDALSTEL"],
    "Cement":    ["ULTRACEMCO","SHREECEM","AMBUJACEM","ACC","DALBHARAT"],
    "NBFC/Fin":  ["BAJFINANCE","BAJAJFINSV","CHOLAFIN","SHRIRAMFIN","MUTHOOTFIN"],
    "Telecom":   ["BHARTIARTL","IDEA","INDUS"],
    "Infra/Cap": ["LT","SIEMENS","ABB","BHEL","BEL","CUMMINSIND"],
    "Adani":     ["ADANIENT","ADANIPORTS","ADANIPOWER","ADANIGREEN"],
}


def find_peers(symbol, limit=5):
    symbol = symbol.upper()
    for stocks in SECTORS.values():
        if symbol in stocks:
            return [s for s in stocks if s != symbol][:limit]
    return []


# ----------------------------------------------------------------------
#  ENGINE
# ----------------------------------------------------------------------
def fetch_stock(symbol):
    symbol = symbol.strip().upper().replace(".NS","").replace(".BO","")
    def _hist(tk_sym):
        last_err=None
        for attempt in range(2):   # one retry on transient errors
            try:
                t=yf.Ticker(tk_sym); h=t.history(period="1y")
                return t,h
            except Exception as e:
                last_err=e
                if "Too Many Requests" in str(e) or "rate" in str(e).lower():
                    raise ValueError("Yahoo is busy (too many requests). Wait 1-2 min and try again.")
                time.sleep(0.6)
        if last_err: raise last_err
        return None,None
    if MARKET == "US":
        ticker,hist=_hist(symbol)
        if hist is None or hist.empty:
            raise ValueError(f"'{symbol}' not found. Check the symbol (US stocks use plain tickers like AAPL).")
    else:
        ticker,hist=_hist(symbol + ".NS")
        if hist is None or hist.empty:
            ticker,hist=_hist(symbol + ".BO")
            if hist is None or hist.empty:
                raise ValueError(f"'{symbol}' ka data nahi mila. Symbol sahi likho (jaise RELIANCE, TCS).")
    try: info = ticker.info
    except Exception: info = {}
    return symbol, ticker, hist, info


def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - (100/(1 + gain/loss))


def calc_atr(hist, period=14):
    h,l,c = hist["High"],hist["Low"],hist["Close"]
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]


def _fmt_fx(v):
    """Format forex/crypto price - more decimals for small values."""
    if v is None: return "-"
    if v >= 1000: return f"{v:,.1f}"
    if v >= 1: return f"{v:,.3f}"
    return f"{v:,.5f}"


def analyze_forex(yahoo_sym):
    """Forex/commodity/crypto: trend + RSI + BUY/SELL verdict + SL/TP. No fundamentals."""
    try:
        h = yf.Ticker(yahoo_sym).history(period="6mo")
        if h.empty or len(h)<60: return None
        c = h["Close"]
        ltp = c.iloc[-1]; prev = c.iloc[-2]
        chg = (ltp-prev)/prev*100
        ema20 = c.ewm(span=20).mean().iloc[-1]
        ema50 = c.ewm(span=50).mean().iloc[-1]
        rsi = calc_rsi(c).iloc[-1]
        atr = calc_atr(h)
        score = 0
        if ltp>ema20: score+=1
        if ltp>ema50: score+=1
        if ema20>ema50: score+=1
        if chg>0: score+=1
        side=None
        if score>=3 and rsi<70:
            verdict="BUY"; vcolor="good"; side="buy"
        elif score<=1 and rsi>30:
            verdict="SELL"; vcolor="bad"; side="sell"
        elif rsi>=70:
            verdict="OVERBOUGHT"; vcolor="warn"
        elif rsi<=30:
            verdict="OVERSOLD"; vcolor="warn"
        else:
            verdict="NEUTRAL"; vcolor="neutral"
        trend = "UP" if ema20>ema50 and ltp>ema20 else ("DOWN" if ema20<ema50 and ltp<ema20 else "SIDEWAYS")
        sl=tp=None; sl_pct=tp_pct=None
        if side=="buy":
            sl = ltp - 1.5*atr; tp = ltp + 3.0*atr
        elif side=="sell":
            sl = ltp + 1.5*atr; tp = ltp - 3.0*atr
        if sl is not None:
            sl_pct=(sl-ltp)/ltp*100; tp_pct=(tp-ltp)/ltp*100
        # support / resistance from recent 60 bars
        recent=c.tail(60)
        support=recent.min(); resistance=recent.max()
        # 52w-style range (6mo here)
        wk_high=c.max(); wk_low=c.min()
        # weekly change (approx 5 bars)
        wk_chg=(ltp-c.iloc[-6])/c.iloc[-6]*100 if len(c)>6 else 0
        return {"ltp":ltp,"chg":chg,"rsi":rsi,"trend":trend,"verdict":verdict,"vcolor":vcolor,
                "side":side,"sl":sl,"tp":tp,"sl_pct":sl_pct,"tp_pct":tp_pct,"atr":atr,
                "hist":h,"ema20":ema20,"ema50":ema50,"support":support,"resistance":resistance,
                "wk_high":wk_high,"wk_low":wk_low,"wk_chg":wk_chg,"score":score}
    except Exception:
        return None


# Timeframe presets: label -> (yahoo period, yahoo interval, typical holding text, atr multiple for SL)
FX_TIMEFRAMES = {
    "Intraday": ("5d",   "15m", "a few hours to 1 day", 1.2),
    "Swing (days)": ("1mo",  "1h",  "2 - 5 days",          1.5),
    "Short (weeks)": ("3mo", "1d",  "1 - 3 weeks",          2.0),
    "Position (months)": ("1y", "1d","1 - 3 months",        2.5),
    "Long (1Y+)": ("2y",  "1wk", "3 months or more",        3.0),
}

def forex_timeframe_signal(yahoo_sym, tf_label):
    """Analyze one timeframe: BUY/SELL + SL/TP + holding period (trend + ATR based)."""
    try:
        period,interval,hold,atr_mult = FX_TIMEFRAMES[tf_label]
        h=yf.Ticker(yahoo_sym).history(period=period,interval=interval)
        if h.empty or len(h)<25: return None
        c=h["Close"]
        ltp=c.iloc[-1]; prev=c.iloc[-2]
        chg=(ltp-prev)/prev*100
        ema9=c.ewm(span=9).mean().iloc[-1]
        ema21=c.ewm(span=21).mean().iloc[-1]
        rsi=calc_rsi(c).iloc[-1]
        atr=calc_atr(h)
        score=0
        if ltp>ema9: score+=1
        if ltp>ema21: score+=1
        if ema9>ema21: score+=1
        if chg>0: score+=1
        # how far is price from the slow EMA? tiny moves = noise, not a real signal
        dist_pct = abs(ltp-ema21)/ema21*100 if ema21 else 0
        side=None
        # require a real score AND a minimum move away from EMA so we don't
        # call BUY/SELL on flat, noisy intraday candles (e.g. 0.1% wiggles).
        strong = dist_pct >= 0.15
        if score>=3 and rsi<72 and strong:
            verdict="BUY"; vcolor="good"; side="buy"
        elif score<=1 and rsi>28 and strong:
            verdict="SELL"; vcolor="bad"; side="sell"
        elif rsi>=72:
            verdict="WAIT (overbought)"; vcolor="warn"
        elif rsi<=28:
            verdict="WAIT (oversold)"; vcolor="warn"
        else:
            verdict="NEUTRAL - no trade"; vcolor="neutral"
        sl=tp=sl_pct=tp_pct=None
        if side=="buy":
            sl=ltp-atr_mult*atr; tp=ltp+atr_mult*2*atr
        elif side=="sell":
            sl=ltp+atr_mult*atr; tp=ltp-atr_mult*2*atr
        if sl is not None:
            sl_pct=(sl-ltp)/ltp*100; tp_pct=(tp-ltp)/ltp*100
        return {"tf":tf_label,"ltp":ltp,"chg":chg,"rsi":rsi,"verdict":verdict,"vcolor":vcolor,
                "side":side,"sl":sl,"tp":tp,"sl_pct":sl_pct,"tp_pct":tp_pct,"hold":hold}
    except Exception:
        return None


# Stock timeframe presets: label -> (period, interval, holding text, ATR multiple for SL, target multiple)
STOCK_TIMEFRAMES = {
    "Swing (days)":      ("3mo", "1d",  "3 - 10 days",      1.5, 2.0),
    "Short (weeks)":     ("6mo", "1d",  "2 - 6 weeks",       2.0, 3.0),
    "Position (months)": ("1y",  "1d",  "1 - 4 months",      2.5, 4.0),
    "Long (1Y+)":        ("2y",  "1wk", "6 months to years", 3.0, 5.0),
}

def stock_timeframe_signal(symbol, market, tf_label):
    """Per-timeframe stock signal: BUY/SELL/HOLD + entry/SL/targets + holding period."""
    try:
        period,interval,hold,sl_mult,tp_mult = STOCK_TIMEFRAMES[tf_label]
        sym=symbol.strip().upper().replace(".NS","").replace(".BO","")
        if market=="US":
            t=yf.Ticker(sym)
        else:
            t=yf.Ticker(sym+".NS")
        h=t.history(period=period,interval=interval)
        if (h is None or h.empty or len(h)<25) and market!="US":
            t=yf.Ticker(sym+".BO"); h=t.history(period=period,interval=interval)
        if h is None or h.empty or len(h)<25: return None
        c=h["Close"]
        ltp=c.iloc[-1]; prev=c.iloc[-2]
        chg=(ltp-prev)/prev*100
        ema20=c.ewm(span=20).mean().iloc[-1]
        ema50=c.ewm(span=50).mean().iloc[-1]
        rsi=calc_rsi(c).iloc[-1]
        atr=calc_atr(h)
        score=0
        if ltp>ema20: score+=1
        if ltp>ema50: score+=1
        if ema20>ema50: score+=1
        if chg>0: score+=1
        side=None
        if score>=3 and rsi<72:
            verdict="BUY"; vcolor="good"; side="buy"
        elif score<=1:
            verdict="AVOID / weak"; vcolor="bad"; side=None
        elif rsi>=72:
            verdict="WAIT (overbought)"; vcolor="warn"
        else:
            verdict="HOLD / neutral"; vcolor="neutral"
        sl=t1=t2=None; sl_pct=t1_pct=t2_pct=None
        if side=="buy":
            sl=ltp-sl_mult*atr
            t1=ltp+tp_mult*atr
            t2=ltp+tp_mult*1.8*atr
            sl_pct=(sl-ltp)/ltp*100; t1_pct=(t1-ltp)/ltp*100; t2_pct=(t2-ltp)/ltp*100
        return {"tf":tf_label,"ltp":ltp,"chg":chg,"rsi":rsi,"verdict":verdict,"vcolor":vcolor,
                "side":side,"sl":sl,"t1":t1,"t2":t2,"sl_pct":sl_pct,"t1_pct":t1_pct,"t2_pct":t2_pct,
                "hold":hold}
    except Exception:
        return None


def analyze_mf(scheme_code):
    """Mutual fund analysis via mfapi.in (free AMFI NAV data).
    Returns NAV, returns (1Y/3Y/5Y annualized), trend, and a recommendation."""
    try:
        url=f"https://api.mfapi.in/mf/{scheme_code}"
        req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"})
        with urllib.request.urlopen(req,timeout=12) as r:
            data=_json.loads(r.read().decode())
        navs=data.get("data",[])
        if not navs or len(navs)<30: return None
        latest=float(navs[0]["nav"])   # navs are newest-first
        def nav_days_ago(days):
            idx=min(days, len(navs)-1)
            try: return float(navs[idx]["nav"])
            except Exception: return None
        def ret(days):
            old=nav_days_ago(days)
            if old and old>0: return (latest-old)/old*100
            return None
        r1y=ret(250); r3y=ret(750); r5y=ret(1250)
        r3y_ann=((1+r3y/100)**(1/3)-1)*100 if r3y is not None else None
        r5y_ann=((1+r5y/100)**(1/5)-1)*100 if r5y is not None else None
        recent=[float(n["nav"]) for n in navs[:50] if n.get("nav")]
        avg50=sum(recent)/len(recent) if recent else latest
        trend="UP" if latest>avg50 else "DOWN"
        if r1y is None:
            verdict="Not enough data"; vcolor="neutral"
        elif r1y>=15 and trend=="UP":
            verdict="STRONG - good momentum"; vcolor="good"
        elif r1y>=8:
            verdict="DECENT - steady"; vcolor="good"
        elif r1y>=0:
            verdict="AVERAGE - watch"; vcolor="warn"
        else:
            verdict="WEAK - underperforming"; vcolor="bad"
        # expense ratio estimate + description (mfapi doesn't give exact ER, so estimate by type)
        scheme_name=data.get("meta",{}).get("scheme_name","")
        sn=scheme_name.lower()
        is_direct="direct" in sn
        if "index" in sn or "etf" in sn:
            er = "0.1% - 0.4%" if is_direct else "0.2% - 1.0%"
            er_note="Index/ETF funds are the cheapest (they just track an index)."
        elif is_direct:
            er = "0.4% - 1.2%"
            er_note="Direct plan - lower cost than Regular (no distributor commission)."
        else:
            er = "1.0% - 2.2%"
            er_note="Regular plan - includes distributor commission. Direct plan would be cheaper."
        plan_type="Direct" if is_direct else ("Regular" if "regular" in sn else "—")
        desc=(f"This fund's NAV is the per-unit price. Expense ratio is the yearly fee the fund "
              f"charges (already reflected in NAV). Lower expense ratio = more of the returns stay with you. "
              f"{er_note}")
        return {"nav":latest,"r1y":r1y,"r3y":r3y_ann,"r5y":r5y_ann,
                "trend":trend,"verdict":verdict,"vcolor":vcolor,
                "fund_house":data.get("meta",{}).get("fund_house",""),
                "navs":navs,"scheme":scheme_name,
                "expense_est":er,"plan_type":plan_type,"desc":desc}
    except Exception:
        return None


def mf_recommendation(mf):
    """Turn a fund's returns + trend into a plain BUY-style call for SIP/lumpsum.
    Returns (action, color_key, one_line_reason)."""
    r1=mf.get("r1y"); r3=mf.get("r3y"); trend=mf.get("trend")
    if r1 is None:
        return ("NOT ENOUGH DATA","neutral","Too little history to judge this fund.")
    # strong long-term compounder, still trending up
    if r1>=15 and (r3 is None or r3>=12) and trend=="UP":
        return ("ACCUMULATE","good",
                "Strong returns and still trending up - good for SIP; lumpsum okay on dips.")
    if r1>=10 and trend=="UP":
        return ("ACCUMULATE (SIP)","good",
                "Decent steady performer in an uptrend - best added via SIP, not all at once.")
    if r1>=8:
        return ("HOLD / SIP ONLY","warn",
                "Okay but not exciting - keep existing units, add only small SIP amounts.")
    if r1>=0 and trend=="UP":
        return ("HOLD","warn",
                "Weak returns but turning up - watch a few months before adding more.")
    if r1>=0:
        return ("HOLD / REVIEW","warn",
                "Flat returns and no clear uptrend - review against a better fund in the same category.")
    return ("AVOID / SWITCH","bad",
            "Negative 1Y return and weak trend - consider a stronger fund in the same category.")


def analyze(symbol, hist, info):
    close,high,low,vol = hist["Close"],hist["High"],hist["Low"],hist["Volume"]
    ltp = close.iloc[-1]; prev = close.iloc[-2]
    change_pct = (ltp-prev)/prev*100
    dma20 = close.rolling(20).mean().iloc[-1]
    dma50 = close.rolling(50).mean().iloc[-1]
    dma200 = close.rolling(200).mean().iloc[-1]
    # EMA - reacts faster to recent price (early momentum signal)
    ema20 = close.ewm(span=20).mean().iloc[-1]
    ema50 = close.ewm(span=50).mean().iloc[-1]
    ema200 = close.ewm(span=200).mean().iloc[-1] if len(close)>=50 else dma200
    high_52,low_52 = high.max(),low.min()
    from_high = (ltp-high_52)/high_52*100
    from_low = (ltp-low_52)/low_52*100
    rsi = calc_rsi(close).iloc[-1]
    atr = calc_atr(hist)
    avg_vol = vol.rolling(20).mean().iloc[-1]
    vol_ratio = vol.iloc[-1]/avg_vol if avg_vol>0 else 0
    recent = close.tail(20)
    support,resistance = recent.min(),recent.max()

    checks=[]
    def add(cond,w,g,b,kg="good",kb="bad"):
        checks.append((w,w,g,kg) if cond else (0,w,b,kb))
    add(ltp>dma20,15,"Price > 20 DMA (short-term up)","Price < 20 DMA (short-term weak)")
    add(ltp>dma50,15,"Price > 50 DMA (medium up)","Price < 50 DMA (medium down)")
    add(ltp>dma200,15,"Price > 200 DMA (long-term bullish)","Price < 200 DMA (long-term bearish)")
    add(dma20>dma50,15,"20 DMA > 50 DMA (momentum up)","20 DMA < 50 DMA (momentum down)")
    # EMA signals - faster/earlier than DMA (react sooner to a turn)
    add(ema20>ema50,10,"EMA20 > EMA50 (early momentum up)","EMA20 < EMA50 (early momentum down)")
    add(ltp>ema20,8,"Price > EMA20 (short-term EMA support)","Price < EMA20 (below short EMA)")
    if 45<=rsi<=65: checks.append((15,15,f"RSI {rsi:.0f} - strong healthy zone","good"))
    elif 40<=rsi<45 or 65<rsi<=70: checks.append((8,15,f"RSI {rsi:.0f} - okay","warn"))
    elif rsi>70: checks.append((3,15,f"RSI {rsi:.0f} - OVERBOUGHT (pullback risk)","warn"))
    elif rsi<30: checks.append((5,15,f"RSI {rsi:.0f} - OVERSOLD (bounce possible)","warn"))
    else: checks.append((6,15,f"RSI {rsi:.0f} - neutral","neutral"))
    if vol_ratio>1.3 and change_pct>0: checks.append((15,15,f"Volume {vol_ratio:.1f}x on up-move (strong buying)","good"))
    elif vol_ratio>1.3 and change_pct<0: checks.append((2,15,f"Volume {vol_ratio:.1f}x on down-move (selling)","bad"))
    else: checks.append((8,15,f"Volume {vol_ratio:.1f}x avg (normal)","neutral"))
    if from_high>-5: checks.append((10,10,"Near 52W high (breakout strength)","good"))
    elif from_high>-20: checks.append((6,10,f"{from_high:.0f}% below 52W high","neutral"))
    else: checks.append((2,10,f"{from_high:.0f}% below 52W high (weak)","warn"))

    got=sum(c[0] for c in checks); total=sum(c[1] for c in checks)
    confidence = got/total*100
    signals=[(t,k) for _,_,t,k in checks]

    # ---- WEEKLY TREND CONFIRMATION (daily ko weekly me resample) ----
    try:
        wk = close.resample("W").last().dropna()
        wk_ema10 = wk.ewm(span=10).mean().iloc[-1]
        wk_price = wk.iloc[-1]
        wk_prev_ema = wk.ewm(span=10).mean().iloc[-2]
        weekly_up = wk_price > wk_ema10 and wk_ema10 >= wk_prev_ema
        weekly_txt = "Weekly trend UP (strong)" if weekly_up else "Weekly trend DOWN/flat (weak)"
    except Exception:
        weekly_up = None; weekly_txt = "Weekly trend: not enough data"

    # ---- VOLUME BREAKOUT DETECTOR ----
    vol_breakout = vol_ratio >= 1.8 and change_pct > 0
    if vol_breakout:
        vol_break_txt = f"VOLUME BREAKOUT! {vol_ratio:.1f}x avg with price up - strong move"
    elif vol_ratio >= 1.8:
        vol_break_txt = f"High volume {vol_ratio:.1f}x but price down - distribution risk"
    else:
        vol_break_txt = ""

    # ---- ENTRY TIMING (chasing roko) ----
    # over-extended agar: RSI high YA 52W high ke bahut paas YA 20DMA se bahut upar
    ext_from_dma20 = (ltp - dma20)/dma20*100
    over_extended = rsi > 72 or from_high > -2 or ext_from_dma20 > 12
    near_entry = abs(ext_from_dma20) <= 4  # 20 DMA ke 4% andar = achha entry zone

    # ---- BEST BUY PRICE (pullback levels jab over-stretched ho) ----
    # 3 realistic pullback levels, jo abhi price se neeche hon
    candidates = []
    if dma20 < ltp: candidates.append(("20 DMA (small dip)", dma20))
    if support < ltp: candidates.append(("Support/breakout zone", support))
    if dma50 < ltp: candidates.append(("50 DMA (bigger dip)", dma50))
    # closest pullback (sabse upar wala neeche level = sabse pehle aayega)
    candidates.sort(key=lambda x:-x[1])
    best_buy = candidates[0] if candidates else None

    if over_extended:
        timing="OVER-EXTENDED - don't chase now, wait for a pullback"
        timing_color="warn"
    elif near_entry and confidence>=55:
        timing="GOOD ENTRY ZONE - can enter now (near 20 DMA)"
        timing_color="good"
    elif confidence>=55:
        timing="A bit extended - better to enter on a dip"
        timing_color="neutral"
    else:
        timing="Setup weak - no entry here"
        timing_color="bad"

    entry_info={"timing":timing,"timing_color":timing_color,"over_extended":over_extended,
                "near_entry":near_entry,"ext_from_dma20":ext_from_dma20,"best_buy":best_buy,
                "all_pullbacks":candidates}

    # weekly ko confidence me thoda weight (alignment bonus/penalty)
    if weekly_up is True and confidence<100: confidence=min(100,confidence+5)
    elif weekly_up is False: confidence=max(0,confidence-5)

    if confidence>=70: verdict="STRONG BUY zone"; vcolor="good"; hold="2-8 weeks (positional swing)"
    elif confidence>=55: verdict="BUY on dips / Accumulate"; vcolor="good"; hold="2-4 weeks (swing)"
    elif confidence>=40: verdict="NEUTRAL - wait"; vcolor="warn"; hold="Avoid fresh entry, watch"
    else: verdict="AVOID / Bearish"; vcolor="bad"; hold="Stay out (no setup)"

    stop_loss = min(ltp-1.5*atr, support*0.985)
    risk = ltp-stop_loss
    target1 = ltp+risk*1.5; target2 = ltp+risk*3.0
    trade_plan={"entry":ltp,"stop_loss":stop_loss,"sl_pct":(stop_loss-ltp)/ltp*100,
                "target1":target1,"t1_pct":(target1-ltp)/ltp*100,
                "target2":target2,"t2_pct":(target2-ltp)/ltp*100,
                "atr":atr,"hold":hold,"risk_per_share":risk}

    def g(key,fmt=None,mult=1):
        v=info.get(key)
        if v is None: return "N/A"
        try:
            v=v*mult; return fmt.format(v) if fmt else str(v)
        except Exception: return str(v)
    def div_yield():
        v=info.get("dividendYield")
        if v is None: return "N/A"
        try:
            v=float(v)
            # Yahoo is inconsistent: sometimes 0.0199 (fraction), sometimes 1.99 (already %).
            # A real yield is almost never above ~25%, so if v<=1 treat as fraction.
            pct = v*100 if v<=1 else v
            return f"{pct:.2f}"
        except Exception:
            return "N/A"
    fundamentals={"Company":info.get("longName",symbol),"Sector":info.get("sector","N/A"),
        "Market Cap (Cr)":g("marketCap","{:,.0f}",1/1e7),"P/E Ratio":g("trailingPE","{:.1f}"),
        "P/B Ratio":g("priceToBook","{:.1f}"),"ROE %":g("returnOnEquity","{:.1f}",100),
        "Debt/Equity":g("debtToEquity","{:.1f}"),"Div Yield %":div_yield()}

    return {"symbol":symbol,"ltp":ltp,"change_pct":change_pct,"dma20":dma20,"dma50":dma50,
        "dma200":dma200,"ema20":ema20,"ema50":ema50,"ema200":ema200,
        "high_52":high_52,"low_52":low_52,"from_high":from_high,"from_low":from_low,
        "rsi":rsi,"atr":atr,"vol_ratio":vol_ratio,"support":support,"resistance":resistance,
        "confidence":confidence,"verdict":verdict,"vcolor":vcolor,"signals":signals,
        "fundamentals":fundamentals,"trade_plan":trade_plan,"hist":hist,
        "weekly_up":weekly_up,"weekly_txt":weekly_txt,"vol_breakout":vol_breakout,
        "vol_break_txt":vol_break_txt,"entry_info":entry_info}


def quick_analyze(symbol):
    try:
        sym,_,hist,info = fetch_stock(symbol)
        d = analyze(sym,hist,info)
        return {"symbol":sym,"ltp":d["ltp"],"change_pct":d["change_pct"],
                "confidence":d["confidence"],"verdict":d["verdict"],"vcolor":d["vcolor"],
                "rsi":d["rsi"],"vol_breakout":d["vol_breakout"],"weekly_up":d["weekly_up"],
                "spark":list(hist["Close"].tail(30).values),
                "over_extended":d["entry_info"]["over_extended"]}
    except Exception:
        return None


def fetch_news(ticker, limit=5):
    try:
        news = ticker.news or []
        out=[]
        for n in news[:limit]:
            content = n.get("content",{}) or {}
            title = n.get("title") or content.get("title","")
            pub = n.get("publisher") or content.get("provider",{}).get("displayName","")
            # URL can live in several places depending on yfinance version / schema
            url = (n.get("link")
                   or content.get("canonicalUrl",{}).get("url")
                   or content.get("clickThroughUrl",{}).get("url")
                   or "")
            if title: out.append((title, pub, url))
        return out
    except Exception:
        return []


def get_market_mood():
    """Index trend - overall market bullish/bearish/neutral (NIFTY or S&P 500)."""
    try:
        sym = "^GSPC" if MARKET=="US" else "^NSEI"
        label = "S&P 500" if MARKET=="US" else "NIFTY"
        idx = yf.Ticker(sym)
        hist = idx.history(period="3mo")
        if hist.empty: return None
        close = hist["Close"]
        ltp = close.iloc[-1]; prev = close.iloc[-2]
        change_pct = (ltp-prev)/prev*100
        dma20 = close.rolling(20).mean().iloc[-1]
        dma50 = close.rolling(50).mean().iloc[-1]
        rsi = calc_rsi(close).iloc[-1]
        score = 0
        if ltp>dma20: score+=1
        if ltp>dma50: score+=1
        if dma20>dma50: score+=1
        if change_pct>0: score+=1
        if score>=3: mood="BULLISH"; color="good"; note="Good time for fresh longs"
        elif score>=2: mood="NEUTRAL"; color="warn"; note="Be selective, mixed signals"
        else: mood="BEARISH"; color="bad"; note="Caution - even good stocks may fall"
        return {"ltp":ltp,"change_pct":change_pct,"mood":mood,"color":color,
                "note":note,"rsi":rsi,"dma20":dma20,"dma50":dma50,"label":label}
    except Exception:
        return None


# NSE sector indices on Yahoo Finance (with fallback symbols)
SECTOR_INDICES = {
    "NIFTY": ["^NSEI"],
    "BANK": ["^NSEBANK"],
    "IT": ["^CNXIT"],
    "AUTO": ["^CNXAUTO"],
    "PHARMA": ["^CNXPHARMA", "^CNXPHARMA"],
    "FMCG": ["^CNXFMCG"],
    "METAL": ["^CNXMETAL"],
    "ENERGY": ["^CNXENERGY"],
    "REALTY": ["^CNXREALTY"],
    "FIN": ["NIFTY_FIN_SERVICE.NS", "^CNXFIN"],
    "PSUBANK": ["^CNXPSUBANK"],
    "MEDIA": ["^CNXMEDIA"],
}

# US sector ETFs (proxy for sector performance)
US_SECTOR_INDICES = {
    "TECH": ["XLK"], "FINANCE": ["XLF"], "HEALTH": ["XLV"], "ENERGY": ["XLE"],
    "CONSUMER": ["XLY"], "STAPLES": ["XLP"], "INDUSTRIAL": ["XLI"],
    "UTILITIES": ["XLU"], "MATERIALS": ["XLB"], "REALESTATE": ["XLRE"], "COMM": ["XLC"],
}


def get_sector_indices():
    """Sector indices ka aaj ka change% laata hai (NSE ya US ETFs). Jo na chale wo skip."""
    indices = US_SECTOR_INDICES if MARKET=="US" else SECTOR_INDICES
    out=[]
    for name, syms in indices.items():
        for sym in syms:
            try:
                t=yf.Ticker(sym)
                h=t.history(period="5d")
                if h.empty or len(h)<2: continue
                ltp=h["Close"].iloc[-1]; prev=h["Close"].iloc[-2]
                chg=(ltp-prev)/prev*100
                out.append({"name":name,"ltp":ltp,"change_pct":chg})
                break
            except Exception:
                continue
    return out


def get_sector_strength():
    """Har sector ka average confidence nikaalta hai (representative stocks se)."""
    out=[]
    for sec, stocks in SECTORS.items():
        confs=[]; changes=[]
        for sym in stocks[:4]:  # speed ke liye top 4 per sector
            r = quick_analyze(sym)
            if r:
                confs.append(r["confidence"]); changes.append(r["change_pct"])
        if confs:
            out.append({"sector":sec,"avg_conf":sum(confs)/len(confs),
                        "avg_change":sum(changes)/len(changes),"count":len(confs)})
    out.sort(key=lambda x:x["avg_conf"],reverse=True)
    return out


def is_buy_the_dip(d):
    """Strong stock jo abhi pullback me hai - uptrend intact, RSI low, near 20DMA."""
    uptrend = d["ltp"]>d["dma50"] and d["dma50"]>d["dma200"]  # long-term up
    pullback = 35 <= d["rsi"] <= 50  # not overbought, cooled off
    near_support = abs((d["ltp"]-d["dma20"])/d["dma20"]*100) <= 6  # near 20DMA
    not_falling_knife = d["ltp"] > d["dma200"]  # still above key support
    return uptrend and pullback and near_support and not_falling_knife


def is_perfect_buy(d):
    """PERFECT BUY = everything aligned AND in a good entry zone right now.
    Returns (is_perfect, score, reasons_list)."""
    reasons=[]; checks_passed=0; total_checks=6

    # 1. Strong confidence
    if d["confidence"]>=70:
        checks_passed+=1; reasons.append(("Strong setup (confidence >= 70%)",True))
    else:
        reasons.append((f"Confidence only {d['confidence']:.0f}% (need 70+)",False))

    # 2. Daily trend up (price above key DMAs)
    if d["ltp"]>d["dma20"] and d["ltp"]>d["dma50"] and d["dma50"]>d["dma200"]:
        checks_passed+=1; reasons.append(("Daily trend clearly UP",True))
    else:
        reasons.append(("Daily trend not fully aligned",False))

    # 3. Weekly trend up
    if d["weekly_up"] is True:
        checks_passed+=1; reasons.append(("Weekly trend UP",True))
    else:
        reasons.append(("Weekly trend not up",False))

    # 4. NOT over-extended (this is the key - good entry zone)
    if not d["entry_info"]["over_extended"]:
        checks_passed+=1; reasons.append(("In entry zone (not over-stretched)",True))
    else:
        reasons.append(("Over-stretched - too late to chase",False))

    # 5. RSI healthy (room to run, not overbought)
    if 45 <= d["rsi"] <= 68:
        checks_passed+=1; reasons.append((f"RSI healthy ({d['rsi']:.0f})",True))
    else:
        reasons.append((f"RSI {d['rsi']:.0f} not ideal (want 45-68)",False))

    # 6. Good risk-reward (SL not too far, target meaningful)
    tp=d["trade_plan"]
    sl_dist=abs(tp["sl_pct"])
    if sl_dist <= 8:  # SL within 8% = controlled risk
        checks_passed+=1; reasons.append((f"Controlled risk (SL {tp['sl_pct']:.1f}%)",True))
    else:
        reasons.append((f"SL too far ({tp['sl_pct']:.1f}%) - risky",False))

    # PERFECT = at least 5 of 6, AND must not be over-extended, AND risk must be controlled
    risk_ok = abs(tp["sl_pct"]) <= 8
    is_perfect = checks_passed>=5 and not d["entry_info"]["over_extended"] and risk_ok
    return is_perfect, checks_passed, total_checks, reasons


# ---- lightweight JSON persistence (watchlist, alerts, journal, settings) ----
WATCHFILE="watchlist.txt"
ALERTFILE="alerts.json"
JOURNALFILE="trade_journal.json"
WATCHJSON="watchlist.json"
SETTINGSFILE="settings.json"

def _load_json(path, default):
    """Read a JSON file; return default if missing/corrupt."""
    try:
        if os.path.exists(path):
            with open(path,"r",encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def _save_json(path, data):
    """Write data to a JSON file (best-effort, manual save only)."""
    try:
        with open(path,"w",encoding="utf-8") as f:
            json.dump(data,f,indent=2)
        return True
    except Exception:
        return False

def _now_str():
    return _dt.datetime.now().strftime("%d-%b-%Y %H:%M")
