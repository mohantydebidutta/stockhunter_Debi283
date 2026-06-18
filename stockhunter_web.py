"""
============================================================
  StockHunter WEB - NSE Stock Analysis (mobile-friendly)
============================================================
  Made for: Debii

  This is the WEB version of StockHunter. It runs in any
  browser - phone, tablet, or PC. Host it free on Streamlit
  Cloud and use it from anywhere.

  RUN LOCALLY (to test):
      pip install streamlit yfinance pandas numpy
      streamlit run stockhunter_web.py

  DEPLOY FREE (to use on phone):
      See the deployment guide (DEPLOY_GUIDE.txt).
============================================================
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ---------------- stock universe ----------------
try:
    from nifty500_list import NIFTY500
    try:
        from nifty500_list import NIFTY1000
    except ImportError:
        NIFTY1000 = NIFTY500
except ImportError:
    NIFTY500 = []; NIFTY1000 = []

SECTORS = {
    "IT": ["TCS","INFY","WIPRO","HCLTECH","TECHM","LTIM","PERSISTENT","COFORGE"],
    "Pvt Bank": ["HDFCBANK","ICICIBANK","AXISBANK","KOTAKBANK","INDUSINDBK","IDFCFIRSTB"],
    "PSU Bank": ["SBIN","BANKBARODA","PNB","CANBK","UNIONBANK","INDIANB"],
    "Auto": ["TATAMOTORS","M&M","MARUTI","BAJAJ-AUTO","EICHERMOT","HEROMOTOCO","TVSMOTOR"],
    "Energy": ["RELIANCE","ONGC","IOC","BPCL","GAIL","NTPC","POWERGRID","COALINDIA"],
    "FMCG": ["HINDUNILVR","ITC","NESTLEIND","BRITANNIA","DABUR","MARICO","TATACONSUM"],
    "Pharma": ["SUNPHARMA","DRREDDY","CIPLA","DIVISLAB","AUROPHARMA","LUPIN","TORNTPHARM"],
    "Metals": ["TATASTEEL","JSWSTEEL","HINDALCO","VEDL","SAIL","NMDC","JINDALSTEL"],
    "Cement": ["ULTRACEMCO","SHREECEM","AMBUJACEM","ACC","DALBHARAT"],
    "NBFC/Fin": ["BAJFINANCE","BAJAJFINSV","CHOLAFIN","SHRIRAMFIN","MUTHOOTFIN"],
}


# ---------------- analysis engine (same logic as desktop) ----------------
def find_peers(symbol, limit=5):
    symbol = symbol.upper()
    for stocks in SECTORS.values():
        if symbol in stocks:
            return [s for s in stocks if s != symbol][:limit]
    return []


@st.cache_data(ttl=300)  # cache 5 min so it's fast
def fetch_stock(symbol):
    symbol = symbol.strip().upper().replace(".NS","").replace(".BO","")
    ticker = yf.Ticker(symbol + ".NS")
    hist = ticker.history(period="1y")
    if hist.empty:
        ticker = yf.Ticker(symbol + ".BO")
        hist = ticker.history(period="1y")
        if hist.empty:
            raise ValueError(f"'{symbol}' ka data nahi mila. Symbol check karo.")
    try: info = ticker.info
    except Exception: info = {}
    return symbol, hist, info


def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - (100/(1 + gain/loss))


def calc_atr(hist, period=14):
    h,l,c = hist["High"],hist["Low"],hist["Close"]
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]


def analyze(symbol, hist, info):
    close,high,low,vol = hist["Close"],hist["High"],hist["Low"],hist["Volume"]
    ltp = close.iloc[-1]; prev = close.iloc[-2]
    change_pct = (ltp-prev)/prev*100
    dma20 = close.rolling(20).mean().iloc[-1]
    dma50 = close.rolling(50).mean().iloc[-1]
    dma200 = close.rolling(200).mean().iloc[-1]
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

    # weekly trend
    try:
        wk = close.resample("W").last().dropna()
        wk_ema10 = wk.ewm(span=10).mean().iloc[-1]
        weekly_up = wk.iloc[-1] > wk_ema10 and wk_ema10 >= wk.ewm(span=10).mean().iloc[-2]
        weekly_txt = "Weekly trend UP (strong)" if weekly_up else "Weekly trend DOWN/flat (weak)"
    except Exception:
        weekly_up=None; weekly_txt="Weekly trend: not enough data"

    vol_breakout = vol_ratio>=1.8 and change_pct>0

    ext_from_dma20 = (ltp-dma20)/dma20*100
    over_extended = rsi>72 or from_high>-2 or ext_from_dma20>12
    near_entry = abs(ext_from_dma20)<=4

    candidates=[]
    if dma20<ltp: candidates.append(("20 DMA (small dip)",dma20))
    if support<ltp: candidates.append(("Support zone",support))
    if dma50<ltp: candidates.append(("50 DMA (big dip)",dma50))
    candidates.sort(key=lambda x:-x[1])
    best_buy=candidates[0] if candidates else None

    if over_extended: timing="OVER-EXTENDED - don't chase, wait for a pullback"; tcolor="warn"
    elif near_entry and confidence>=55: timing="GOOD ENTRY ZONE - can enter now"; tcolor="good"
    elif confidence>=55: timing="A bit extended - better on a dip"; tcolor="neutral"
    else: timing="Setup weak - no entry"; tcolor="bad"

    if weekly_up is True and confidence<100: confidence=min(100,confidence+5)
    elif weekly_up is False: confidence=max(0,confidence-5)

    if confidence>=70: verdict="STRONG BUY zone"; vcolor="good"; hold="2-8 weeks"
    elif confidence>=55: verdict="BUY on dips"; vcolor="good"; hold="2-4 weeks"
    elif confidence>=40: verdict="NEUTRAL - wait"; vcolor="warn"; hold="Watch"
    else: verdict="AVOID"; vcolor="bad"; hold="Stay out"

    stop_loss = min(ltp-1.5*atr, support*0.985)
    risk = ltp-stop_loss
    target1 = ltp+risk*1.5; target2 = ltp+risk*3.0

    def g(key,fmt=None,mult=1):
        v=info.get(key)
        if v is None: return "N/A"
        try:
            v=v*mult; return fmt.format(v) if fmt else str(v)
        except Exception: return str(v)

    return {
        "symbol":symbol,"ltp":ltp,"change_pct":change_pct,"dma20":dma20,"dma50":dma50,"dma200":dma200,
        "high_52":high_52,"low_52":low_52,"from_high":from_high,"from_low":from_low,"rsi":rsi,"atr":atr,
        "vol_ratio":vol_ratio,"support":support,"resistance":resistance,"confidence":confidence,
        "verdict":verdict,"vcolor":vcolor,"hold":hold,"signals":signals,
        "weekly_up":weekly_up,"weekly_txt":weekly_txt,"vol_breakout":vol_breakout,
        "timing":timing,"tcolor":tcolor,"over_extended":over_extended,"near_entry":near_entry,
        "ext_from_dma20":ext_from_dma20,"best_buy":best_buy,"all_pullbacks":candidates,
        "entry":ltp,"stop_loss":stop_loss,"sl_pct":(stop_loss-ltp)/ltp*100,
        "target1":target1,"t1_pct":(target1-ltp)/ltp*100,
        "target2":target2,"t2_pct":(target2-ltp)/ltp*100,
        "company":info.get("longName",symbol),"sector":info.get("sector","N/A"),
        "mcap":g("marketCap","{:,.0f}",1/1e7),"pe":g("trailingPE","{:.1f}"),
        "pb":g("priceToBook","{:.1f}"),"roe":g("returnOnEquity","{:.1f}",100),
        "de":g("debtToEquity","{:.1f}"),"hist":hist,
    }


@st.cache_data(ttl=300)
def quick_analyze(symbol):
    try:
        sym,hist,info = fetch_stock(symbol)
        d = analyze(sym,hist,info)
        return {"symbol":sym,"ltp":d["ltp"],"change_pct":d["change_pct"],
                "confidence":d["confidence"],"verdict":d["verdict"],"vcolor":d["vcolor"],
                "rsi":d["rsi"],"vol_breakout":d["vol_breakout"],"weekly_up":d["weekly_up"],
                "over_extended":d["over_extended"]}
    except Exception:
        return None


def is_perfect_buy(d):
    passed=0
    if d["confidence"]>=70: passed+=1
    if d["ltp"]>d["dma20"] and d["ltp"]>d["dma50"] and d["dma50"]>d["dma200"]: passed+=1
    if d["weekly_up"] is True: passed+=1
    if not d["over_extended"]: passed+=1
    if 45<=d["rsi"]<=68: passed+=1
    if abs(d["sl_pct"])<=8: passed+=1
    risk_ok = abs(d["sl_pct"])<=8
    return (passed>=5 and not d["over_extended"] and risk_ok), passed


@st.cache_data(ttl=300)
def get_market_mood():
    try:
        h = yf.Ticker("^NSEI").history(period="3mo")
        if h.empty: return None
        c=h["Close"]; ltp=c.iloc[-1]; prev=c.iloc[-2]
        chg=(ltp-prev)/prev*100
        d20=c.rolling(20).mean().iloc[-1]; d50=c.rolling(50).mean().iloc[-1]
        score=sum([ltp>d20,ltp>d50,d20>d50,chg>0])
        if score>=3: mood="BULLISH"; emoji="🟢"
        elif score>=2: mood="NEUTRAL"; emoji="🟡"
        else: mood="BEARISH"; emoji="🔴"
        return {"ltp":ltp,"chg":chg,"mood":mood,"emoji":emoji}
    except Exception:
        return None


# ---------------- UI ----------------
st.set_page_config(page_title="StockHunter", page_icon="📈", layout="wide")

# header
col1,col2 = st.columns([2,1])
with col1:
    st.markdown("## 📈 StockHunter")
    st.caption("NSE Analysis - swing trading")
with col2:
    mood=get_market_mood()
    if mood:
        st.metric(f"NIFTY {mood['emoji']} {mood['mood']}", f"{mood['ltp']:,.0f}", f"{mood['chg']:+.1f}%")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analyze", "⭐ Perfect Buy", "📊 Market Scan", "💎 Buy the Dip"])

# ---- Analyze tab ----
with tab1:
    sym = st.text_input("Enter stock symbol", value="TCS", key="analyze_sym").strip().upper()
    if st.button("Analyze", type="primary"):
        if sym:
            try:
                with st.spinner(f"Analyzing {sym}..."):
                    s,hist,info = fetch_stock(sym)
                    d = analyze(s,hist,info)
                colA,colB,colC = st.columns(3)
                colA.metric(d["company"][:20], f"Rs {d['ltp']:,.1f}", f"{d['change_pct']:+.1f}%")
                colB.metric("Confidence", f"{d['confidence']:.0f}%", d["verdict"])
                colC.metric("Hold time", d["hold"])

                ok,passed = is_perfect_buy(d)
                if ok:
                    st.success(f"⭐ PERFECT BUY NOW ({passed}/6 checks) - good time to buy at current levels!")
                else:
                    st.warning(f"Not a perfect buy right now ({passed}/6 checks). {d['timing']}")

                st.subheader("Trade Plan")
                t1,t2,t3,t4 = st.columns(4)
                t1.metric("Entry", f"Rs {d['entry']:,.1f}")
                t2.metric("Stop-Loss", f"Rs {d['stop_loss']:,.1f}", f"{d['sl_pct']:.1f}%")
                t3.metric("Target 1", f"Rs {d['target1']:,.1f}", f"+{d['t1_pct']:.1f}%")
                t4.metric("Target 2", f"Rs {d['target2']:,.1f}", f"+{d['t2_pct']:.1f}%")

                if d["over_extended"] and d["best_buy"]:
                    st.info(f"Stock is stretched. Best buy on pullback: **Rs {d['best_buy'][1]:,.1f}** ({d['best_buy'][0]})")

                st.subheader("Price Chart")
                chart_df = d["hist"].tail(90)[["Close"]].copy()
                chart_df["20 DMA"]=d["hist"]["Close"].rolling(20).mean().tail(90)
                chart_df["50 DMA"]=d["hist"]["Close"].rolling(50).mean().tail(90)
                st.line_chart(chart_df)

                st.subheader("Key Levels")
                k1,k2,k3 = st.columns(3)
                k1.write(f"**20 DMA:** Rs {d['dma20']:,.1f}")
                k1.write(f"**50 DMA:** Rs {d['dma50']:,.1f}")
                k1.write(f"**200 DMA:** Rs {d['dma200']:,.1f}")
                k2.write(f"**Support:** Rs {d['support']:,.1f}")
                k2.write(f"**Resistance:** Rs {d['resistance']:,.1f}")
                k2.write(f"**RSI:** {d['rsi']:.0f}")
                k3.write(f"**52W High:** Rs {d['high_52']:,.1f}")
                k3.write(f"**52W Low:** Rs {d['low_52']:,.1f}")
                k3.write(f"**{d['weekly_txt']}**")

                st.subheader("Fundamentals")
                f1,f2,f3 = st.columns(3)
                f1.write(f"**Market Cap (Cr):** {d['mcap']}")
                f1.write(f"**P/E:** {d['pe']}")
                f2.write(f"**P/B:** {d['pb']}")
                f2.write(f"**ROE %:** {d['roe']}")
                f3.write(f"**Debt/Equity:** {d['de']}")
                f3.write(f"**Sector:** {d['sector']}")

                st.subheader("Signals")
                for txt,kind in d["signals"]:
                    icon={"good":"✅","bad":"❌","warn":"⚠️","neutral":"➖"}[kind]
                    st.write(f"{icon} {txt}")

                peers=find_peers(s)
                if peers:
                    st.subheader("Similar Stocks (peers)")
                    prows=[]
                    for p in peers:
                        pr=quick_analyze(p)
                        if pr: prows.append({"Stock":pr["symbol"],"Price":f"Rs {pr['ltp']:,.1f}",
                                             "Day%":f"{pr['change_pct']:+.1f}%","Conf%":f"{pr['confidence']:.0f}%",
                                             "Signal":pr["verdict"]})
                    if prows: st.dataframe(pd.DataFrame(prows),hide_index=True,use_container_width=True)

                st.caption("This is analysis, not advice. Always use a stop-loss.")
            except Exception as e:
                st.error(f"Error: {e}")

# ---- Perfect Buy tab ----
with tab2:
    st.write("Finds stocks **perfect to buy right now** - strong, trending, in entry zone, price above Rs 80.")
    n_scan = st.slider("How many stocks to scan (more = slower)", 50, len(NIFTY1000) if NIFTY1000 else 300, 200, 50)
    if st.button("Find Perfect Buys", type="primary"):
        if not NIFTY1000:
            st.error("Stock list not found.")
        else:
            prog = st.progress(0, "Scanning...")
            perfects=[]
            syms=NIFTY1000[:n_scan]
            for i,sym in enumerate(syms,1):
                try:
                    s,hist,info = fetch_stock(sym)
                    d=analyze(s,hist,info)
                    if d["ltp"]<80: 
                        prog.progress(i/len(syms), f"Scanning {i}/{len(syms)}...")
                        continue
                    ok,passed = is_perfect_buy(d)
                    if ok:
                        perfects.append({"Stock":s,"Price":f"Rs {d['ltp']:,.1f}","Score":f"{passed}/6",
                                         "Conf%":f"{d['confidence']:.0f}%","Entry":f"Rs {d['entry']:,.0f}",
                                         "Stop-Loss":f"Rs {d['stop_loss']:,.0f} ({d['sl_pct']:.0f}%)",
                                         "Target":f"Rs {d['target1']:,.0f} (+{d['t1_pct']:.0f}%)"})
                except Exception:
                    pass
                prog.progress(i/len(syms), f"Scanning {i}/{len(syms)}...")
            prog.empty()
            if perfects:
                st.success(f"Found {len(perfects)} perfect buys!")
                st.dataframe(pd.DataFrame(perfects),hide_index=True,use_container_width=True)
                st.caption("Type any stock in the Analyze tab for full details. Always use the stop-loss.")
            else:
                st.warning("No perfect buys right now - markets may be stretched. Try later or check Buy the Dip.")

# ---- Market Scan tab ----
with tab3:
    st.write("Scans top stocks and ranks by confidence.")
    n_scan2 = st.slider("How many to scan", 50, len(NIFTY1000) if NIFTY1000 else 300, 100, 50, key="ms")
    if st.button("Run Market Scan", type="primary"):
        if not NIFTY1000:
            st.error("Stock list not found.")
        else:
            prog = st.progress(0,"Scanning...")
            results=[]
            syms=NIFTY1000[:n_scan2]
            for i,sym in enumerate(syms,1):
                r=quick_analyze(sym)
                if r: results.append(r)
                prog.progress(i/len(syms), f"Scanning {i}/{len(syms)}...")
            prog.empty()
            results.sort(key=lambda x:x["confidence"],reverse=True)
            rows=[{"Stock":r["symbol"],"Price":f"Rs {r['ltp']:,.1f}","Day%":f"{r['change_pct']:+.1f}%",
                   "RSI":f"{r['rsi']:.0f}","Conf%":f"{r['confidence']:.0f}%","Signal":r["verdict"]}
                  for r in results[:30]]
            st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)

# ---- Buy the Dip tab ----
with tab4:
    st.write("Strong stocks that are currently in a pullback (quality at a discount).")
    n_scan3 = st.slider("How many to scan", 50, len(NIFTY1000) if NIFTY1000 else 200, 200, 50, key="dip")
    if st.button("Find Dips", type="primary"):
        if not NIFTY1000:
            st.error("Stock list not found.")
        else:
            prog=st.progress(0,"Scanning...")
            dips=[]
            syms=NIFTY1000[:n_scan3]
            for i,sym in enumerate(syms,1):
                try:
                    s,hist,info=fetch_stock(sym)
                    d=analyze(s,hist,info)
                    uptrend=d["ltp"]>d["dma50"] and d["dma50"]>d["dma200"]
                    pullback=35<=d["rsi"]<=50
                    near=abs((d["ltp"]-d["dma20"])/d["dma20"]*100)<=6
                    if uptrend and pullback and near and d["ltp"]>d["dma200"]:
                        dips.append({"Stock":s,"Price":f"Rs {d['ltp']:,.1f}","RSI":f"{d['rsi']:.0f}",
                                     "Conf%":f"{d['confidence']:.0f}%","From 52WH":f"{d['from_high']:.0f}%"})
                except Exception:
                    pass
                prog.progress(i/len(syms), f"Scanning {i}/{len(syms)}...")
            prog.empty()
            if dips:
                st.success(f"Found {len(dips)} dip setups!")
                st.dataframe(pd.DataFrame(dips),hide_index=True,use_container_width=True)
            else:
                st.warning("No clean dip setups right now.")
