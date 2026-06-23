"""
============================================================
  StockHunter (Web)  -  Streamlit version
  Same engine, same look as the desktop app.
  Run:  streamlit run stockhunter_web.py
  Needs: pip install streamlit yfinance pandas numpy plotly
============================================================
"""
import streamlit as st
import pandas as pd
import numpy as np
import datetime as _dt

import stockhunter_core as core
from stockhunter_core import (
    analyze, quick_analyze, fetch_stock, fetch_news, find_peers,
    analyze_forex, forex_timeframe_signal, stock_timeframe_signal,
    analyze_mf, mf_recommendation, FX_TIMEFRAMES, STOCK_TIMEFRAMES,
    get_market_mood, get_sector_indices, get_sector_strength,
    is_buy_the_dip, is_perfect_buy, active_list, min_price,
    FOREX_INSTRUMENTS, MUTUAL_FUNDS, _fmt_fx,
)

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

import yfinance as yf

# ---------- exact desktop palette ----------
BG="#0d1311"; CARD="#15201b"; CARD2="#1d2c24"
ACCENT="#10b981"; ACCENT2="#34d399"; GOLD="#d4af37"
TEXT="#eaf2ed"; MUTED="#7e9588"
GOOD="#34d399"; BAD="#f87171"; WARN="#fbbf24"; NEUTRAL="#7e9588"
BORDER="#243730"
COLORMAP={"good":GOOD,"bad":BAD,"warn":WARN,"neutral":NEUTRAL}

st.set_page_config(page_title="StockHunter - Markets, Forex & Funds",
                   page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# =========================================================
#  AI engine (Google Gemini) - shared by all AI features
# =========================================================
def ai_generate(prompt, api_key, model="gemini-2.5-flash", timeout=60):
    import urllib.request, urllib.error, json as _j
    if not api_key or not api_key.strip():
        raise RuntimeError("No API key. Add your free Gemini key in the sidebar (>).")
    url=f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key.strip()}"
    body={"contents":[{"parts":[{"text":prompt}]}],
          "generationConfig":{"temperature":0.4,"maxOutputTokens":2048}}
    data=_j.dumps(body).encode("utf-8")
    req=urllib.request.Request(url,data=data,headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req,timeout=timeout) as resp:
            out=_j.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code in (400,403): raise RuntimeError("API key invalid or not enabled. Check it in the sidebar.")
        if e.code==429: raise RuntimeError("Free limit reached for now. Wait a bit and retry.")
        raise RuntimeError(f"AI error {e.code}")
    except Exception as e:
        raise RuntimeError(f"Could not reach Gemini: {e}")
    try:
        return "".join(p.get("text","") for p in out["candidates"][0]["content"]["parts"]).strip()
    except Exception:
        raise RuntimeError("AI returned no text (possibly blocked).")

def ai_ready():
    return bool(st.session_state.get("gemini_key","").strip())

def ai_call(prompt):
    return ai_generate(prompt, st.session_state.get("gemini_key",""),
                       st.session_state.get("gemini_model","gemini-2.5-flash"))

def clean_md(s):
    import re
    s=re.sub(r"\*\*\*(.+?)\*\*\*",r"\1",s); s=re.sub(r"\*\*(.+?)\*\*",r"\1",s)
    s=re.sub(r"\*(.+?)\*",r"\1",s); s=re.sub(r"`(.+?)`",r"\1",s)
    s=re.sub(r"^#+\s*","",s,flags=re.MULTILINE)
    s=re.sub(r"^\s*[-*•]\s+","• ",s,flags=re.MULTILINE); s=re.sub(r"\*+","",s)
    return s.strip()

def render_ai_text(txt):
    """Render AI markdown text as clean section cards in Streamlit."""
    import re
    if not txt.strip():
        st.info("No response."); return
    txt="\n".join(l for l in txt.split("\n") if not (len(l.strip())>=3 and set(l.strip())<=set("-*_ ")))
    blocks=[b.strip() for b in re.split(r"\n\s*\n", txt) if b.strip()]
    for b in blocks:
        lines=b.split("\n"); raw=lines[0].strip(); first=clean_md(raw); ht=first.rstrip(":")
        looks_label=first.endswith(":") and len(ht.split())<=6
        looks_caps=ht.isupper() and len(ht)<48 and len(ht.split())<=6
        looks_md=(raw.startswith("#") or raw.startswith("**")) and len(ht.split())<=6
        if looks_label or looks_caps or looks_md:
            body=clean_md("\n".join(lines[1:])).strip()
            st.markdown(f"<div class='sh-card2'><div class='sh-accent' style='font-weight:700;font-size:1.02rem'>{ht}</div>"
                        f"<div style='margin-top:4px'>{body}</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='sh-card2'>{clean_md(b)}</div>", unsafe_allow_html=True)

# ---------- API key: OWNER-CONTROLLED, hidden from people you share with ----------
# Priority for the key (first one found wins):
#   1. st.secrets["GEMINI_API_KEY"]      (best for Streamlit Cloud - never visible to users)
#   2. owner_key.txt   on disk           (best for your own laptop)
#   3. env var GEMINI_API_KEY
# People you share the app with just USE it - they never see or change the key.
# To change it yourself, open the sidebar with your OWNER PASSWORD (see OWNER_PASSWORD below).
import os as _os

OWNER_PASSWORD = "debii123"   # <-- CHANGE THIS to your own secret. Only you should know it.
OWNER_KEYFILE  = "owner_key.txt"

def _load_owner_key():
    # 1) Streamlit secrets (recommended on Cloud)
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass
    # 2) local owner_key.txt
    try:
        if _os.path.exists(OWNER_KEYFILE):
            with open(OWNER_KEYFILE,"r",encoding="utf-8") as f:
                v=f.read().strip()
                if v: return v
    except Exception:
        pass
    # 3) environment variable
    return (_os.environ.get("GEMINI_API_KEY") or "").strip()

def _save_owner_key(val):
    try:
        with open(OWNER_KEYFILE,"w",encoding="utf-8") as f:
            f.write(val.strip())
        return True
    except Exception:
        return False

# the active key used everywhere - always the owner key, loaded fresh
st.session_state.gemini_key = _load_owner_key()
if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = "gemini-2.5-flash"

with st.sidebar:
    st.markdown("### 🤖 AI status")
    if st.session_state.gemini_key:
        st.success("AI is ready.")
    else:
        st.warning("AI is not configured yet. The owner needs to set the API key.")

    # Owner-only panel: hidden behind a password so people you share with can't see/change the key.
    with st.expander("🔒 Owner settings"):
        pw = st.text_input("Owner password", type="password", key="owner_pw",
                           placeholder="Only the owner knows this")
        if pw and pw == OWNER_PASSWORD:
            st.caption("Owner verified. You can set or change the key here. "
                       "Nobody using the app can see this without the password.")
            newk = st.text_input("Gemini API key", value=st.session_state.gemini_key,
                                 type="password", key="owner_newkey")
            st.session_state.gemini_model = st.selectbox("Model",
                ["gemini-2.5-flash","gemini-2.5-flash-lite","gemini-2.5-pro"],
                index=["gemini-2.5-flash","gemini-2.5-flash-lite","gemini-2.5-pro"].index(st.session_state.gemini_model))
            cobtn=st.columns(2)
            if cobtn[0].button("Save key"):
                if _save_owner_key(newk):
                    st.session_state.gemini_key=newk.strip()
                    st.success("Saved. AI key updated.")
                else:
                    st.error("Could not write owner_key.txt. On Streamlit Cloud, use Secrets instead "
                             "(add GEMINI_API_KEY in app settings).")
            if cobtn[1].button("Test connection"):
                try:
                    r=ai_generate("Reply with exactly: OK", st.session_state.gemini_key,
                                  st.session_state.gemini_model, timeout=30)
                    st.success("Connected! AI is ready." if "ok" in r.lower() else f"Connected: {r[:40]}")
                except Exception as e:
                    st.error(str(e))
        elif pw:
            st.error("Wrong owner password.")

# ---------- global CSS to match the emerald-charcoal desktop theme ----------
st.markdown(f"""
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="StockHunter">
<meta name="theme-color" content="{BG}">
<style>
.stApp {{ background:{BG}; color:{TEXT}; }}
section.main > div {{ padding-top:0.6rem; }}
#MainMenu, footer, header {{ visibility:hidden; }}
h1,h2,h3,h4,h5,h6 {{ color:{TEXT}; }}
.block-container {{ padding-top:1rem; padding-bottom:2rem; max-width:1400px; }}
/* phone: tighter padding + scrollable tab bar so all tabs are reachable */
@media (max-width: 640px) {{
  .block-container {{ padding-left:0.5rem; padding-right:0.5rem; }}
  .stTabs [data-baseweb="tab-list"] {{ overflow-x:auto; flex-wrap:nowrap; }}
}}
/* cards */
.sh-card {{ background:{CARD}; border:1px solid {BORDER}; border-radius:10px;
           padding:14px 18px; margin-bottom:12px; }}
.sh-card2 {{ background:{CARD2}; border:1px solid {BORDER}; border-radius:10px;
            padding:14px 18px; margin-bottom:12px; }}
.sh-label {{ color:{MUTED}; font-size:0.8rem; letter-spacing:.04em; text-transform:uppercase; }}
.sh-accent {{ color:{ACCENT}; }}
.sh-good {{ color:{GOOD}; }} .sh-bad {{ color:{BAD}; }} .sh-warn {{ color:{WARN}; }}
.sh-muted {{ color:{MUTED}; }} .sh-gold {{ color:{GOLD}; }}
.sh-mono {{ font-family:'Consolas','Menlo',monospace; }}
.sh-big {{ font-size:2.4rem; font-weight:700; line-height:1; }}
.sh-pill {{ display:inline-block; padding:3px 12px; border-radius:14px;
           font-weight:600; font-size:0.82rem; }}
/* tab bar */
.stTabs [data-baseweb="tab-list"] {{ gap:2px; background:transparent; flex-wrap:wrap; }}
.stTabs [data-baseweb="tab"] {{ background:{CARD}; color:{MUTED}; border-radius:8px 8px 0 0;
    padding:8px 14px; border:1px solid {BORDER}; font-size:0.85rem; }}
.stTabs [aria-selected="true"] {{ background:{ACCENT} !important; color:#ffffff !important; }}
/* inputs / buttons */
.stTextInput input, .stNumberInput input {{ background:{CARD2}; color:{TEXT}; border:1px solid {BORDER}; }}
.stButton button {{ background:{ACCENT}; color:#ffffff; border:none; border-radius:8px;
    font-weight:600; padding:6px 18px; }}
.stButton button:hover {{ background:{ACCENT2}; color:#0d1311; }}
.stSelectbox div[data-baseweb="select"] > div {{ background:{CARD2}; border:1px solid {BORDER}; color:{TEXT}; }}
.stRadio label, .stCheckbox label {{ color:{TEXT}; }}
hr {{ border-color:{BORDER}; }}
a {{ color:{ACCENT}; }}
.sh-row {{ display:flex; justify-content:space-between; padding:3px 0;
          border-bottom:1px solid {BORDER}33; }}
.sh-row .k {{ color:{MUTED}; }} .sh-row .v {{ font-family:Consolas,monospace; }}
</style>
""", unsafe_allow_html=True)


# ---------- helpers ----------
def pill(text, kind):
    col=COLORMAP.get(kind, NEUTRAL)
    return f'<span class="sh-pill" style="background:{col}22;color:{col};border:1px solid {col}66;">{text}</span>'

def row_html(k, v, col=TEXT):
    return f'<div class="sh-row"><span class="k">{k}</span><span class="v" style="color:{col}">{v}</span></div>'

def signal_kind(text_kind):
    return {"good":"good","bad":"bad","warn":"warn","neutral":"neutral"}.get(text_kind,"neutral")

@st.cache_data(ttl=120, show_spinner=False)
def cached_analyze(symbol, market):
    core.set_market(market)
    sym,_,hist,info = fetch_stock(symbol)
    d = analyze(sym, hist, info)
    news = fetch_news(yf.Ticker(sym if market=="US" else sym+".NS"))
    return d, news

@st.cache_data(ttl=120, show_spinner=False)
def cached_mood(market):
    core.set_market(market)
    return get_market_mood()

@st.cache_data(ttl=120, show_spinner=False)
def cached_sectors(market):
    core.set_market(market)
    return get_sector_indices()

@st.cache_data(ttl=120, show_spinner=False)
def cached_forex(market):
    core.set_market(market)
    # FOREX_INSTRUMENTS is a list of (name, symbol, category) tuples
    if market=="Crypto":
        items=[(n,s,c) for n,s,c in FOREX_INSTRUMENTS if c=="Crypto"]
    elif market=="Commodity":
        items=[(n,s,c) for n,s,c in FOREX_INSTRUMENTS if c=="Commodity"]
    else:  # Forex
        items=[(n,s,c) for n,s,c in FOREX_INSTRUMENTS if c in ("Major","Cross","Exotic")]
    out=[]
    for nm,sym,cat in items:
        fx=analyze_forex(sym)
        if fx:
            fx["name"]=nm; fx["cat"]=cat; fx["yahoo"]=sym
            out.append(fx)
    order={"buy":0,"sell":1,None:2}
    out.sort(key=lambda x:order.get(x["side"],2))
    return out


def chart_figure(symbol, market, period, interval, d, ma_mode="SMA"):
    """Plotly candlestick + SMA/EMA + volume + unified hover (Price + MAs)."""
    from plotly.subplots import make_subplots
    core.set_market(market)
    ysym = symbol if market=="US" else symbol+".NS"
    h = yf.Ticker(ysym).history(period=period, interval=interval)
    if (h is None or h.empty) and market!="US":
        h = yf.Ticker(symbol+".BO").history(period=period, interval=interval)
    if h is None or h.empty or len(h)<5:
        return None
    h = h.tail(120)
    c=h["Close"]
    # two stacked panels: price (big) + volume (small), shared x-axis
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.78,0.22], vertical_spacing=0.04)
    fig.add_trace(go.Candlestick(x=h.index, open=h["Open"], high=h["High"],
        low=h["Low"], close=h["Close"], name="Price",
        increasing_line_color=GOOD, decreasing_line_color=BAD,
        increasing_fillcolor=GOOD, decreasing_fillcolor=BAD), row=1, col=1)
    lbl1,lbl2 = ("20 EMA","50 EMA") if ma_mode=="EMA" else ("20 SMA","50 SMA")
    ma1 = c.ewm(span=20).mean() if ma_mode=="EMA" else c.rolling(20).mean()
    ma2 = c.ewm(span=50).mean() if ma_mode=="EMA" else c.rolling(50).mean()
    if len(h)>=20:
        fig.add_trace(go.Scatter(x=h.index,y=ma1,name=lbl1,line=dict(color=ACCENT2,width=1.4)), row=1,col=1)
    if len(h)>=50:
        fig.add_trace(go.Scatter(x=h.index,y=ma2,name=lbl2,line=dict(color=GOLD,width=1.4)), row=1,col=1)
    # volume bars - green if close>=open else red
    vcols=[GOOD if cl>=op else BAD for op,cl in zip(h["Open"],h["Close"])]
    fig.add_trace(go.Bar(x=h.index,y=h["Volume"],name="Volume",marker_color=vcols,
                         marker_line_width=0,opacity=0.6,showlegend=False), row=2,col=1)
    tp=d["trade_plan"]
    fig.add_hline(y=tp["stop_loss"], line=dict(color=BAD,dash="dash",width=1),
                  annotation_text="SL", row=1, col=1)
    fig.add_hline(y=tp["target1"], line=dict(color=GOOD,dash="dash",width=1),
                  annotation_text="Target1", row=1, col=1)
    fig.update_layout(template="plotly_dark", paper_bgcolor=CARD, plot_bgcolor=CARD,
        height=460, margin=dict(l=10,r=10,t=10,b=10),
        xaxis_rangeslider_visible=False, legend=dict(orientation="h",y=1.04,x=0),
        font=dict(color=TEXT),
        hovermode="x unified",          # <- one tooltip showing Price + both MAs
        hoverlabel=dict(bgcolor=CARD2, bordercolor=BORDER, font=dict(color=TEXT)))
    fig.update_xaxes(gridcolor=BORDER); fig.update_yaxes(gridcolor=BORDER)
    fig.update_yaxes(title_text="Vol", row=2, col=1, showgrid=False)
    return fig


def fx_chart_figure(fx, ma_mode="EMA"):
    from plotly.subplots import make_subplots
    h=fx.get("hist")
    if h is None or h.empty: return None
    h=h.tail(120); c=h["Close"]
    has_vol = "Volume" in h.columns and h["Volume"].fillna(0).sum()>0
    if has_vol:
        fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.78,0.22],vertical_spacing=0.04)
    else:
        fig=make_subplots(rows=1,cols=1)
    fig.add_trace(go.Candlestick(x=h.index,open=h["Open"],high=h["High"],low=h["Low"],close=h["Close"],
        name="Price",increasing_line_color=GOOD,decreasing_line_color=BAD,
        increasing_fillcolor=GOOD,decreasing_fillcolor=BAD),row=1,col=1)
    if ma_mode=="EMA":
        m1,m2,l1,l2=c.ewm(span=20).mean(),c.ewm(span=50).mean(),"EMA20","EMA50"
    else:
        m1,m2,l1,l2=c.rolling(20).mean(),c.rolling(50).mean(),"SMA20","SMA50"
    fig.add_trace(go.Scatter(x=h.index,y=m1,name=l1,line=dict(color=ACCENT2,width=1.4)),row=1,col=1)
    fig.add_trace(go.Scatter(x=h.index,y=m2,name=l2,line=dict(color=GOLD,width=1.4)),row=1,col=1)
    if has_vol:
        vcols=[GOOD if cl>=op else BAD for op,cl in zip(h["Open"],h["Close"])]
        fig.add_trace(go.Bar(x=h.index,y=h["Volume"],marker_color=vcols,marker_line_width=0,
                             opacity=0.6,showlegend=False),row=2,col=1)
    fig.update_layout(template="plotly_dark",paper_bgcolor=CARD,plot_bgcolor=CARD,height=460,
        margin=dict(l=10,r=10,t=10,b=10),xaxis_rangeslider_visible=False,
        legend=dict(orientation="h",y=1.04,x=0),font=dict(color=TEXT),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=CARD2,bordercolor=BORDER,font=dict(color=TEXT)))
    fig.update_xaxes(gridcolor=BORDER); fig.update_yaxes(gridcolor=BORDER)
    return fig


# =========================================================
#  HEADER
# =========================================================
if "market" not in st.session_state: st.session_state.market="India"
core.set_market(st.session_state.market)

hc1,hc2,hc3 = st.columns([3,4,3])
with hc1:
    st.markdown(f"<span style='font-size:1.8rem;font-weight:800;color:{ACCENT}'>StockHunter</span> "
                f"<span class='sh-muted'>Analysis Desk</span>", unsafe_allow_html=True)
with hc3:
    mood = cached_mood(st.session_state.market)
    # --- proper IST clock + market open/close status ---
    try:
        from datetime import timezone, timedelta
        ist = _dt.datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    except Exception:
        ist = _dt.datetime.now()
    now = ist.strftime("%d-%b-%Y %H:%M") + " IST"
    # market hours: India 9:15-15:30 IST Mon-Fri; US 19:00-01:30 IST (approx); crypto 24x7
    wd = ist.weekday()  # 0=Mon ... 6=Sun
    mins = ist.hour*60 + ist.minute
    mk_now = st.session_state.market
    if mk_now in ("Crypto",):
        mstat, mcol = "OPEN (24x7)", GOOD
    elif mk_now == "US":
        is_open = (wd < 5) and (19*60 <= mins or mins <= 90)
        mstat, mcol = ("OPEN", GOOD) if is_open else ("CLOSED", BAD)
    elif mk_now in ("India",):
        is_open = (wd < 5) and (9*60+15 <= mins <= 15*60+30)
        mstat, mcol = ("OPEN", GOOD) if is_open else ("CLOSED", BAD)
    else:
        mstat, mcol = "", MUTED
    statline = f" · <span style='color:{mcol};font-weight:600'>{mstat}</span>" if mstat else ""
    if mood:
        mc = COLORMAP.get(mood.get("color","neutral"),MUTED)
        st.markdown(f"<div style='text-align:right'><span class='sh-muted'>{now}{statline}</span><br>"
                    f"<b>{mood['label']} {mood['ltp']:,.0f} ({mood['change_pct']:+.1f}%)</b> | "
                    f"<span style='color:{mc}'>{mood['mood']}</span></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align:right' class='sh-muted'>{now}{statline}</div>", unsafe_allow_html=True)

# market segment buttons
mkt_cols = st.columns(6)
mkt_defs = [("India","🇮🇳 Indian Market"),("US","🇺🇸 US Market"),("Crypto","₿ Crypto"),
            ("Forex","💱 Forex"),("Commodity","🛢 Commodity"),("MF","🏦 Mutual Funds")]
for i,(key,label) in enumerate(mkt_defs):
    with mkt_cols[i]:
        if st.button(label, key=f"mkt_{key}", use_container_width=True):
            # switching market -> wipe any leftover search/analysis from previous market
            for _k in ("analyze_sym","cmp_rows","cmp_syms","cmp_ai_txt","mf_sel",
                       "scan_perfect","scan_market","scan_dip","mb_result","sector_out"):
                st.session_state.pop(_k, None)
            st.session_state.market = key
            st.rerun()

# sector strip
if st.session_state.market in ("India","US"):
    secs = cached_sectors(st.session_state.market)
    if secs:
        strip = "  ".join(
            f"<span class='sh-muted'>{s['name']}</span> "
            f"<span style='color:{GOOD if s['change_pct']>=0 else BAD}'>{s['change_pct']:+.1f}%</span>"
            for s in secs)
        st.markdown(f"<div class='sh-card2' style='padding:6px 14px'>{strip}</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
#  STImarket-specific rendering
# =========================================================
MK = st.session_state.market

# ========== AI feature helpers (web) ==========
def _num(v):
    try: return float(str(v).replace("%","").replace(",","").strip())
    except Exception: return None

def run_multibagger_one(sym):
    try:
        with st.spinner(f"Assessing {sym}..."):
            s2,_,hist,info=fetch_stock(sym); full=analyze(s2,hist,info); f=full["fundamentals"]
            prompt=("You are a long-term analyst hunting multibaggers. Be honest - most stocks are NOT. "
                "Use clear, simple, plain ENGLISH only (no Hindi or Hinglish). No intro line, no '---'. First line must be 'ODDS:' "
                "then HIGH/MEDIUM/LOW + short reason.\n\n"
                f"Company: {f.get('Company',sym)} ({sym}), sector {f.get('Sector','?')}. "
                f"Market Cap {f.get('Market Cap (Cr)','?')} Cr, P/E {f.get('P/E Ratio','?')}, "
                f"ROE {f.get('ROE %','?')}%, Debt/Equity {f.get('Debt/Equity','?')}.\n\n"
                "Headings: ODDS, MARKET SIZE & OPPORTUNITY, REVENUE & PROFIT GROWTH, INDUSTRY TAILWINDS, "
                "MANAGEMENT QUALITY, RISKS THAT COULD STOP GROWTH, BOTTOM LINE. Under 340 words.")
            txt=ai_call(prompt)
        st.session_state["mb_result"]={"mode":"one","sym":sym,"txt":txt,"cands":None}
    except Exception as e:
        st.session_state["mb_result"]={"mode":"err","txt":str(e)}

def run_multibagger_scan():
    try:
        universe=active_list()[:100]
        cands=[]
        prog=st.progress(0.0, text="Scanning market...")
        for i,sym in enumerate(universe):
            try:
                s2,_,hist,info=fetch_stock(sym); d=analyze(s2,hist,info); f=d["fundamentals"]
                mcap=_num(f.get("Market Cap (Cr)")); roe=_num(f.get("ROE %"))
                de=_num(f.get("Debt/Equity")); pe=_num(f.get("P/E Ratio"))
                if mcap is None or roe is None: continue
                score=0
                if mcap<60000: score+=2
                if mcap<20000: score+=1
                if roe>=15: score+=2
                if roe>=20: score+=1
                if de is not None and de<1: score+=1
                if pe is not None and pe<40: score+=1
                if d["ltp"]>d["dma200"]: score+=1
                if score>=5:
                    cands.append({"sym":s2,"mcap":mcap,"roe":roe,"de":de,"pe":pe,"score":score,
                                  "ltp":d["ltp"],"change":d["change_pct"]})
            except Exception: pass
            if i%10==0: prog.progress(min(1.0,i/len(universe)), text=f"Scanning... {i}/{len(universe)}")
        prog.empty()
        cands.sort(key=lambda x:(-x["score"],x["mcap"])); top=cands[:15]
        if not top:
            st.session_state["mb_result"]={"mode":"err","txt":"No strong candidates found right now."}; return
        lines=[f"{c['sym']}: mcap {c['mcap']:.0f}cr, ROE {c['roe']:.0f}%, D/E {c['de']}, P/E {c['pe']}" for c in top]
        prompt=("You are a long-term analyst. From this PRE-SCREENED list (small/mid cap, decent ROE, low debt, "
            "uptrend), pick best MULTIBAGGER potential over 3-5 years. Rank and explain briefly. "
            "Use clear, simple, plain ENGLISH only (no Hindi or Hinglish). No intro line, no '---'.\n\n"+"\n".join(lines)+
            "\n\nGive a short 2-line note for EVERY symbol in the list (why it could multiply + main risk), "
            "using each stock symbol as a heading. Most promising first. End with a TOP PICK heading. Under 600 words.")
        with st.spinner("AI ranking candidates..."):
            txt=ai_call(prompt)
        st.session_state["mb_result"]={"mode":"scan","cands":top,"txt":txt}
    except Exception as e:
        st.session_state["mb_result"]={"mode":"err","txt":str(e)}

def show_multibagger_result():
    import re
    r=st.session_state.get("mb_result")
    if not r: return
    if r["mode"]=="err": st.error(r["txt"]); return
    if r["mode"]=="scan":
        st.caption("Tip: click any row to open that stock's full analysis in the Analyze tab.")
        df=pd.DataFrame([{"Stock":c["sym"],"Price":f"{core.CUR}{c['ltp']:,.1f}","Chg%":f"{c['change']:+.1f}",
            "Mkt Cap (Cr)":f"{c['mcap']:,.0f}","ROE%":f"{c['roe']:.0f}","D/E":c["de"],"P/E":c["pe"],
            "Score":c["score"]} for c in r["cands"]])
        clickable_table(df, key="tbl_multibagger")
    if r["mode"]=="one":
        m=re.search(r"ODDS\s*:\s*(HIGH|MEDIUM|LOW)\s*[-–:]?\s*(.*)", r["txt"], re.IGNORECASE)
        if m:
            odds=m.group(1).upper(); col={"HIGH":GOOD,"MEDIUM":WARN,"LOW":BAD}.get(odds,WARN)
            st.markdown(f"<div style='background:{col};color:#0d1311;padding:10px 16px;border-radius:8px;"
                        f"font-weight:700;font-size:1.1rem;margin-bottom:8px'>MULTIBAGGER POTENTIAL: {odds} "
                        f"&nbsp;<span style='font-weight:400'>{clean_md(m.group(2))}</span></div>", unsafe_allow_html=True)
    render_ai_text(r["txt"])

def run_macro():
    mkt="India (NSE/BSE)" if MK=="India" else "US markets"
    try:
        with st.spinner("Getting macro read..."):
            prompt=(f"Act as a macro strategist. Big-picture read for {mkt} for a retail investor. "
                "Use clear, simple, plain ENGLISH only (no Hindi or Hinglish). Where unsure of latest number say 'check latest data'. "
                "No intro line, no '---'.\n\nHeadings: INTEREST RATES, INFLATION, GDP GROWTH, CURRENCY, "
                "GEOPOLITICS, NET MARKET IMPACT. Under 300 words.")
            st.session_state["macro_txt"]=ai_call(prompt)
    except Exception as e:
        st.error(str(e))

def load_macro_news():
    try:
        idx="^NSEI" if MK=="India" else "^GSPC"
        st.session_state["macro_news"]=fetch_news(yf.Ticker(idx), limit=12)
    except Exception as e:
        st.error(f"News error: {e}")

def run_portfolio():
    try:
        rows=[]; total=0.0; invested=0.0
        with st.spinner("Pricing holdings & analyzing..."):
            for h in st.session_state.pf_rows:
                sym=h["sym"].upper(); qty=h["qty"]; buy=h["buy"]
                ltp=buy; verdict="N/A"; sl=tp=None; sector=""
                try:
                    s2,_,hist,info=fetch_stock(sym); dd=analyze(s2,hist,info)
                    ltp=dd["ltp"]; verdict=dd["verdict"]; sl=dd["trade_plan"]["stop_loss"]
                    tp=dd["trade_plan"]["target1"]; sector=dd["fundamentals"].get("Sector","")
                except Exception:
                    rq=quick_analyze(sym)
                    if rq: ltp=rq["ltp"]; verdict=rq["verdict"]
                val=ltp*qty; inv=buy*qty; total+=val; invested+=inv
                rows.append({"sym":sym,"qty":qty,"buy":buy,"ltp":ltp,"val":val,"inv":inv,
                    "pnl_pct":((ltp-buy)/buy*100 if buy else 0),"pnl_abs":(ltp-buy)*qty,
                    "verdict":verdict,"sl":sl,"tp":tp,"sector":sector})
            lines=[f"{r['sym']} ({r['sector']}): qty {r['qty']:.0f}, buy {r['buy']:.0f}, ltp {r['ltp']:.0f}, "
                   f"value {r['val']:.0f} ({(r['val']/total*100 if total else 0):.0f}%), P&L {r['pnl_pct']:+.0f}%, "
                   f"signal {r['verdict']}" for r in rows]
            prompt=("Act as an experienced portfolio manager. Be honest and specific. "
                "Use clear, simple, plain ENGLISH only (no Hindi or Hinglish). Currency INR. No intro line, no '---'.\n\n"
                f"Total invested ~{invested:,.0f}. Current value ~{total:,.0f}. Holdings:\n"+"\n".join(lines)+
                "\n\nHeadings: OVERVIEW, OVEREXPOSURE, WEAK POSITIONS, HIDDEN RISKS, MISSING SECTORS, "
                "BETTER ALLOCATION, ACTION SUMMARY. End with 'Educational analysis, not investment advice.' Under 420 words.")
            txt=ai_call(prompt)
        st.session_state["pf_result"]={"rows":rows,"total":total,"invested":invested,"txt":txt}
    except Exception as e:
        st.error(str(e))

def show_portfolio_result():
    r=st.session_state["pf_result"]; total=r["total"]; invested=r["invested"]
    pnl=total-invested; pnl_pct=(pnl/invested*100) if invested else 0
    m=st.columns(4)
    m[0].metric("Capital deployed", f"{core.CUR} {invested:,.0f}")
    m[1].metric("Current value", f"{core.CUR} {total:,.0f}")
    m[2].metric("Overall P&L", f"{core.CUR} {pnl:,.0f}", f"{pnl_pct:+.1f}%")
    m[3].metric("Holdings", f"{len(r['rows'])}")
    df=pd.DataFrame([{"Stock":x["sym"],"Qty":f"{x['qty']:.0f}","Buy":f"{x['buy']:,.0f}","LTP":f"{x['ltp']:,.0f}",
        "Invested":f"{x['inv']:,.0f}","Value":f"{x['val']:,.0f}","Wt%":f"{(x['val']/total*100 if total else 0):.0f}",
        "P&L Rs":f"{x['pnl_abs']:+,.0f}","P&L%":f"{x['pnl_pct']:+.0f}",
        "SL":f"{x['sl']:.0f}" if x['sl'] else "-","Target":f"{x['tp']:.0f}" if x['tp'] else "-",
        "Signal":x["verdict"]} for x in r["rows"]])
    st.dataframe(df,use_container_width=True,hide_index=True)
    st.markdown("#### 🧑‍💼 Portfolio Manager Review (AI)")
    render_ai_text(r["txt"])

def render_stock_market():
    TAB_LABELS=["📈 Analyze","⭐ PERFECT BUY","🔎 Market Scan","💎 Buy the Dip",
                "🚀 Multibagger","🔄 Sector Rotation","⚖️ Compare","⭐ Watchlist",
                "🌐 Macro / News","💼 Portfolio"]
    T_ANALYZE,T_PERFECT,T_SCAN,T_DIP,T_MULTI,T_SECTOR,T_COMPARE,T_WATCH,T_MACRO,T_PF=range(10)
    # a controllable "tab" via radio - lets a row-click jump straight to Analyze
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = TAB_LABELS[0]
    if st.session_state.pop("jump_to_analyze", False):
        st.session_state.active_tab = TAB_LABELS[0]
    active = st.radio("Section", TAB_LABELS, horizontal=True, key="active_tab",
                      label_visibility="collapsed")
    cur = TAB_LABELS.index(active)

    # ---------- ANALYZE ----------
    if cur==T_ANALYZE:
        c1,c2 = st.columns([5,1])
        with c1:
            sym = st.text_input("Symbol", value=st.session_state.get("analyze_sym","HDFCBANK" if MK=="India" else "AAPL"),
                                label_visibility="collapsed", key="analyze_input")
        with c2:
            go_btn = st.button("Analyze", use_container_width=True, key="analyze_go")
        if go_btn and sym.strip():
            st.session_state.analyze_sym = sym.strip().upper()
        cur_sym = st.session_state.get("analyze_sym")
        if cur_sym:
            try:
                with st.spinner(f"Analyzing {cur_sym}..."):
                    d, news = cached_analyze(cur_sym, MK)
            except Exception as e:
                st.error(f"Could not analyze {cur_sym}. ({e})")
                d=None
            if d:
                render_analysis(d, news, cur_sym)

    # ---------- MULTIBAGGER ----------
    if cur==T_MULTI:
        st.markdown("AI judges long-term **multibagger potential** - market size, growth, tailwinds, management, risks.")
        mc1,mc2 = st.columns([3,2])
        with mc1:
            mb_sym = st.text_input("Symbol to assess", value="", key="mb_sym",
                                   placeholder="e.g. "+("DIXON" if MK=="India" else "PLTR"))
        with mc2:
            st.write(""); st.write("")
        b1,b2 = st.columns(2)
        with b1:
            mb_one = st.button("Assess this stock", key="mb_one", use_container_width=True)
        with b2:
            mb_scan = st.button("🔍 Scan market for multibaggers", key="mb_scan", use_container_width=True)
        if mb_one:
            if not ai_ready(): st.warning("Add your free Gemini key in the sidebar (left).")
            elif not mb_sym.strip(): st.warning("Type a symbol first.")
            else: run_multibagger_one(mb_sym.strip().upper())
        if mb_scan:
            if not ai_ready(): st.warning("Add your free Gemini key in the sidebar (left).")
            else: run_multibagger_scan()
        show_multibagger_result()

    # ---------- PERFECT BUY ----------
    if cur==T_PERFECT:
        st.markdown("Shows only stocks **perfect to buy right now** - strong, trending, in entry zone.")
        if st.button("Find Perfect Buys (Top 100)", key="perfect_go"):
            run_scan_generic("perfect")
        show_scan_result("perfect")

    # ---------- MARKET SCAN ----------
    if cur==T_SCAN:
        cqs = st.columns(2)
        with cqs[0]:
            if st.button("Quick Scan (Top 100)", key="ms_quick"): run_scan_generic("market",100)
        with cqs[1]:
            if st.button("Bigger Scan (Top 300)", key="ms_full"): run_scan_generic("market",300)
        show_scan_result("market")

    # ---------- BUY THE DIP ----------
    if cur==T_DIP:
        st.markdown("Quality stocks in an uptrend that have **cooled off near the 20 DMA** - good pullback entries.")
        if st.button("Find Dips (Top 200)", key="dip_go"): run_scan_generic("dip",200)
        show_scan_result("dip")

    # ---------- SECTOR ROTATION ----------
    if cur==T_SECTOR:
        st.markdown("Where the **money is flowing** - sectors ranked by average confidence.")
        if st.button("Scan Sectors", key="sector_go"):
            with st.spinner("Scanning sectors..."):
                core.set_market(MK)
                st.session_state.sector_out = get_sector_strength()
        for s in st.session_state.get("sector_out",[]):
            cc = GOOD if s["avg_change"]>=0 else BAD
            st.markdown(row_html(f"{s['sector']}  ({s['count']} stocks)",
                        f"conf {s['avg_conf']:.0f}%   {s['avg_change']:+.1f}%", cc), unsafe_allow_html=True)

    # ---------- COMPARE ----------
    if cur==T_COMPARE:
        st.markdown("Compare 2-5 stocks side by side. Pick each stock from its own dropdown (type to search by symbol).")
        core.set_market(MK)
        all_syms = sorted(set(active_list()))
        # company-name + symbol both shown in the dropdown label
        opts = ["—"] + all_syms
        nslots = st.slider("How many stocks to compare?", 2, 5, 3, key="cmp_n")
        slot_cols = st.columns(nslots)
        chosen=[]
        for i in range(nslots):
            with slot_cols[i]:
                pick = st.selectbox(f"Stock {i+1}", opts, key=f"cmp_slot_{i}",
                                    index=0)
                if pick and pick!="—":
                    chosen.append(pick)
        if st.button("Compare", key="cmp_go"):
            syms=[s.strip().upper() for s in chosen][:5]
            if len(syms)<2:
                st.warning("Pick at least 2 stocks.")
            else:
                rows=[]
                with st.spinner("Comparing..."):
                    for s in syms:
                        r=quick_analyze_cached(s,MK)
                        if r: rows.append(r)
                st.session_state["cmp_rows"]=rows; st.session_state["cmp_syms"]=syms
        rows=st.session_state.get("cmp_rows",[])
        if rows:
            df=pd.DataFrame([{"Stock":r["symbol"],"Price":f"{core.CUR} {r['ltp']:,.1f}",
                "Day%":f"{r['change_pct']:+.1f}","Conf%":f"{r['confidence']:.0f}",
                "Signal":r["verdict"]} for r in rows])
            clickable_table(df, key="tbl_compare")
            if ai_ready() and st.button("🤖 AI: which is the better long-term pick?", key="cmp_ai"):
                try:
                    with st.spinner("AI comparing..."):
                        info_lines=[]
                        for s in st.session_state.get("cmp_syms",[]):
                            try:
                                s2,_,h,inf=fetch_stock(s); dd=analyze(s2,h,inf); ff=dd["fundamentals"]
                                info_lines.append(f"{s2}: P/E {ff.get('P/E Ratio')}, ROE {ff.get('ROE %')}%, "
                                    f"Debt/Equity {ff.get('Debt/Equity')}, conf {dd['confidence']:.0f}, verdict {dd['verdict']}")
                            except Exception: pass
                        p=("Act as an equity analyst. Compare these stocks and pick the better LONG-TERM pick. "
                           "Rank on growth, profitability, valuation, debt, management quality, future opportunity. "
                           "Use clear, simple, plain ENGLISH only (no Hindi or Hinglish). No intro line, no '---'.\n\n"+"\n".join(info_lines)+
                           "\n\nHeadings: RANKING, BEST PICK, WATCH-OUTS. Under 260 words.")
                        st.session_state["cmp_ai_txt"]=ai_call(p)
                except Exception as e:
                    st.error(str(e))
            if st.session_state.get("cmp_ai_txt"):
                render_ai_text(st.session_state["cmp_ai_txt"])

    # ---------- WATCHLIST ----------
    if cur==T_WATCH:
        wl = load_watch()
        cwl = st.columns([4,1])
        with cwl[0]:
            newsym = st.text_input("Add symbol", key="wl_add_in", label_visibility="collapsed", placeholder="e.g. RELIANCE")
        with cwl[1]:
            if st.button("Add", key="wl_add_btn") and newsym.strip():
                s=newsym.strip().upper()
                if s not in wl: wl.append(s); save_watch(wl); st.rerun()
        if st.button("Refresh watchlist", key="wl_refresh"): st.cache_data.clear()
        if not wl:
            st.info("Watchlist empty. Add a symbol above.")
        for s in wl:
            r=quick_analyze_cached(s,MK)
            rc=st.columns([5,1,1])
            with rc[0]:
                if r:
                    cc=GOOD if r["change_pct"]>=0 else BAD
                    st.markdown(f"<b>{s}</b>  <span class='sh-mono'>{core.CUR} {r['ltp']:,.1f}</span> "
                        f"<span style='color:{cc}'>{r['change_pct']:+.1f}%</span>  {pill(r['verdict'],r['vcolor'])} "
                        f"<span class='sh-muted'>conf {r['confidence']:.0f}%</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<b>{s}</b> <span class='sh-muted'>data N/A</span>", unsafe_allow_html=True)
            with rc[1]:
                if st.button("Analyze", key=f"wl_an_{s}"):
                    st.session_state.analyze_sym=s; st.session_state["jump_to_analyze"]=True; st.rerun()
            with rc[2]:
                if st.button("Remove", key=f"wl_rm_{s}"):
                    wl.remove(s); save_watch(wl); st.rerun()

    # ---------- MACRO / NEWS ----------
    if cur==T_MACRO:
        st.markdown("Big-picture **macro read** - rates, inflation, GDP, currency, geopolitics - and market news.")
        mcol1,mcol2 = st.columns(2)
        with mcol1:
            if st.button("Get Macro Read (AI)", key="macro_go", use_container_width=True):
                if not ai_ready(): st.warning("Add your free Gemini key in the sidebar.")
                else: run_macro()
        with mcol2:
            if st.button("Load Market News", key="macro_news", use_container_width=True):
                load_macro_news()
        if st.session_state.get("macro_txt"):
            render_ai_text(st.session_state["macro_txt"])
        for item in st.session_state.get("macro_news",[]):
            title,pub,url=item[0],item[1] if len(item)>1 else "",item[2] if len(item)>2 else ""
            if url: st.markdown(f"- [{title}]({url})  <span class='sh-muted'>({pub})</span>", unsafe_allow_html=True)
            else: st.markdown(f"- {title}  <span class='sh-muted'>({pub})</span>", unsafe_allow_html=True)

    # ---------- PORTFOLIO ----------
    if cur==T_PF:
        st.markdown("Add your holdings, then get an **AI portfolio manager review** + a full table.")
        if "pf_rows" not in st.session_state: st.session_state.pf_rows=[
            {"sym":"TCS","qty":10,"buy":3500.0},{"sym":"RELIANCE","qty":15,"buy":2800.0}]
        f1,f2,f3,f4 = st.columns([3,2,2,2])
        with f1: pf_s=st.text_input("Stock", key="pf_s", placeholder="e.g. HDFCBANK")
        with f2: pf_q=st.text_input("Shares", key="pf_q", placeholder="10")
        with f3: pf_b=st.text_input("Buy price", key="pf_b", placeholder="1600")
        with f4:
            st.write("")
            if st.button("➕ Add", key="pf_add", use_container_width=True):
                try:
                    if pf_s.strip() and float(pf_q) and float(pf_b):
                        st.session_state.pf_rows.append({"sym":pf_s.strip().upper(),"qty":float(pf_q),"buy":float(pf_b)})
                        st.rerun()
                except Exception:
                    st.warning("Shares and buy price must be numbers.")
        if st.session_state.pf_rows:
            st.caption("Current holdings:")
            for i,h in enumerate(st.session_state.pf_rows):
                hc=st.columns([3,2,2,2])
                hc[0].write(h["sym"]); hc[1].write(f"{h['qty']:.0f} sh"); hc[2].write(f"{core.CUR} {h['buy']:,.1f}")
                if hc[3].button("✕ remove", key=f"pf_rm_{i}"):
                    st.session_state.pf_rows.pop(i); st.rerun()
        cpf1,cpf2 = st.columns(2)
        with cpf1:
            if st.button("▶ Analyze My Portfolio", key="pf_go", use_container_width=True):
                if not ai_ready(): st.warning("Add your free Gemini key in the sidebar.")
                elif not st.session_state.pf_rows: st.warning("Add at least one holding.")
                else: run_portfolio()
        with cpf2:
            if st.button("Clear all", key="pf_clear", use_container_width=True):
                st.session_state.pf_rows=[]; st.rerun()
        if st.session_state.get("pf_result"):
            show_portfolio_result()


def render_analysis(d, news, sym):
    left,right = st.columns([1,1])
    with left:
        cc = GOOD if d["change_pct"]>=0 else BAD
        st.markdown(f"<div class='sh-card'><b style='font-size:1.2rem'>{d['fundamentals']['Company']}</b><br>"
            f"<span class='sh-muted'>{sym} · {d['fundamentals']['Sector']}</span><br>"
            f"<span class='sh-big'>{core.CUR} {d['ltp']:,.2f}</span> "
            f"<span style='color:{cc};font-size:1.1rem'>{d['change_pct']:+.2f}%</span></div>",
            unsafe_allow_html=True)

        # confidence + segmented bar
        vc=COLORMAP.get(d["vcolor"],MUTED)
        conf=d["confidence"]
        st.markdown(f"<div class='sh-card'><span class='sh-label'>Buy Confidence</span><br>"
            f"<span class='sh-big' style='color:{vc}'>{conf:.0f}%</span> {pill(d['verdict'],d['vcolor'])}"
            f"<div style='margin-top:8px;height:12px;border-radius:6px;overflow:hidden;display:flex'>"
            f"<div style='flex:40;background:{BAD}'></div><div style='flex:30;background:{WARN}'></div>"
            f"<div style='flex:30;background:{GOOD}'></div></div>"
            f"<div style='position:relative;margin-top:-18px;margin-left:calc({min(conf,100)}% - 6px)'>"
            f"<span style='color:{TEXT};font-size:0.8rem'>▲ {conf:.0f}%</span></div>"
            f"<div style='margin-top:12px' class='sh-muted'>Hold: {d['trade_plan']['hold']}</div></div>",
            unsafe_allow_html=True)

        # price action + EMA
        rows = "".join([
            row_html("20 DMA", f"{core.CUR} {d['dma20']:,.1f}", GOOD if d['ltp']>d['dma20'] else BAD),
            row_html("50 DMA", f"{core.CUR} {d['dma50']:,.1f}", GOOD if d['ltp']>d['dma50'] else BAD),
            row_html("200 DMA", f"{core.CUR} {d['dma200']:,.1f}", GOOD if d['ltp']>d['dma200'] else BAD),
        ])
        emarows = "".join([
            row_html("20 EMA", f"{core.CUR} {d['ema20']:,.1f}", GOOD if d['ltp']>d['ema20'] else BAD),
            row_html("50 EMA", f"{core.CUR} {d['ema50']:,.1f}", GOOD if d['ltp']>d['ema50'] else BAD),
            row_html("200 EMA", f"{core.CUR} {d['ema200']:,.1f}", GOOD if d['ltp']>d['ema200'] else BAD),
        ])
        morerows = "".join([
            row_html("Support (20d)", f"{core.CUR} {d['support']:,.1f}", MUTED),
            row_html("Resistance (20d)", f"{core.CUR} {d['resistance']:,.1f}", MUTED),
            row_html("52W High", f"{core.CUR} {d['high_52']:,.1f} ({d['from_high']:.1f}%)", MUTED),
            row_html("52W Low", f"{core.CUR} {d['low_52']:,.1f} (+{d['from_low']:.1f}%)", MUTED),
            row_html("RSI (14)", f"{d['rsi']:.1f}", WARN if (d['rsi']>70 or d['rsi']<30) else GOOD),
            row_html("Volume vs 20d", f"{d['vol_ratio']:.2f}x", TEXT),
        ])
        st.markdown(f"<div class='sh-card'><span class='sh-label'>Price Action / Levels</span>{rows}"
            f"<div class='sh-muted' style='margin:4px 0'>— EMA (faster, reacts sooner) —</div>{emarows}"
            f"{morerows}</div>", unsafe_allow_html=True)

        # signals
        sig_html = "".join(
            f"<div style='padding:2px 0'><span style='color:{COLORMAP.get(k,MUTED)}'>"
            f"{'✓' if k=='good' else ('✗' if k=='bad' else '•')}</span> {t}</div>"
            for t,k in d["signals"])
        st.markdown(f"<div class='sh-card'><span class='sh-label'>Signals (weighted)</span>{sig_html}</div>",
                    unsafe_allow_html=True)

        # position size calculator
        st.markdown("<div class='sh-card'><span class='sh-label'>Position Size Calculator</span></div>", unsafe_allow_html=True)
        pc = st.columns(3)
        cap = pc[0].number_input("Capital", value=100000, step=10000, key="pos_cap")
        rk = pc[1].number_input("Risk %", value=1.0, step=0.5, key="pos_risk")
        if pc[2].button("Calculate", key="pos_go"):
            tp=d["trade_plan"]; rps=tp["risk_per_share"]
            if rps>0:
                shares=int((cap*rk/100)/rps)
                invest=shares*tp["entry"]; maxloss=shares*rps
                st.markdown(f"<div class='sh-card2'>Shares: <b>{shares}</b> · Investment: {core.CUR} {invest:,.0f}<br>"
                    f"Max loss (SL hit): <span style='color:{BAD}'>{core.CUR} {maxloss:,.0f}</span> "
                    f"({rk}% = {core.CUR} {cap*rk/100:,.0f})</div>", unsafe_allow_html=True)

    with right:
        # chart with SMA/EMA toggle + timeframe
        if HAS_PLOTLY:
            tc = st.columns([3,2])
            with tc[0]:
                tf = st.selectbox("Timeframe", ["5m","15m","1H","4H","1D","1W","1M"], index=4, key="chart_tf")
            with tc[1]:
                ma = st.radio("MA", ["SMA","EMA"], horizontal=True, key="chart_ma")
            tfmap={"5m":("5d","5m"),"15m":("5d","15m"),"1H":("1mo","1h"),"4H":("3mo","1h"),
                   "1D":("6mo","1d"),"1W":("2y","1wk"),"1M":("5y","1mo")}
            per,iv = tfmap[tf]
            with st.spinner("Loading chart..."):
                fig = chart_figure(sym, MK, per, iv, d, ma)
            if fig: st.plotly_chart(fig, use_container_width=True)
            else: st.info("Chart data not available for this timeframe.")

        # trade plan by timeframe (BUY/SELL)
        st.markdown(f"<div class='sh-card2'><span class='sh-accent'><b>🎯 TRADE PLAN BY TIMEFRAME</b></span><br>"
            "<span class='sh-muted'>Pick a holding horizon - signal, entry, SL, targets & holding period update.</span></div>",
            unsafe_allow_html=True)
        tflabel = st.radio("Horizon", list(STOCK_TIMEFRAMES.keys()), horizontal=True, key="stf")
        with st.spinner("..."):
            sig = stock_timeframe_signal(sym, MK, tflabel)
        if sig:
            vc=COLORMAP.get(sig["vcolor"],MUTED)
            rec = sig.get("verdict","")
            st.markdown(f"<div class='sh-card' style='border-color:{vc}'>"
                f"<b style='color:{vc};font-size:1.1rem'>{tflabel}: {rec}</b><br>"
                f"<span class='sh-muted'>RSI {sig['rsi']:.0f} · Hold: {sig['hold']}</span></div>",
                unsafe_allow_html=True)
            if sig.get("side") in ("buy","sell"):
                gc=st.columns(4)
                gc[0].markdown(f"<div class='sh-card2' style='text-align:center'><b>{core.CUR} {sig['ltp']:,.1f}</b><br><span class='sh-muted'>Entry</span></div>", unsafe_allow_html=True)
                gc[1].markdown(f"<div class='sh-card2' style='text-align:center'><b style='color:{BAD}'>{core.CUR} {sig['sl']:,.1f}</b><br><span class='sh-muted'>Stop-Loss</span></div>", unsafe_allow_html=True)
                gc[2].markdown(f"<div class='sh-card2' style='text-align:center'><b style='color:{GOOD}'>{core.CUR} {sig['t1']:,.1f}</b><br><span class='sh-muted'>Target 1</span></div>", unsafe_allow_html=True)
                gc[3].markdown(f"<div class='sh-card2' style='text-align:center'><b style='color:{GOOD}'>{core.CUR} {sig['t2']:,.1f}</b><br><span class='sh-muted'>Target 2</span></div>", unsafe_allow_html=True)

        # fundamentals
        fr = "".join(row_html(k, d["fundamentals"][k]) for k in
                     ["Market Cap (Cr)","P/E Ratio","P/B Ratio","ROE %","Debt/Equity","Div Yield %"])
        st.markdown(f"<div class='sh-card'><span class='sh-label'>Fundamentals (context)</span>{fr}</div>", unsafe_allow_html=True)

        # news (clickable)
        if news:
            items=""
            for it in news:
                title=it[0]; url=it[2] if len(it)>2 else ""
                if url: items+=f"<div style='padding:3px 0'><a href='{url}' target='_blank'>🔗 {title}</a></div>"
                else: items+=f"<div style='padding:3px 0'>• {title}</div>"
            st.markdown(f"<div class='sh-card'><span class='sh-label'>Latest News</span>{items}</div>", unsafe_allow_html=True)

        # ---- AI DEEP ANALYSIS ----
        st.markdown("#### 🤖 AI Deep Analysis")
        if not ai_ready():
            st.info("Add your free Gemini API key in the sidebar (left) to unlock AI analysis.")
        else:
            f=d["fundamentals"]
            a1,a2,a3=st.columns(3)
            run_deep = a1.button("Generate AI Analysis", key=f"ai_deep_{sym}", use_container_width=True)
            run_rf   = a2.button("🚩 Spot Red Flags", key=f"ai_rf_{sym}", use_container_width=True)
            run_sm   = a3.button("💰 Smart Money Check", key=f"ai_sm_{sym}", use_container_width=True)
            base=(f"Stock: {f.get('Company')} ({sym}), sector {f.get('Sector')}. "
                  f"Price {core.CUR}{d['ltp']:.1f}, P/E {f.get('P/E Ratio')}, P/B {f.get('P/B Ratio')}, "
                  f"ROE {f.get('ROE %')}%, Debt/Equity {f.get('Debt/Equity')}, Div Yield {f.get('Div Yield %')}%, "
                  f"Market Cap {f.get('Market Cap (Cr)')} Cr, RSI {d['rsi']:.0f}.\n\n")
            try:
                if run_deep:
                    p=("You are a clear, honest equity analyst for a retail investor. Use clear, simple, plain ENGLISH only (no Hindi or Hinglish). "
                       "No buy/sell command. No intro line, no '---'.\n\n"+base+
                       "Headings: BUSINESS MODEL, FINANCIAL PERFORMANCE, VALUATION, INVESTMENT THESIS. Under 320 words.")
                    with st.spinner("AI reading the numbers..."): st.session_state[f"ai_out_{sym}"]=ai_call(p)
                if run_rf:
                    p=("Act as a sharp forensic accountant / short-seller. Be skeptical. Use clear, simple, plain ENGLISH only (no Hindi or Hinglish). "
                       "No intro line, no '---'. If data missing, say 'data not visible here'.\n\n"+base+
                       "Headings: ACCOUNTING RED FLAGS, CASH FLOW, DEBT CONCERNS, PROMOTER RISK, GOVERNANCE, VERDICT. Under 280 words.")
                    with st.spinner("Scanning for red flags..."): st.session_state[f"ai_out_{sym}"]=ai_call(p)
                if run_sm:
                    p=("Act as an analyst tracking 'smart money'. Explain what big/informed players appear to be doing and "
                       "whether it signals confidence or concern. Where you lack live data say 'check latest filings'. "
                       "Use clear, simple, plain ENGLISH only (no Hindi or Hinglish). No intro line, no '---'.\n\n"+base+
                       "Headings: PROMOTER ACTIVITY, FII / DII TREND, MUTUAL FUND HOLDING, INSIDER / BULK DEALS, "
                       "WHAT SMART MONEY SEEMS TO BE DOING. Under 260 words.")
                    with st.spinner("Checking smart money..."): st.session_state[f"ai_out_{sym}"]=ai_call(p)
            except Exception as e:
                st.error(str(e))
            if st.session_state.get(f"ai_out_{sym}"):
                render_ai_text(st.session_state[f"ai_out_{sym}"])


# ---------- generic scan helpers ----------
@st.cache_data(ttl=120, show_spinner=False)
def quick_analyze_cached(sym, market):
    core.set_market(market)
    return quick_analyze(sym)

def _is_liquid(d):
    """Keep only tradeable, liquid names. Turnover = avg daily volume x price."""
    try:
        avg_vol = d.get("avg_vol") or 0
        turnover = avg_vol * d["ltp"]
        # India: >= Rs 5 crore/day ; US: >= $5 million/day
        floor = 5_000_000 if core.MARKET=="US" else 5_00_00_000
        return turnover >= floor
    except Exception:
        return True

def run_scan_generic(kind, count=100):
    core.set_market(MK)
    syms = active_list()[:count]
    res=[]
    prog=st.progress(0.0, text=f"Scanning {len(syms)}...")
    for i,s in enumerate(syms):
        r=quick_analyze(s)
        if r:
            if kind=="perfect":
                d=safe_full(s)
                if d and is_perfect_buy(d)[0] and _is_liquid(d): res.append(r)
            elif kind=="dip":
                d=safe_full(s)
                if d and is_buy_the_dip(d) and _is_liquid(d): res.append(r)
            else:
                res.append(r)
        prog.progress((i+1)/len(syms))
    prog.empty()
    res.sort(key=lambda x:x["confidence"], reverse=True)
    st.session_state[f"scan_{kind}"]=res

def safe_full(s):
    try:
        core.set_market(MK)
        sym,_,hist,info=fetch_stock(s)
        return analyze(sym,hist,info)
    except Exception:
        return None

# ---------- clickable table helper (works across Streamlit versions) ----------
def clickable_table(df, key, sym_col="Stock"):
    """Show a dataframe where picking a row jumps to the Analyze tab for that symbol.
    Uses native row-selection if available, else falls back to a selectbox + button."""
    picked=None
    try:
        ev = st.dataframe(df, use_container_width=True, hide_index=True,
                          on_select="rerun", selection_mode="single-row", key=key)
        sel = getattr(ev, "selection", None) or (ev.get("selection") if isinstance(ev, dict) else None)
        rows = (sel or {}).get("rows", []) if sel else []
        if rows:
            picked = str(df.iloc[rows[0]][sym_col])
    except TypeError:
        # old Streamlit: no on_select -> plain table + manual picker
        st.dataframe(df, use_container_width=True, hide_index=True)
        opts=list(df[sym_col].astype(str))
        cpk=st.columns([3,1])
        chosen=cpk[0].selectbox("Open a stock in Analyze", opts, key=key+"_pick")
        if cpk[1].button("Open ▶", key=key+"_openbtn"):
            picked=chosen
    if picked:
        st.session_state.analyze_sym = picked.strip().upper()
        st.session_state["jump_to_analyze"] = True
        st.rerun()

def show_scan_result(kind):
    res=st.session_state.get(f"scan_{kind}",[])
    if not res: return
    st.caption("Tip: click any row to open that stock in the Analyze tab.")
    # each scan gets its OWN column so the three tabs don't all just say "STRONG BUY"
    if kind=="perfect":
        col_name="Perfect-Buy?"
        def tag(r): return "✅ PERFECT BUY" if r["confidence"]>=70 else "Good"
    elif kind=="dip":
        col_name="Dip Setup"
        def tag(r): return "💎 BUY THE DIP"
    else:
        col_name="Strength"
        def tag(r):
            c=r["confidence"]
            return "🔥 Very strong" if c>=75 else ("Strong" if c>=60 else ("Moderate" if c>=45 else "Weak"))
    df=pd.DataFrame([{"Stock":r["symbol"],"Price":f"{core.CUR} {r['ltp']:,.1f}",
        "Day%":f"{r['change_pct']:+.1f}","Conf%":f"{r['confidence']:.0f}",
        col_name:tag(r)} for r in res])
    clickable_table(df, key=f"tbl_{kind}")
    st.download_button("⬇ Download Excel (CSV)", df.to_csv(index=False).encode(),
                       file_name=f"stockhunter_{kind}.csv", mime="text/csv", key=f"dl_{kind}")


# ---------- watchlist persistence ----------
def load_watch():
    return core._load_json(core.WATCHJSON, [])
def save_watch(wl):
    core._save_json(core.WATCHJSON, wl)


# =========================================================
#  FOREX / CRYPTO / COMMODITY
# =========================================================
def render_forex_market(cat_label):
    st.markdown(f"##### {cat_label}")
    sel = st.session_state.get(f"fx_sel_{MK}")
    if sel is None:
        # list view
        with st.spinner("Loading instruments..."):
            flist = cached_forex(MK)
        if not flist:
            st.info("No instruments found. Make sure forex_list.py is present.")
        # --- search / filter box ---
        q = st.text_input("Search (e.g. USD, BTC, GOLD)", key=f"fx_q_{MK}",
                          placeholder="Type to filter - e.g. USD shows all USD pairs").strip().upper()
        shown = flist
        if q:
            shown = [fx for fx in flist
                     if q in fx.get("name","").upper() or q in fx.get("yahoo","").upper()]
            if not shown:
                st.warning(f"No instrument matches '{q}'.")
        for fx in shown:
            cc=GOOD if fx["chg"]>=0 else BAD
            rc=st.columns([5,1])
            with rc[0]:
                st.markdown(f"<b>{fx['name']}</b>  <span class='sh-mono'>{_fmt_fx(fx['ltp'])}</span> "
                    f"<span style='color:{cc}'>{fx['chg']:+.2f}%</span>  {pill(fx['verdict'],fx['vcolor'])}",
                    unsafe_allow_html=True)
            with rc[1]:
                if st.button("Open", key=f"fxopen_{fx['yahoo']}"):
                    st.session_state[f"fx_sel_{MK}"]=fx; st.rerun()
    else:
        if st.button("← Back to list", key="fx_back"):
            st.session_state[f"fx_sel_{MK}"]=None; st.rerun()
        render_forex_detail(sel)


def render_forex_detail(fx):
    left,right=st.columns([1,1])
    with left:
        cc=GOOD if fx["chg"]>=0 else BAD
        st.markdown(f"<div class='sh-card'><b style='font-size:1.3rem;color:{ACCENT}'>{fx['name']}</b> "
            f"<span class='sh-muted'>{fx['cat']}</span><br>"
            f"<span class='sh-big'>{_fmt_fx(fx['ltp'])}</span> "
            f"<span style='color:{cc}'>{fx['chg']:+.2f}% today</span></div>", unsafe_allow_html=True)
        vc=COLORMAP.get(fx["vcolor"],MUTED)
        rec="BUY" if fx["side"]=="buy" else ("SELL" if fx["side"]=="sell" else fx["verdict"])
        st.markdown(f"<div class='sh-card'><span class='sh-label'>Overall view (daily / bigger picture)</span><br>"
            f"<span class='sh-big' style='color:{vc}'>{rec}</span><br>"
            f"<span class='sh-muted'>RSI {fx['rsi']:.0f} · Trend: {fx['trend']}</span><br>"
            f"<span class='sh-muted' style='font-size:0.8rem'>Daily-chart bias. A shorter timeframe below can differ - that's a counter-trend trade.</span></div>",
            unsafe_allow_html=True)
        if fx["side"] in ("buy","sell"):
            st.markdown(f"<div class='sh-card'><span class='sh-label'>Trade Levels</span>"
                + row_html("Entry around", _fmt_fx(fx['ltp']))
                + row_html("Stop-Loss", f"{_fmt_fx(fx['sl'])} ({fx['sl_pct']:+.1f}%)", BAD)
                + row_html("Take-Profit", f"{_fmt_fx(fx['tp'])} ({fx['tp_pct']:+.1f}%)", GOOD)
                + "</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='sh-card'><span class='sh-label'>Key Levels</span>"
            + row_html("Support", _fmt_fx(fx.get('support')))
            + row_html("Resistance", _fmt_fx(fx.get('resistance')))
            + row_html("6mo High", _fmt_fx(fx.get('wk_high')))
            + row_html("6mo Low", _fmt_fx(fx.get('wk_low')))
            + "</div>", unsafe_allow_html=True)
    with right:
        if HAS_PLOTLY:
            ma = st.radio("MA", ["EMA","SMA"], horizontal=True, key="fx_ma")
            fig=fx_chart_figure(fx, ma)
            if fig: st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"<div class='sh-card2'><span class='sh-accent'><b>🎯 TRADE PLAN BY TIMEFRAME</b></span></div>", unsafe_allow_html=True)
        tflabel=st.radio("Horizon", list(FX_TIMEFRAMES.keys()), horizontal=True, key="fx_tf")
        with st.spinner("..."):
            sig=forex_timeframe_signal(fx["yahoo"], tflabel)
        if sig:
            vc=COLORMAP.get(sig["vcolor"],MUTED)
            rec="BUY" if sig["side"]=="buy" else ("SELL" if sig["side"]=="sell" else sig["verdict"])
            st.markdown(f"<div class='sh-card' style='border-color:{vc}'>"
                f"<b style='color:{vc};font-size:1.1rem'>{tflabel}: {rec}</b><br>"
                f"<span class='sh-muted'>RSI {sig['rsi']:.0f} · Hold: {sig['hold']}</span></div>", unsafe_allow_html=True)
            if sig["side"] in ("buy","sell") and fx.get("side") in ("buy","sell") and sig["side"]!=fx["side"]:
                st.markdown(f"<div class='sh-card2' style='color:{WARN}'>⚠ Counter-trend: daily says {fx['side'].upper()}, "
                    f"this timeframe says {sig['side'].upper()}. Smaller size, tighter stop.</div>", unsafe_allow_html=True)
            if sig["side"] in ("buy","sell"):
                gc=st.columns(3)
                gc[0].markdown(f"<div class='sh-card2' style='text-align:center'><b>{_fmt_fx(sig['ltp'])}</b><br><span class='sh-muted'>Entry</span></div>", unsafe_allow_html=True)
                gc[1].markdown(f"<div class='sh-card2' style='text-align:center'><b style='color:{BAD}'>{_fmt_fx(sig['sl'])}<br>({sig['sl_pct']:+.1f}%)</b><br><span class='sh-muted'>Stop-Loss</span></div>", unsafe_allow_html=True)
                gc[2].markdown(f"<div class='sh-card2' style='text-align:center'><b style='color:{GOOD}'>{_fmt_fx(sig['tp'])}<br>({sig['tp_pct']:+.1f}%)</b><br><span class='sh-muted'>Take-Profit</span></div>", unsafe_allow_html=True)


# =========================================================
#  MUTUAL FUNDS
# =========================================================
@st.cache_data(ttl=300, show_spinner=False)
def _cached_mf_ok(code):
    return analyze_mf(code)

def cached_mf(code):
    mf=_cached_mf_ok(code)
    if mf and mf.get("error"):
        _cached_mf_ok.clear()          # don't keep a failed result cached
    return mf

def render_mf_market():
    sel = st.session_state.get("mf_sel")
    if sel is None:
        st.markdown("##### Mutual Funds - returns + buy recommendation")
        names=[n for n,c,k in MUTUAL_FUNDS]
        ncode={n:c for n,c,k in MUTUAL_FUNDS}
        ncat={n:k for n,c,k in MUTUAL_FUNDS}
        pick = st.selectbox("Pick a fund", names, key="mf_pick") if names else None
        if pick and st.button("Analyze fund", key="mf_go"):
            with st.spinner("Loading fund..."):
                mf=cached_mf(ncode[pick])
            if mf and not mf.get("error"):
                mf["name"]=pick; mf["cat"]=ncat.get(pick,"")
                st.session_state.mf_sel=mf; st.rerun()
            else:
                reason = (mf or {}).get("error","unknown reason")
                st.error(f"Could not load this fund ({reason}). The free NAV source may be busy - "
                         "wait a few seconds and tap Analyze again.")
        # quick list with action
        if names:
            st.markdown("<span class='sh-label'>Tap a fund above and Analyze. Quick view:</span>", unsafe_allow_html=True)
    else:
        if st.button("← Back to funds", key="mf_back"):
            st.session_state.mf_sel=None; st.rerun()
        render_mf_detail(sel)

def render_mf_detail(mf):
    vc=COLORMAP.get(mf["vcolor"],MUTED)
    st.markdown(f"<div class='sh-card'><b style='font-size:1.2rem;color:{ACCENT}'>{mf['name']}</b><br>"
        f"<span class='sh-muted'>{mf.get('cat','')} · {mf.get('fund_house','')}</span><br>"
        f"<span class='sh-big'>Rs {mf['nav']:,.2f}</span> {pill(mf['verdict'],mf['vcolor'])}</div>", unsafe_allow_html=True)
    action,akey,reason=mf_recommendation(mf)
    acol=COLORMAP.get(akey,TEXT)
    st.markdown(f"<div class='sh-card'><span class='sh-label'>Recommendation</span><br>"
        f"<span class='sh-big' style='color:{acol}'>{action}</span><br><span>{reason}</span></div>", unsafe_allow_html=True)
    gc=st.columns(4)
    gc[0].markdown(f"<div class='sh-card2' style='text-align:center'><b style='color:{GOOD if (mf['r1y'] or 0)>=0 else BAD}'>{mf['r1y']:+.1f}%</b><br><span class='sh-muted'>1 Year</span></div>" if mf['r1y'] is not None else "<div class='sh-card2'>N/A</div>", unsafe_allow_html=True)
    gc[1].markdown(f"<div class='sh-card2' style='text-align:center'><b>{mf['r3y']:+.1f}%</b><br><span class='sh-muted'>3Y/yr</span></div>" if mf['r3y'] is not None else "<div class='sh-card2'>N/A</div>", unsafe_allow_html=True)
    gc[2].markdown(f"<div class='sh-card2' style='text-align:center'><b>{mf['r5y']:+.1f}%</b><br><span class='sh-muted'>5Y/yr</span></div>" if mf['r5y'] is not None else "<div class='sh-card2'>N/A</div>", unsafe_allow_html=True)
    gc[3].markdown(f"<div class='sh-card2' style='text-align:center'><b>{mf['trend']}</b><br><span class='sh-muted'>Trend</span></div>", unsafe_allow_html=True)

    # NAV chart (past NAV history so you can see the trend while analyzing)
    st.markdown("<span class='sh-label'>NAV history</span>", unsafe_allow_html=True)
    if HAS_PLOTLY and mf.get("navs"):
        navs=list(reversed(mf["navs"]))
        xs=[]; ys=[]
        for n in navs:
            try:
                xs.append(_dt.datetime.strptime(n["date"],"%d-%m-%Y"))
                ys.append(float(n["nav"]))
            except Exception:
                continue
        if len(xs)>=2:
            fig=go.Figure(go.Scatter(x=xs,y=ys,line=dict(color=ACCENT,width=1.6),fill="tozeroy",
                fillcolor=f"{ACCENT}22",name="NAV",hovertemplate="%{x|%d-%b-%Y}<br>NAV: %{y:.2f}<extra></extra>"))
            fig.update_layout(template="plotly_dark",paper_bgcolor=CARD,plot_bgcolor=CARD,height=320,
                margin=dict(l=10,r=10,t=10,b=10),font=dict(color=TEXT),
                hovermode="x unified",hoverlabel=dict(bgcolor=CARD2,bordercolor=BORDER,font=dict(color=TEXT)))
            fig.update_xaxes(gridcolor=BORDER); fig.update_yaxes(gridcolor=BORDER)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("NAV history not available for this fund.")
    elif not HAS_PLOTLY:
        st.caption("Install plotly to see the NAV chart.")

    # return calculator
    st.markdown("<div class='sh-card'><span class='sh-label'>Return Calculator</span></div>", unsafe_allow_html=True)
    exp = mf.get("r5y") or mf.get("r3y") or mf.get("r1y") or 10.0
    mode = st.radio("Mode", ["Lumpsum","SIP"], horizontal=True, key="mf_calc_mode")
    cc=st.columns(3)
    amt = cc[0].number_input("Monthly amount (Rs)" if mode=="SIP" else "Amount (Rs)",
                             value=5000 if mode=="SIP" else 100000, step=1000, key="mf_amt")
    yrs = cc[1].number_input("Years", value=5.0, step=1.0, key="mf_yrs")
    rate = cc[2].number_input("Return %/yr", value=float(round(exp,1)), step=0.5, key="mf_rate")
    if mode=="Lumpsum":
        invested=amt; fv=amt*((1+rate/100)**yrs)
    else:
        i=(rate/100)/12; n=int(round(yrs*12)); invested=amt*n
        fv = amt*n if abs(i)<1e-9 else amt*(((1+i)**n-1)/i)*(1+i)
    gain=fv-invested
    rc=st.columns(3)
    rc[0].markdown(f"<div class='sh-card2' style='text-align:center'><b>Rs {invested:,.0f}</b><br><span class='sh-muted'>Invested</span></div>", unsafe_allow_html=True)
    rc[1].markdown(f"<div class='sh-card2' style='text-align:center'><b style='color:{GOOD}'>Rs {fv:,.0f}</b><br><span class='sh-muted'>Est. value</span></div>", unsafe_allow_html=True)
    rc[2].markdown(f"<div class='sh-card2' style='text-align:center'><b style='color:{GOOD if gain>=0 else BAD}'>Rs {gain:,.0f}</b><br><span class='sh-muted'>Est. gain</span></div>", unsafe_allow_html=True)
    st.caption("Estimate only - past performance is not a guarantee. Not financial advice.")


# =========================================================
#  ROUTER
# =========================================================
if MK in ("India","US"):
    render_stock_market()
elif MK in ("Crypto","Forex","Commodity"):
    render_forex_market({"Crypto":"₿ Crypto","Forex":"💱 Forex","Commodity":"🛢 Commodity"}[MK])
elif MK=="MF":
    render_mf_market()

st.markdown(f"<div style='margin-top:20px;color:{MUTED};font-size:0.8rem'>"
    "This is analysis, not advice. Levels are technical - the market can break them. "
    "Always use a stop-loss and manage your risk. Data via Yahoo Finance (may be delayed).</div>",
    unsafe_allow_html=True)
