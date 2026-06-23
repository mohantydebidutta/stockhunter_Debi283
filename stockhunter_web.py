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
    resolve_symbol, search_suggestions, full_universe,
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
    import urllib.request, urllib.error, json as _j, time as _t
    if not api_key or not api_key.strip():
        raise RuntimeError("No API key. The owner needs to set the Gemini key (sidebar > Owner settings).")
    url=f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key.strip()}"
    body={"contents":[{"parts":[{"text":prompt}]}],
          "generationConfig":{"temperature":0.4,"maxOutputTokens":2048}}
    data=_j.dumps(body).encode("utf-8")
    last_err=None
    # Google's servers sometimes return 503 (overloaded) or 429 (busy) - retry a few times.
    for attempt in range(4):
        req=urllib.request.Request(url,data=data,headers={"Content-Type":"application/json"})
        try:
            with urllib.request.urlopen(req,timeout=timeout) as resp:
                out=_j.loads(resp.read().decode("utf-8"))
            try:
                return "".join(p.get("text","") for p in out["candidates"][0]["content"]["parts"]).strip()
            except Exception:
                raise RuntimeError("AI returned no text (possibly blocked by a safety filter).")
        except urllib.error.HTTPError as e:
            if e.code in (400,403):
                raise RuntimeError("API key invalid or not enabled. The owner should re-check the key.")
            if e.code in (429,503,500,502):
                last_err=e
                _t.sleep(1.5*(attempt+1))   # back off and retry
                continue
            raise RuntimeError(f"AI error {e.code}")
        except Exception as e:
            last_err=e
            _t.sleep(1.0*(attempt+1))
            continue
    # all retries failed
    code = getattr(last_err,"code",None)
    if code==503:
        raise RuntimeError("Google's AI is overloaded right now (503). Please wait a minute and try again.")
    if code==429:
        raise RuntimeError("Free AI limit hit for now (429). Wait a bit and retry.")
    raise RuntimeError(f"Could not reach Gemini ({last_err}). Try again shortly.")

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
.stApp {{ background:
    radial-gradient(1200px 600px at 80% -10%, #12211b 0%, transparent 60%),
    radial-gradient(900px 500px at -10% 10%, #101a16 0%, transparent 55%),
    {BG};
  color:{TEXT}; }}
section.main > div {{ padding-top:0.6rem; }}
#MainMenu, footer, header {{ visibility:hidden; }}
h1,h2,h3,h4,h5,h6 {{ color:{TEXT}; }}
.block-container {{ padding-top:1rem; padding-bottom:2rem; max-width:1400px; }}
/* phone: tighter padding + scrollable tab bar so all tabs are reachable */
@media (max-width: 640px) {{
  .block-container {{ padding-left:0.5rem; padding-right:0.5rem; }}
  .stTabs [data-baseweb="tab-list"] {{ overflow-x:auto; flex-wrap:nowrap; }}
}}
/* ---------- cards with soft depth ---------- */
.sh-card {{ background:linear-gradient(180deg, {CARD} 0%, #121b16 100%);
           border:1px solid {BORDER}; border-radius:14px;
           padding:16px 20px; margin-bottom:13px;
           box-shadow:0 6px 18px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.03); }}
.sh-card2 {{ background:linear-gradient(180deg, {CARD2} 0%, #18261f 100%);
            border:1px solid {BORDER}; border-radius:14px;
            padding:16px 20px; margin-bottom:13px;
            box-shadow:0 4px 14px rgba(0,0,0,0.30), inset 0 1px 0 rgba(255,255,255,0.04); }}
.sh-card:hover, .sh-card2:hover {{ border-color:{ACCENT}55; transition:border-color .2s ease; }}
.sh-label {{ color:{MUTED}; font-size:0.8rem; letter-spacing:.05em; text-transform:uppercase; }}
.sh-accent {{ color:{ACCENT}; }}
.sh-good {{ color:{GOOD}; }} .sh-bad {{ color:{BAD}; }} .sh-warn {{ color:{WARN}; }}
.sh-muted {{ color:{MUTED}; }} .sh-gold {{ color:{GOLD}; }}
.sh-mono {{ font-family:'Consolas','Menlo',monospace; }}
.sh-big {{ font-size:2.5rem; font-weight:800; line-height:1;
          text-shadow:0 2px 10px rgba(16,185,129,0.18); }}
.sh-pill {{ display:inline-block; padding:3px 12px; border-radius:14px;
           font-weight:600; font-size:0.82rem;
           box-shadow:inset 0 1px 0 rgba(255,255,255,0.06); }}
/* ---------- radio "tab" bar (our section switcher) - pill buttons with depth ---------- */
div[role="radiogroup"] {{ gap:8px; flex-wrap:wrap; }}
div[role="radiogroup"] > label {{
    background:linear-gradient(180deg, {CARD2} 0%, #16221b 100%);
    border:1px solid {BORDER}; border-radius:10px;
    padding:7px 14px; margin:0; cursor:pointer;
    box-shadow:0 3px 8px rgba(0,0,0,0.30), inset 0 1px 0 rgba(255,255,255,0.04);
    transition:transform .08s ease, box-shadow .15s ease, border-color .15s ease; }}
div[role="radiogroup"] > label:hover {{
    transform:translateY(-2px); border-color:{ACCENT}88;
    box-shadow:0 7px 16px rgba(16,185,129,0.18); }}
/* hide the little radio dot so they read as buttons */
div[role="radiogroup"] > label > div:first-child {{ display:none; }}
/* the SELECTED section glows emerald */
div[role="radiogroup"] > label:has(input:checked) {{
    background:linear-gradient(180deg, {ACCENT} 0%, #0e9f6e 100%);
    border-color:{ACCENT2};
    box-shadow:0 6px 18px rgba(16,185,129,0.45), inset 0 1px 0 rgba(255,255,255,0.25);
    transform:translateY(-1px); }}
div[role="radiogroup"] > label:has(input:checked) p {{ color:#ffffff !important; font-weight:700; }}
/* ---------- inputs ---------- */
.stTextInput input, .stNumberInput input {{
    background:{CARD2}; color:{TEXT}; border:1px solid {BORDER}; border-radius:9px;
    box-shadow:inset 0 2px 5px rgba(0,0,0,0.30); }}
.stTextInput input:focus, .stNumberInput input:focus {{
    border-color:{ACCENT}; box-shadow:0 0 0 2px {ACCENT}33, inset 0 2px 5px rgba(0,0,0,0.30); }}
/* ---------- buttons: 3D, lift on hover, PRESS DOWN on click ---------- */
.stButton button, .stDownloadButton button {{
    background:linear-gradient(180deg, {ACCENT} 0%, #0e9f6e 100%);
    color:#ffffff; border:1px solid {ACCENT2}; border-radius:10px;
    font-weight:700; padding:8px 20px;
    box-shadow:0 5px 14px rgba(16,185,129,0.35), inset 0 1px 0 rgba(255,255,255,0.25);
    transition:transform .08s ease, box-shadow .15s ease, filter .15s ease; }}
.stButton button:hover, .stDownloadButton button:hover {{
    transform:translateY(-2px); filter:brightness(1.06);
    box-shadow:0 9px 20px rgba(16,185,129,0.45), inset 0 1px 0 rgba(255,255,255,0.3); }}
.stButton button:active, .stDownloadButton button:active {{
    transform:translateY(2px) scale(0.99);
    box-shadow:0 2px 6px rgba(16,185,129,0.30), inset 0 2px 6px rgba(0,0,0,0.35); }}
/* secondary (non-active) market buttons sit darker, lift on hover */
.stButton button[kind="secondary"] {{
    background:linear-gradient(180deg, {CARD2} 0%, #16221b 100%);
    color:{TEXT}; border:1px solid {BORDER};
    box-shadow:0 3px 8px rgba(0,0,0,0.30), inset 0 1px 0 rgba(255,255,255,0.04); }}
.stButton button[kind="secondary"]:hover {{
    transform:translateY(-2px); border-color:{ACCENT}88; filter:none;
    box-shadow:0 7px 16px rgba(16,185,129,0.18); }}
.stButton button[kind="primary"] {{
    box-shadow:0 6px 18px rgba(16,185,129,0.5), inset 0 1px 0 rgba(255,255,255,0.3); }}
.stSelectbox div[data-baseweb="select"] > div {{
    background:{CARD2}; border:1px solid {BORDER}; color:{TEXT}; border-radius:9px;
    box-shadow:inset 0 2px 5px rgba(0,0,0,0.25); }}
.stSlider [data-baseweb="slider"] {{ padding-top:6px; }}
.stRadio label, .stCheckbox label {{ color:{TEXT}; }}
hr {{ border-color:{BORDER}; }}
a {{ color:{ACCENT2}; }}
.sh-row {{ display:flex; justify-content:space-between; padding:4px 0;
          border-bottom:1px solid {BORDER}55; }}
.sh-row .k {{ color:{MUTED}; }} .sh-row .v {{ font-family:Consolas,monospace; }}
/* market segment buttons row a touch bigger */
.stColumns .stButton button {{ font-size:0.92rem; }}
</style>
""", unsafe_allow_html=True)


# ---------- helpers ----------
def pill(text, kind):
    col=COLORMAP.get(kind, NEUTRAL)
    return (f'<span class="sh-pill" style="background:linear-gradient(180deg,{col}33,{col}1a);'
            f'color:{col};border:1px solid {col}88;box-shadow:0 2px 8px {col}33,'
            f'inset 0 1px 0 rgba(255,255,255,0.08);">{text}</span>')

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
    st.markdown(f"<span style='font-size:2rem;font-weight:900;color:{ACCENT};"
                f"text-shadow:0 2px 14px rgba(16,185,129,0.45);letter-spacing:-0.5px'>StockHunter</span> "
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
        is_active = (st.session_state.market == key)
        if st.button(label, key=f"mkt_{key}", use_container_width=True,
                     type=("primary" if is_active else "secondary")):
            # switching market -> wipe ALL leftover search/analysis/widget state from previous market
            for _k in ("analyze_sym","analyze_pick","cmp_rows","cmp_syms","cmp_ai_txt","mf_sel",
                       "scan_perfect","scan_market","scan_dip","mb_result","sector_out",
                       "cmp_slot_0","cmp_slot_1","cmp_slot_2","cmp_slot_3","cmp_slot_4",
                       "fx_sel_India","fx_sel_US","fx_sel_Crypto","fx_sel_Forex","fx_sel_Commodity"):
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

def run_multibagger_scan(count=None):
    try:
        if not count:
            # "Entire NSE" -> download the WHOLE NSE equity list (~2000 stocks)
            full = full_universe()
        else:
            full = active_list()
        universe = full if not count else full[:count]
        cands=[]
        prog=st.progress(0.0, text=f"Scanning {len(universe)} stocks...")
        rl_hits=0
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
            except Exception as e:
                # if Yahoo throttles us, pause briefly and keep going instead of crashing
                if "too many requests" in str(e).lower() or "rate" in str(e).lower():
                    rl_hits+=1
                    import time as _t; _t.sleep(2.0)
            prog.progress(min(1.0,(i+1)/len(universe)),
                          text=f"Scanning... {i+1}/{len(universe)} (found {len(cands)})")
        prog.empty()
        if rl_hits:
            st.info(f"Yahoo throttled {rl_hits} stocks during this scan - they were skipped. "
                    "Run again later to cover them if needed.")
        cands.sort(key=lambda x:(-x["score"],x["mcap"])); top=cands[:25]
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
        st.caption("Tip: tap any stock name to open its full analysis.")
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
    st.caption("Tap any stock name to open its full analysis.")
    cols_o=["Stock","Qty","Buy","LTP","Invested","Value","Wt%","P&L Rs","P&L%","SL","Target","Signal"]
    h=st.columns([1.4,0.7,0.8,0.8,1,1,0.7,1,0.8,0.8,0.8,1.2])
    for i,lbl in enumerate(cols_o):
        h[i].markdown(f"<span class='sh-label' style='font-size:0.7rem'>{lbl}</span>", unsafe_allow_html=True)
    st.markdown(f"<hr style='margin:2px 0 6px 0;border-color:{BORDER}'>", unsafe_allow_html=True)
    for x in r["rows"]:
        rc=st.columns([1.4,0.7,0.8,0.8,1,1,0.7,1,0.8,0.8,0.8,1.2])
        if rc[0].button(x["sym"], key=f"pf_open_{x['sym']}", use_container_width=True, type="secondary"):
            st.session_state.analyze_sym=x["sym"].strip().upper()
            st.session_state["jump_to_analyze"]=True; st.rerun()
        wt = (x['val']/total*100 if total else 0)
        vals=[f"{x['qty']:.0f}",f"{x['buy']:,.0f}",f"{x['ltp']:,.0f}",f"{x['inv']:,.0f}",
              f"{x['val']:,.0f}",f"{wt:.0f}",f"{x['pnl_abs']:+,.0f}",f"{x['pnl_pct']:+.0f}",
              f"{x['sl']:.0f}" if x['sl'] else "-", f"{x['tp']:.0f}" if x['tp'] else "-", x["verdict"]]
        for ci,v in enumerate(vals, start=1):
            rc[ci].markdown(f"<div style='padding-top:6px;font-size:0.85rem'>{v}</div>", unsafe_allow_html=True)
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

    # short note shown at the top of every tab
    TAB_NOTES = {
        T_ANALYZE: "📈 Pick any stock to see its full analysis, chart and a ready trade plan.",
        T_PERFECT: "⭐ Only the stocks that look great to buy right now - strong and trending.",
        T_SCAN:    "🔎 Scan the market and rank every stock from strongest to weakest. Tap one to open it.",
        T_DIP:     "💎 Quality stocks that have cooled off a bit - good spots to buy on a pullback.",
        T_MULTI:   "🚀 AI judges which stocks could grow many times over the long run.",
        T_SECTOR:  "🔄 See which sectors are hot right now and where the money is moving.",
        T_COMPARE: "⚖️ Put 2-5 stocks side by side and let AI pick the better long-term bet.",
        T_WATCH:   "⭐ Your saved stocks with live price, signal and one-tap analysis.",
        T_MACRO:   "🌐 The big picture for the market right now, plus the latest news.",
        T_PF:      "💼 Add your holdings to get an AI review and a full profit/loss table.",
    }
    note = TAB_NOTES.get(cur)
    if note:
        st.markdown(f"<div class='sh-card2' style='padding:9px 16px;margin-bottom:10px'>"
                    f"<span class='sh-muted'>{note}</span></div>", unsafe_allow_html=True)

    # ---------- ANALYZE ----------
    if cur==T_ANALYZE:
        core.set_market(MK)
        # build options: "Company Name (SYMBOL)" for known names + every raw symbol.
        name_map = core.US_NAMES if MK=="US" else core.INDIA_NAMES
        label_to_sym = {}
        opts = ["— search by company name or symbol —"]
        seen=set()
        for nm,sym in sorted(name_map.items(), key=lambda x:x[1]):
            if sym in seen: continue
            label=f"{nm.title()} ({sym})"
            label_to_sym[label]=sym; opts.append(label); seen.add(sym)
        for s in sorted(set(active_list())):
            if s not in seen:
                label_to_sym[s]=s; opts.append(s); seen.add(s)
        c1,c2 = st.columns([5,1])
        with c1:
            pick = st.selectbox("Symbol", opts, index=0, key="analyze_pick",
                                label_visibility="collapsed",
                                help="Type a company name (e.g. Reliance) or symbol (e.g. RELIANCE)")
        with c2:
            go_btn = st.button("Analyze", use_container_width=True, key="analyze_go")
        chosen_sym = label_to_sym.get(pick) if pick and pick!=opts[0] else None
        if chosen_sym and (go_btn or chosen_sym!=st.session_state.get("analyze_sym")):
            st.session_state.analyze_sym = chosen_sym
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
        mc1,mc2 = st.columns([3,2])
        with mc1:
            mb_sym = st.text_input("Company name or symbol to assess", value="", key="mb_sym",
                                   placeholder="e.g. "+("Dixon or DIXON" if MK=="India" else "Palantir or PLTR"))
        with mc2:
            st.write(""); st.write("")
        mb_cnt = scan_depth_picker("multibagger")
        b1,b2 = st.columns(2)
        with b1:
            mb_one = st.button("Assess this stock", key="mb_one", use_container_width=True)
        with b2:
            mb_scan = st.button("🔍 Scan market for multibaggers", key="mb_scan", use_container_width=True)
        if mb_one:
            if not ai_ready(): st.warning("The owner needs to set the Gemini key first.")
            elif not mb_sym.strip(): st.warning("Type a company name or symbol first.")
            else: run_multibagger_one(resolve_symbol(mb_sym.strip()))
        if mb_scan:
            if not ai_ready(): st.warning("The owner needs to set the Gemini key first.")
            else:
                run_multibagger_scan(mb_cnt)
        show_multibagger_result()

    # ---------- PERFECT BUY ----------
    if cur==T_PERFECT:
        cnt = scan_depth_picker("perfect")
        if st.button("Find Perfect Buys", key="perfect_go", type="primary"):
            run_scan_generic("perfect", cnt)
        show_scan_result("perfect")

    # ---------- MARKET SCAN ----------
    if cur==T_SCAN:
        cnt = scan_depth_picker("market")
        if st.button("Scan Market", key="ms_go", type="primary"):
            run_scan_generic("market", cnt)
        show_scan_result("market")

    # ---------- BUY THE DIP ----------
    if cur==T_DIP:
        cnt = scan_depth_picker("dip")
        if st.button("Find Dips", key="dip_go", type="primary"):
            run_scan_generic("dip", cnt)
        show_scan_result("dip")

    # ---------- SECTOR ROTATION ----------
    if cur==T_SECTOR:
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
        st.markdown("Compare 2-5 stocks side by side. Pick each from its dropdown (type a company name or symbol).")
        core.set_market(MK)
        name_map = core.US_NAMES if MK=="US" else core.INDIA_NAMES
        label_to_sym = {}
        opts = ["—"]; seen=set()
        for nm,sym in sorted(name_map.items(), key=lambda x:x[1]):
            if sym in seen: continue
            label=f"{nm.title()} ({sym})"; label_to_sym[label]=sym; opts.append(label); seen.add(sym)
        for s in sorted(set(active_list())):
            if s not in seen:
                label_to_sym[s]=s; opts.append(s); seen.add(s)
        nslots = st.slider("How many stocks to compare?", 2, 5, 3, key="cmp_n")
        slot_cols = st.columns(nslots)
        chosen=[]
        for i in range(nslots):
            with slot_cols[i]:
                pick = st.selectbox(f"Stock {i+1}", opts, key=f"cmp_slot_{i}", index=0)
                if pick and pick!="—":
                    chosen.append(label_to_sym.get(pick, pick))
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
        mcol1,mcol2 = st.columns(2)
        with mcol1:
            if st.button("Get Macro Read (AI)", key="macro_go_btn", use_container_width=True):
                if not ai_ready(): st.warning("The owner needs to set the Gemini key first.")
                else: run_macro()
        with mcol2:
            if st.button("Load Market News", key="macro_news_btn", use_container_width=True):
                load_macro_news()
        if st.session_state.get("macro_txt"):
            render_ai_text(st.session_state["macro_txt"])
        _news = st.session_state.get("macro_news")
        if not isinstance(_news, list): _news = []
        for item in _news:
            try:
                if isinstance(item,(list,tuple)):
                    title=item[0]; pub=item[1] if len(item)>1 else ""; url=item[2] if len(item)>2 else ""
                else:
                    title=str(item); pub=""; url=""
            except Exception:
                continue
            if url: st.markdown(f"- [{title}]({url})  <span class='sh-muted'>({pub})</span>", unsafe_allow_html=True)
            else: st.markdown(f"- {title}  <span class='sh-muted'>({pub})</span>", unsafe_allow_html=True)

    # ---------- PORTFOLIO ----------
    if cur==T_PF:
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

        # confidence + segmented bar (premium look)
        vc=COLORMAP.get(d["vcolor"],MUTED)
        conf=d["confidence"]
        st.markdown(f"<div class='sh-card'><span class='sh-label'>Buy Confidence</span><br>"
            f"<span class='sh-big' style='color:{vc}'>{conf:.0f}%</span> &nbsp;{pill(d['verdict'],d['vcolor'])}"
            f"<div style='margin-top:14px;height:16px;border-radius:9px;overflow:hidden;display:flex;"
            f"box-shadow:inset 0 2px 5px rgba(0,0,0,0.4), 0 1px 0 rgba(255,255,255,0.05);position:relative'>"
            f"<div style='flex:40;background:linear-gradient(180deg,#ff8f8f,{BAD})'></div>"
            f"<div style='flex:30;background:linear-gradient(180deg,#ffd97a,{WARN})'></div>"
            f"<div style='flex:30;background:linear-gradient(180deg,#6ee7b7,{GOOD})'></div>"
            f"<div style='position:absolute;top:0;bottom:0;left:0;width:{min(conf,100)}%;"
            f"background:linear-gradient(90deg,transparent,rgba(255,255,255,0.10));"
            f"border-right:3px solid #fff;box-shadow:0 0 12px rgba(255,255,255,0.6)'></div></div>"
            f"<div style='position:relative;margin-top:6px;margin-left:calc({min(conf,100)}% - 14px)'>"
            f"<span style='color:{vc};font-weight:700;font-size:0.85rem;"
            f"text-shadow:0 0 8px {vc}66'>▲ {conf:.0f}%</span></div>"
            f"<div style='margin-top:8px' class='sh-muted'>Hold: {d['trade_plan']['hold']}</div></div>",
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

        # fundamentals - only show the ones we actually have (skip N/A)
        fkeys = ["Market Cap (Cr)","P/E Ratio","P/B Ratio","ROE %","Debt/Equity","Div Yield %"]
        have = [(k, d["fundamentals"].get(k)) for k in fkeys]
        have = [(k,v) for k,v in have if v not in (None,"","N/A","nan","None")]
        if have:
            fr = "".join(row_html(k, v) for k,v in have)
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
            st.info("AI is not configured yet. The owner needs to set the Gemini key.")
        else:
            f=d["fundamentals"]
            a1,a2,a3=st.columns(3)
            run_deep = a1.button("Generate AI Analysis", key=f"ai_deep_{sym}", use_container_width=True)
            run_rf   = a2.button("🚩 Spot Red Flags", key=f"ai_rf_{sym}", use_container_width=True)
            run_sm   = a3.button("💰 Smart Money Check", key=f"ai_sm_{sym}", use_container_width=True)
            # only feed the AI the numbers we actually have (skip N/A so it doesn't harp on missing data)
            def _has(v): return v not in (None,"","N/A","nan","None")
            parts=[f"Stock: {f.get('Company',sym)} ({sym})"]
            if _has(f.get('Sector')): parts.append(f"sector {f.get('Sector')}")
            parts.append(f"Price {core.CUR}{d['ltp']:.1f}")
            for lbl,key in [("P/E","P/E Ratio"),("P/B","P/B Ratio"),("ROE","ROE %"),
                            ("Debt/Equity","Debt/Equity"),("Div Yield","Div Yield %"),
                            ("Market Cap (Cr)","Market Cap (Cr)")]:
                if _has(f.get(key)): parts.append(f"{lbl} {f.get(key)}")
            parts.append(f"RSI {d['rsi']:.0f}")
            base = ", ".join(parts) + ".\n\n"
            try:
                if run_deep:
                    p=("You are a clear, honest equity analyst for a retail investor. Use clear, simple, plain ENGLISH only (no Hindi or Hinglish). "
                       "No buy/sell command. No intro line, no '---'.\n\n"+base+
                       "Headings: BUSINESS MODEL, FINANCIAL PERFORMANCE, VALUATION, INVESTMENT THESIS. Under 320 words.")
                    with st.spinner("AI reading the numbers..."): st.session_state[f"ai_out_{sym}"]=ai_call(p)
                if run_rf:
                    p=("Act as a sharp forensic accountant / short-seller. Be skeptical but useful. "
                       "Use clear, simple, plain ENGLISH only (no Hindi or Hinglish). No intro line, no '---'. "
                       "Use the numbers below AND your own general knowledge of this well-known company "
                       "(its cash flow track record, debt history, promoter/management reputation, governance). "
                       "Do NOT just say 'data not visible' - give your best informed read and only add "
                       "'verify latest filings' as a short caveat where needed.\n\n"+base+
                       "Headings: ACCOUNTING RED FLAGS, CASH FLOW, DEBT CONCERNS, PROMOTER RISK, GOVERNANCE, VERDICT. Under 300 words.")
                    with st.spinner("Scanning for red flags..."): st.session_state[f"ai_out_{sym}"]=ai_call(p)
                if run_sm:
                    p=("Act as an analyst tracking 'smart money'. Explain what big/informed players appear to be doing "
                       "and whether it signals confidence or concern. "
                       "Use clear, simple, plain ENGLISH only (no Hindi or Hinglish). No intro line, no '---'. "
                       "Use the numbers below AND your general knowledge of this company's typical FII/DII and "
                       "promoter holding pattern. Give a concrete, informed read - do NOT just say 'data not visible'; "
                       "add 'verify latest filings' only as a brief caveat.\n\n"+base+
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

def scan_depth_picker(key):
    """Lets the user choose how many stocks to scan: Quick / Deep / Entire NSE / Custom number.
    Returns the count (None = entire NSE)."""
    choice = st.radio("How many stocks to scan?",
        ["Quick (Top 100)", "Deep (Top 300)", "Entire NSE (everything)", "Custom number"],
        horizontal=True, key=f"depth_{key}")
    if choice.startswith("Entire"):
        st.warning("Entire NSE scans every listed stock (~2000) - this can take 15-30 min "
                   "and Yahoo may throttle some (they'll be skipped). Best run when you have time.")
        return None
    if choice.startswith("Custom"):
        n = st.number_input("Number of stocks to scan", min_value=10, max_value=3000,
                            value=150, step=50, key=f"depthnum_{key}")
        return int(n)
    return 100 if choice.startswith("Quick") else 300

def run_scan_generic(kind, count=100):
    core.set_market(MK)
    if count is None:
        syms = full_universe()                 # whole NSE (every listed stock)
    else:
        syms = active_list()[:count]
    res=[]
    prog=st.progress(0.0, text=f"Scanning {len(syms)} stocks...")
    rl_hits=0
    for i,s in enumerate(syms):
        try:
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
        except Exception as e:
            if "too many requests" in str(e).lower() or "rate" in str(e).lower():
                rl_hits+=1
                import time as _t; _t.sleep(2.0)
        prog.progress(min(1.0,(i+1)/len(syms)),
                      text=f"Scanning... {i+1}/{len(syms)} (found {len(res)})")
    prog.empty()
    if rl_hits:
        st.info(f"Yahoo throttled {rl_hits} stocks during this scan - they were skipped. "
                "Run again later to cover them.")
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
    """Render the table as clickable ROWS - tapping a row (the stock itself) opens
    it in Analyze. No extra buttons, no checkbox column."""
    cols_order = list(df.columns)
    # header row
    head = st.columns([2,1.3,1,1,1.6])
    headers = cols_order[:5] + [""]*(5-len(cols_order))
    for i,h in enumerate(headers):
        head[i].markdown(f"<span class='sh-label'>{h}</span>", unsafe_allow_html=True)
    st.markdown(f"<hr style='margin:2px 0 6px 0;border-color:{BORDER}'>", unsafe_allow_html=True)
    # one clickable button per row, laid out to look like a table row
    for ridx in range(len(df)):
        row = df.iloc[ridx]
        sym = str(row[sym_col])
        rc = st.columns([2,1.3,1,1,1.6])
        # first cell = the stock as a button (this is the click target)
        if rc[0].button(sym, key=f"{key}_row_{sym}_{ridx}", use_container_width=True,
                        type="secondary"):
            st.session_state.analyze_sym = sym.strip().upper()
            st.session_state["jump_to_analyze"] = True
            st.rerun()
        # other cells = plain values aligned next to it
        for ci,colname in enumerate(cols_order[1:5], start=1):
            rc[ci].markdown(f"<div style='padding-top:6px'>{row[colname]}</div>", unsafe_allow_html=True)

def show_scan_result(kind):
    res=st.session_state.get(f"scan_{kind}",[])
    if not res: return
    st.caption("Tip: tap any stock name to open its analysis.")
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
    # expense ratio + plan type
    er = mf.get("expense_est"); plan = mf.get("plan_type")
    if er or plan:
        ec=st.columns(2)
        ec[0].markdown(f"<div class='sh-card2' style='text-align:center'>"
            f"<b style='color:{ACCENT}'>{er or '—'}</b><br><span class='sh-muted'>Expense ratio (est.)</span></div>",
            unsafe_allow_html=True)
        ec[1].markdown(f"<div class='sh-card2' style='text-align:center'>"
            f"<b>{plan or '—'}</b><br><span class='sh-muted'>Plan type</span></div>",
            unsafe_allow_html=True)
        if mf.get("desc"):
            st.caption(mf["desc"])
    # returns - show what we have; a clean dash where a period isn't available yet
    metrics=[("1 Year", mf.get("r1y")), ("3Y/yr", mf.get("r3y")),
             ("5Y/yr", mf.get("r5y")), ("Trend", mf.get("trend"))]
    gc=st.columns(4)
    for i,(lbl,val) in enumerate(metrics):
        if lbl=="Trend":
            disp = val if val else "—"; col=TEXT
        elif val is None:
            disp = "—"; col=MUTED
        else:
            disp = f"{val:+.1f}%"; col = GOOD if val>=0 else BAD
        gc[i].markdown(f"<div class='sh-card2' style='text-align:center'>"
            f"<b style='color:{col}'>{disp}</b><br><span class='sh-muted'>{lbl}</span></div>",
            unsafe_allow_html=True)

    # NAV chart (past NAV history so you can see the trend)
    st.markdown("<span class='sh-label'>NAV history</span>", unsafe_allow_html=True)
    navs_raw = mf.get("navs")
    if HAS_PLOTLY and navs_raw:
        try:
            navs=list(navs_raw)
            # mfapi gives newest first; we want oldest -> newest for a left-to-right chart
            xs=[]; ys=[]
            for n in navs:
                try:
                    d_=n.get("date") if isinstance(n,dict) else None
                    v_=n.get("nav") if isinstance(n,dict) else None
                    if d_ is None or v_ is None: continue
                    dt_=_dt.datetime.strptime(str(d_).strip(),"%d-%m-%Y")
                    val=float(v_)
                    xs.append(dt_); ys.append(val)
                except Exception:
                    continue
            # sort oldest -> newest
            if xs:
                pairs=sorted(zip(xs,ys), key=lambda p:p[0])
                xs=[p[0] for p in pairs]; ys=[p[1] for p in pairs]
            if len(xs)>=2:
                lo=min(ys); span=(max(ys)-lo) or 1
                fig=go.Figure(go.Scatter(x=xs,y=ys,mode="lines",
                    line=dict(color=ACCENT,width=2),fill="tozeroy",
                    fillcolor="rgba(16,185,129,0.12)",name="NAV",
                    hovertemplate="%{x|%d-%b-%Y}<br>NAV: %{y:.2f}<extra></extra>"))
                fig.update_layout(template="plotly_dark",paper_bgcolor=CARD,plot_bgcolor=CARD,height=320,
                    margin=dict(l=10,r=10,t=10,b=10),font=dict(color=TEXT),showlegend=False,
                    hovermode="x unified",hoverlabel=dict(bgcolor=CARD2,bordercolor=BORDER,font=dict(color=TEXT)))
                fig.update_xaxes(gridcolor=BORDER)
                fig.update_yaxes(gridcolor=BORDER, range=[lo-span*0.1, max(ys)+span*0.1])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("Not enough NAV history to draw a chart.")
        except Exception as e:
            st.caption(f"NAV chart could not be drawn ({e}).")
    elif not HAS_PLOTLY:
        st.caption("Install plotly to see the NAV chart.")
    elif not navs_raw:
        st.caption("No NAV history available for this fund.")

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

    # ---- similar funds in the same category, with their returns, for comparison ----
    cat = (mf.get("cat") or "").strip()
    if cat:
        st.markdown(f"<div class='sh-card'><span class='sh-label'>How its competition is performing "
                    f"({cat})</span></div>", unsafe_allow_html=True)
        # find same-category funds (exclude the current one)
        peers=[(n,c) for n,c,k in MUTUAL_FUNDS if (k or "").strip()==cat and n!=mf.get("name")][:5]
        if not peers:
            st.caption("No other funds from this category are in the list to compare.")
        else:
            with st.spinner("Loading peer funds..."):
                rows=[]
                # include the current fund at the top so the user sees where it stands
                rows.append({"Fund":mf.get("name","")+"  (this one)","1Y":mf.get("r1y"),
                             "3Y/yr":mf.get("r3y"),"5Y/yr":mf.get("r5y"),
                             "Expense":mf.get("expense_est","—")})
                for n,c in peers:
                    pm=cached_mf(c)
                    if pm and not pm.get("error"):
                        rows.append({"Fund":n,"1Y":pm.get("r1y"),"3Y/yr":pm.get("r3y"),
                                     "5Y/yr":pm.get("r5y"),"Expense":pm.get("expense_est","—")})
            def _f(v): return f"{v:+.1f}%" if isinstance(v,(int,float)) else "—"
            st.caption("Tap any fund name to open it. Higher 3Y/5Y returns with a lower expense ratio "
                       "is usually the better long-term pick.")
            # header
            h=st.columns([2.4,1,1,1,1.3])
            for i,lbl in enumerate(["Fund","1Y","3Y/yr","5Y/yr","Expense (est.)"]):
                h[i].markdown(f"<span class='sh-label'>{lbl}</span>", unsafe_allow_html=True)
            st.markdown(f"<hr style='margin:2px 0 6px 0;border-color:{BORDER}'>", unsafe_allow_html=True)
            # current fund first (not clickable - already open), then clickable peers
            cur_row=rows[0]
            cr=st.columns([2.4,1,1,1,1.3])
            cr[0].markdown(f"<div style='padding-top:6px'><b style='color:{ACCENT}'>{cur_row['Fund']}</b></div>", unsafe_allow_html=True)
            cr[1].markdown(f"<div style='padding-top:6px'>{_f(cur_row['1Y'])}</div>", unsafe_allow_html=True)
            cr[2].markdown(f"<div style='padding-top:6px'>{_f(cur_row['3Y/yr'])}</div>", unsafe_allow_html=True)
            cr[3].markdown(f"<div style='padding-top:6px'>{_f(cur_row['5Y/yr'])}</div>", unsafe_allow_html=True)
            cr[4].markdown(f"<div style='padding-top:6px'>{cur_row['Expense']}</div>", unsafe_allow_html=True)
            # peer rows - fund name is the clickable button
            peer_map={n:c for n,c in peers}
            for r in rows[1:]:
                rc=st.columns([2.4,1,1,1,1.3])
                fname=r["Fund"]
                if rc[0].button(fname, key=f"peer_{peer_map.get(fname,fname)}", use_container_width=True, type="secondary"):
                    pm=cached_mf(peer_map.get(fname))
                    if pm and not pm.get("error"):
                        pm["name"]=fname; pm["cat"]=cat
                        st.session_state.mf_sel=pm; st.rerun()
                    else:
                        st.warning(f"Could not load {fname} right now - try again.")
                rc[1].markdown(f"<div style='padding-top:6px'>{_f(r['1Y'])}</div>", unsafe_allow_html=True)
                rc[2].markdown(f"<div style='padding-top:6px'>{_f(r['3Y/yr'])}</div>", unsafe_allow_html=True)
                rc[3].markdown(f"<div style='padding-top:6px'>{_f(r['5Y/yr'])}</div>", unsafe_allow_html=True)
                rc[4].markdown(f"<div style='padding-top:6px'>{r['Expense']}</div>", unsafe_allow_html=True)


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
