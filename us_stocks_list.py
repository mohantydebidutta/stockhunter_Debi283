# ============================================================
#   US Stock List - S&P 500 top liquid companies
#   Used by both StockHunter desktop and web apps.
#   Symbols are plain US tickers (no suffix) - Yahoo Finance
#   serves these directly (e.g. AAPL, MSFT, TSLA).
# ============================================================

US_STOCKS = [
    # Mega-cap tech
    "AAPL","MSFT","GOOGL","GOOG","AMZN","NVDA","META","TSLA","AVGO","ORCL",
    "ADBE","CRM","AMD","INTC","CSCO","QCOM","TXN","INTU","IBM","NOW",
    "AMAT","MU","ADI","LRCX","KLAC","SNPS","CDNS","MRVL","FTNT","PANW",
    "ANET","ROP","ADSK","NXPI","MCHP","ON","CTSH","IT","ACN","HPQ",
    "DELL","WDC","STX","HPE","KEYS","GLW","TEL","APH","CDW","ZBRA",
    # Communication / media
    "NFLX","DIS","CMCSA","T","VZ","TMUS","CHTR","WBD","PARA","FOXA",
    "EA","TTWO","OMC","IPG","LYV","MTCH","NWSA",
    # Consumer discretionary
    "HD","MCD","NKE","LOW","SBUX","TJX","BKNG","MAR","CMG","ORLY",
    "AZO","ROST","YUM","HLT","GM","F","APTV","LULU","DHI","LEN",
    "NVR","PHM","GRMN","EBAY","ETSY","DPZ","DRI","EXPE","RCL","CCL",
    "NCLH","WYNN","LVS","MGM","KMX","BBY","ULTA","TSCO","POOL","WHR",
    # Consumer staples
    "WMT","PG","KO","PEP","COST","MDLZ","PM","MO","CL","KMB",
    "GIS","KHC","HSY","STZ","KDP","MNST","SYY","ADM","KR","DG",
    "DLTR","WBA","CLX","CHD","MKC","HRL","CAG","CPB","TSN","TAP",
    # Healthcare
    "UNH","JNJ","LLY","ABBV","MRK","PFE","TMO","ABT","DHR","BMY",
    "AMGN","MDT","ISRG","SYK","GILD","CVS","CI","ELV","VRTX","REGN",
    "ZTS","BSX","BDX","HCA","HUM","CNC","MCK","BIIB","IDXX","IQV",
    "DXCM","EW","RMD","MTD","WST","BAX","ALGN","HOLX","COO","STE",
    "MRNA","CTLT","DVA","UHS","CRL","TECH","PODD","RVTY",
    # Financials
    "BRK-B","JPM","V","MA","BAC","WFC","GS","MS","AXP","SCHW",
    "C","BLK","SPGI","CB","PGR","MMC","ICE","CME","AON","PNC",
    "USB","TFC","COF","BK","AIG","MET","PRU","TRV","AFL","ALL",
    "MSCI","AJG","FIS","FISV","GPN","PYPL","DFS","NDAQ","STT","FITB",
    "HBAN","RF","CFG","KEY","MTB","SYF","NTRS","WTW","BRO","CINF",
    # Industrials
    "CAT","HON","UPS","BA","GE","RTX","LMT","DE","UNP","ETN",
    "ADP","NOC","GD","CSX","NSC","EMR","FDX","ITW","WM","MMM",
    "PH","CMI","GWW","PCAR","ROK","CARR","OTIS","PAYX","FAST","AME",
    "CTAS","DOV","IR","XYL","EFX","VRSK","BR","ODFL","WAB","URI",
    "LHX","TDG","TT","IEX","NDSN","PWR","HWM","AXON","JCI","SWK",
    # Energy
    "XOM","CVX","COP","SLB","EOG","MPC","PSX","VLO","OXY","WMB",
    "KMI","HAL","DVN","HES","FANG","BKR","OKE","TRGP","CTRA","MRO",
    # Materials
    "LIN","APD","SHW","ECL","FCX","NEM","DOW","DD","NUE","CTVA",
    "VMC","MLM","PPG","ALB","IFF","LYB","STLD","CF","MOS","FMC",
    "PKG","IP","AVY","BALL","AMCR","CE","EMN","CELH",
    # Utilities
    "NEE","DUK","SO","D","AEP","SRE","EXC","XEL","PEG","ED",
    "WEC","ES","AEE","DTE","ETR","FE","PPL","CMS","CNP","ATO",
    "AES","LNT","NI","EVRG","PNW",
    # Real estate
    "PLD","AMT","EQIX","CCI","PSA","O","WELL","SPG","DLR","SBAC",
    "VICI","AVB","EQR","EXR","ARE","INVH","MAA","UDR","CPT","KIM",
    "REG","HST","BXP","FRT","VTR",
]

# de-dupe, keep order
_seen = set(); _clean = []
for _s in US_STOCKS:
    if _s not in _seen:
        _seen.add(_s); _clean.append(_s)
US_STOCKS = _clean
