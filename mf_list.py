# ============================================================
#   Mutual Funds list (Indian MFs via mfapi.in - free AMFI data)
#   Each entry: (Display name, mfapi scheme code, category)
#   Includes India equity, debt, hybrid, index, and the
#   India-domiciled funds that invest in US / global markets
#   (these are how an Indian investor gets US exposure).
#   Scheme codes are AMFI codes used by https://www.mfapi.in/
# ============================================================

MUTUAL_FUNDS = [
    # ---------- Large Cap ----------
    ("Nippon India Large Cap Fund - Direct Growth", "118632", "Large Cap"),
    ("ICICI Pru Bluechip Fund - Direct Growth", "120586", "Large Cap"),
    ("Axis Bluechip Fund - Direct Growth", "120465", "Large Cap"),
    ("Mirae Asset Large Cap Fund - Direct Growth", "118825", "Large Cap"),
    ("SBI Bluechip Fund - Direct Growth", "119598", "Large Cap"),
    ("HDFC Top 100 Fund - Direct Growth", "118989", "Large Cap"),
    ("Kotak Bluechip Fund - Direct Growth", "112942", "Large Cap"),

    # ---------- Mid Cap ----------
    ("Motilal Oswal Midcap Fund - Direct Growth", "127039", "Mid Cap"),
    ("Kotak Emerging Equity Fund - Direct Growth", "112943", "Mid Cap"),
    ("PGIM India Midcap Opp Fund - Direct Growth", "135803", "Mid Cap"),
    ("HDFC Mid-Cap Opportunities - Direct Growth", "118989", "Mid Cap"),
    ("Nippon India Growth Fund - Direct Growth", "118649", "Mid Cap"),
    ("Quant Mid Cap Fund - Direct Growth", "120841", "Mid Cap"),

    # ---------- Small Cap ----------
    ("Nippon India Small Cap Fund - Direct Growth", "118778", "Small Cap"),
    ("Quant Small Cap Fund - Direct Growth", "120828", "Small Cap"),
    ("Axis Small Cap Fund - Direct Growth", "125354", "Small Cap"),
    ("SBI Small Cap Fund - Direct Growth", "125497", "Small Cap"),
    ("HDFC Small Cap Fund - Direct Growth", "130503", "Small Cap"),
    ("Tata Small Cap Fund - Direct Growth", "143278", "Small Cap"),

    # ---------- Flexi / Multi Cap ----------
    ("Parag Parikh Flexi Cap Fund - Direct Growth", "122639", "Flexi Cap"),
    ("Quant Flexi Cap Fund - Direct Growth", "120843", "Flexi Cap"),
    ("HDFC Flexi Cap Fund - Direct Growth", "119010", "Flexi Cap"),
    ("Kotak Flexicap Fund - Direct Growth", "112932", "Flexi Cap"),
    ("Quant Active Fund - Direct Growth", "120823", "Multi Cap"),

    # ---------- Index Funds ----------
    ("UTI Nifty 50 Index Fund - Direct Growth", "120716", "Index"),
    ("HDFC Index Nifty 50 - Direct Growth", "119063", "Index"),
    ("ICICI Pru Nifty 50 Index - Direct Growth", "120620", "Index"),
    ("Motilal Oswal Nifty Midcap 150 Index - Direct", "147625", "Index"),
    ("UTI Nifty Next 50 Index Fund - Direct Growth", "125354", "Index"),

    # ---------- ELSS (Tax Saver) ----------
    ("Quant ELSS Tax Saver Fund - Direct Growth", "120847", "ELSS"),
    ("Mirae Asset ELSS Tax Saver - Direct Growth", "135781", "ELSS"),
    ("Parag Parikh ELSS Tax Saver - Direct Growth", "139655", "ELSS"),

    # ---------- Hybrid / Balanced ----------
    ("ICICI Pru Equity & Debt Fund - Direct Growth", "120251", "Hybrid"),
    ("HDFC Balanced Advantage - Direct Growth", "119025", "Hybrid"),
    ("Quant Multi Asset Fund - Direct Growth", "120824", "Hybrid"),

    # ---------- US / Global investing (India-domiciled) ----------
    ("Motilal Oswal Nasdaq 100 FOF - Direct Growth", "147794", "US/Global"),
    ("Franklin India Feeder US Opp - Direct Growth", "118527", "US/Global"),
    ("Edelweiss US Technology FOF - Direct Growth", "148918", "US/Global"),
    ("ICICI Pru US Bluechip Equity - Direct Growth", "120595", "US/Global"),
    ("Nippon India US Equity Opp - Direct Growth", "135945", "US/Global"),
    ("DSP US Flexible Equity FOF - Direct Growth", "130914", "US/Global"),
    ("PGIM India Global Equity Opp - Direct Growth", "120179", "US/Global"),
    ("Motilal Oswal S&P 500 Index - Direct Growth", "147845", "US/Global"),
]
