import io
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import requests

st.set_page_config(page_title="Macrofinancial Risk Monitor", page_icon="📈", layout="wide")

# FRED's public graph CSV endpoint does not require an API key.
# The S&P 500 FRED series is limited to a rolling history.
CATALOG = {
    "US 2Y Treasury": ("United States", "Rates", "DGS2", "rate"),
    "US 5Y Treasury": ("United States", "Rates", "DGS5", "rate"),
    "US 10Y Treasury": ("United States", "Rates", "DGS10", "rate"),
    "US 30Y Treasury": ("United States", "Rates", "DGS30", "rate"),
    "US IG OAS": ("United States", "Credit", "BAMLC0A0CM", "rate"),
    "US BBB OAS": ("United States", "Credit", "BAMLC0A4CBBB", "rate"),
    "US HY OAS": ("United States", "Credit", "BAMLH0A0HYM2", "rate"),
    "VIX": ("United States", "Volatility", "VIXCLS", "index"),
    "S&P 500": ("United States", "Equities", "SP500", "index"),
    "USD/SGD": ("Singapore", "FX", "DEXSIUS", "fx"),
    "USD/CNY": ("China", "FX", "DEXCHUS", "fx"),
    "USD/JPY": ("Japan", "FX", "DEXJPUS", "fx"),
    "EUR/USD": ("Euro area", "FX", "DEXUSEU", "fx"),
    "GBP/USD": ("United Kingdom", "FX", "DEXUSUK", "fx"),
}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred(code):
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    r = requests.get(url, params={"id": code}, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    frame = pd.read_csv(io.StringIO(r.text))
    if frame.shape[1] < 2:
        raise ValueError("FRED returned no usable observations")
    dates = pd.to_datetime(frame.iloc[:, 0], errors="coerce")
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
    s = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(dates), name=code)
    return s.loc[~s.index.isna()].dropna().sort_index()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all():
    series, failures = {}, {}
    for name, (_, _, code, _) in CATALOG.items():
        try:
            series[name] = fetch_fred(code).rename(name)
        except Exception as exc:
            failures[name] = str(exc)
    return series, failures

# Upload a CSV exported from an official MAS source. The app does not assume
# a particular MAS API schema or invent a 10Y SGS series from an unrelated rate.
def read_sg_csv(upload):
    df = pd.read_csv(upload)
    if df.empty:
        raise ValueError("CSV has no rows")
    return df

def normalise_uploaded(df, date_col, value_col, label):
    dates = pd.to_datetime(df[date_col], errors="coerce")
    values = pd.to_numeric(df[value_col].astype(str).str.replace(",", "", regex=False), errors="coerce")
    s = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(dates), name=label)
    s = s.loc[~s.index.isna()].dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if s.empty:
        raise ValueError("No valid date/value observations")
    return s

def asof_change(s, days, kind):
    s = s.dropna().sort_index()
    if len(s) < 2:
        return np.nan
    latest_date = s.index[-1]
    target = latest_date - pd.Timedelta(days=days)
    earlier = s.loc[s.index <= target]
    if earlier.empty:
        return np.nan
    # Avoid silently treating a very stale release as a current daily observation.
    max_gap = 10 if kind in ("rate", "fx", "index") else 45
    if (target - earlier.index[-1]).days > max_gap:
        return np.nan
    current, prior = s.iloc[-1], earlier.iloc[-1]
    if kind == "rate":
        return (current - prior) * 100  # bp
    if kind == "fx":
        return (current / prior - 1) * 100 if prior else np.nan
    return (current / prior - 1) * 100 if prior else np.nan

def rolling_changes(s, days, kind):
    s = s.dropna().sort_index()
    if s.empty:
        return pd.Series(dtype=float)
    now = pd.DataFrame({"date": s.index, "current": s.to_numpy()})
    past = pd.DataFrame({"past_date": s.index, "prior": s.to_numpy()})
    now["target"] = now["date"] - pd.Timedelta(days=days)
    joined = pd.merge_asof(now.sort_values("target"), past, left_on="target", right_on="past_date", direction="backward")
    joined = joined.sort_values("date")
    gap = (joined["target"] - joined["past_date"]).dt.days
    joined.loc[gap > 10, "prior"] = np.nan
    if kind == "rate":
        values = (joined["current"] - joined["prior"]) * 100
    else:
        values = (joined["current"] / joined["prior"] - 1) * 100
    return pd.Series(values.to_numpy(), index=pd.DatetimeIndex(joined["date"])).replace([np.inf, -np.inf], np.nan).dropna()

st.title("Macrofinancial Risk Monitor")
st.caption("Free public-data monitor · FRED market series · optional official Singapore data uploads")

with st.sidebar:
    st.header("Coverage")
    selected_countries = st.multiselect("Economies", ["Singapore", "United States", "China", "Japan", "Euro area", "United Kingdom"], default=["Singapore", "United States"])
    selected_categories = st.multiselect("Asset classes", ["Rates", "Credit", "FX", "Equities", "Volatility"], default=["Rates", "Credit", "FX", "Equities", "Volatility"])
    st.divider()
    st.subheader("Singapore SGS / SORA (optional)")
    st.caption("Upload public official data as CSV. Format: a date column and one numeric value column; yield/rate levels should be in percent, e.g. 2.75, not 0.0275.")
    upload = st.file_uploader("Upload CSV", type="csv")
    sg_name = st.text_input("Series name", "SG 10Y SGS")

with st.spinner("Loading public market data..."):
    series, failures = fetch_all()
meta = {name: {"country": country, "category": category, "kind": kind, "source": "FRED"} for name, (country, category, _, kind) in CATALOG.items() if name in series}

if upload is not None:
    try:
        raw = read_sg_csv(upload)
        cols = list(raw.columns)
        with st.sidebar:
            date_col = st.selectbox("Date column", cols, index=0)
            value_col = st.selectbox("Value column", cols, index=min(1, len(cols)-1))
        if sg_name.strip() and sg_name not in series:
            series[sg_name] = normalise_uploaded(raw, date_col, value_col, sg_name)
            meta[sg_name] = {"country": "Singapore", "category": "Rates", "kind": "rate", "source": "User-uploaded public CSV"}
    except Exception as exc:
        st.sidebar.error(f"Could not read CSV: {exc}")

# Derived rates: percentage-point difference between two yield levels.
if "US 10Y Treasury" in series and "US 2Y Treasury" in series:
    series["US 10Y–2Y curve"] = (series["US 10Y Treasury"] - series["US 2Y Treasury"]).dropna()
    meta["US 10Y–2Y curve"] = {"country": "United States", "category": "Rates", "kind": "rate", "source": "Calculated from FRED"}
if "US HY OAS" in series and "US IG OAS" in series:
    series["US HY–IG OAS gap"] = (series["US HY OAS"] - series["US IG OAS"]).dropna()
    meta["US HY–IG OAS gap"] = {"country": "United States", "category": "Credit", "kind": "rate", "source": "Calculated from FRED"}

if failures:
    with st.expander(f"Data retrieval warnings ({len(failures)})"):
        for name, message in failures.items():
            st.write(f"{name}: {message}")

names = [n for n, m in meta.items() if m["country"] in selected_countries and m["category"] in selected_categories]
if not names:
    st.info("No series match your filters. Select another economy or category, or upload Singapore rate data.")
    st.stop()

periods = {"1D": 1, "1W": 7, "1M": 30, "3M": 90, "1Y": 365}
rows = []
for name in names:
    s = series[name].dropna()
    if s.empty:
        continue
    m = meta[name]
    row = {"Economy": m["country"], "Category": m["category"], "Indicator": name, "Latest": round(float(s.iloc[-1]), 4), "As of": s.index[-1].date().isoformat(), "Unit of change": "bp" if m["kind"] == "rate" else "%", "Source": m["source"]}
    for p, d in periods.items():
        row[p] = asof_change(s, d, m["kind"])
    rows.append(row)
summary = pd.DataFrame(rows)

market_tab, chart_tab, stress_tab, data_tab = st.tabs(["Market Overview", "Historical Charts", "Stress Benchmarking", "Data & Sources"])
with market_tab:
    st.subheader("Latest market movements")
    st.caption("Rate and spread changes in basis points; FX and index changes in percent. Latest levels are in their original source units (rates and OAS in %, VIX in points). Each series uses its own latest observation date.")
    display = summary.drop(columns=["Source"]).copy()
    st.dataframe(display, hide_index=True, use_container_width=True)
    st.download_button("Download market movements CSV", summary.to_csv(index=False), file_name="macro_market_movements.csv", mime="text/csv")
    st.info("Singapore SGS and SORA are not auto-populated in this version. Upload an official public CSV using the sidebar to add them; the FRED USD/SGD series is already available.")

with chart_tab:
    st.subheader("Historical series")
    selected = st.multiselect("Indicators", names, default=names[:1], key="chart_names")
    window = st.selectbox("Chart period", ["3M", "1Y", "5Y", "10Y", "All"], index=2)
    rebase = st.checkbox("Rebase to 100 (compare relative changes; suitable for positive-valued levels)", value=False)
    if selected:
        frames = []
        for n in selected:
            s = series[n].dropna()
            if window != "All":
                days = {"3M": 90, "1Y": 365, "5Y": 1825, "10Y": 3650}[window]
                s = s.loc[s.index >= s.index.max() - pd.Timedelta(days=days)]
            if rebase:
                if s.empty or s.iloc[0] <= 0 or (s <= 0).any():
                    st.warning(f"Cannot rebase {n}: series includes non-positive values.")
                    continue
                s = s / s.iloc[0] * 100
            frames.append(pd.DataFrame({"Date": s.index, "Value": s.to_numpy(), "Indicator": n}))
        if frames:
            plot = pd.concat(frames, ignore_index=True)
            fig = px.line(plot, x="Date", y="Value", color="Indicator", title="Indexed to 100" if rebase else "Historical levels", template="plotly_white")
            fig.update_layout(hovermode="x unified", height=520)
            st.plotly_chart(fig, use_container_width=True)
            if not rebase and len(selected) > 1:
                st.caption("Different indicators may have incompatible units. Use one series at a time, or rebase compatible positive-valued series to 100.")

with stress_tab:
    st.subheader("Historical movement benchmarking")
    indicator = st.selectbox("Indicator", names)
    horizon = st.selectbox("Change horizon", ["1M", "3M", "1Y"], index=1)
    days = {"1M": 30, "3M": 90, "1Y": 365}[horizon]
    changes = rolling_changes(series[indicator], days, meta[indicator]["kind"])
    if len(changes) < 20:
        st.warning("Insufficient observations for a meaningful distribution.")
    else:
        latest = float(changes.iloc[-1])
        percentile = float((changes <= latest).mean() * 100)
        unit = "bp" if meta[indicator]["kind"] == "rate" else "%"
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Latest {horizon} change", f"{latest:+.2f} {unit}")
        c2.metric("Signed historical percentile", f"{percentile:.1f}%")
        c3.metric("Historical observations", f"{len(changes):,}")
        fig = px.histogram(x=changes.to_numpy(), nbins=60, labels={"x": f"{horizon} change ({unit})"}, title=f"{indicator}: historical {horizon} changes")
        fig.add_vline(x=latest, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Percentiles describe the historical distribution of overlapping observations, not forward-looking probabilities. A high signed percentile is not always adverse. Series coverage differs by indicator.")

with data_tab:
    st.subheader("Data availability and sources")
    availability = pd.DataFrame([{"Indicator": n, "Economy": meta[n]["country"], "Source": meta[n]["source"], "First observation": s.index.min().date().isoformat(), "Latest observation": s.index.max().date().isoformat(), "Count": len(s)} for n, s in series.items() if not s.empty])
    st.dataframe(availability, hide_index=True, use_container_width=True)
    st.markdown("**FRED:** https://fred.stlouisfed.org/  \\n**MAS statistics:** https://www.mas.gov.sg/statistics  \\n**MAS SGS information:** https://www.mas.gov.sg/bonds-and-bills/singapore-government-securities")
    st.caption("FRED FX series are expressed as stated in the indicator label (e.g. USD/SGD is SGD per USD). The dashboard does not assume that cross-country data are equally timely or directly comparable.")
    all_data = pd.concat(series, axis=1).sort_index()
    st.download_button("Download all loaded observations", all_data.to_csv(index_label="Date"), file_name="macrofinancial_observations.csv", mime="text/csv")

st.divider()
st.caption("Research prototype. Verify series definitions, units, availability, and any restrictions before using for official work. Do not upload confidential or proprietary data to a public cloud app.")
