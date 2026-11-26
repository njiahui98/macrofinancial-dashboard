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
