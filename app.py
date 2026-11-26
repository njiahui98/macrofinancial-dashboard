
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from io import StringIO

st.set_page_config(
    page_title="Macrofinancial Risk Monitor",
    page_icon="📈",
    layout="wide"
)

SERIES = {
    "US 2Y Yield": ("DGS2", "bp"),
    "US 5Y Yield": ("DGS5", "bp"),
    "US 10Y Yield": ("DGS10", "bp"),
    "US 30Y Yield": ("DGS30", "bp"),
    "US IG OAS": ("BAMLC0A0CM", "bp"),
    "US HY OAS": ("BAMLH0A0HYM2", "bp"),
    "US BBB OAS": ("BAMLC0A4CBBB", "bp"),
    "VIX": ("VIXCLS", "points"),
    "S&P 500": ("SP500", "%"),
    "USD/SGD": ("DEXSIUS", "%"),
}

@st.cache_data(ttl=3600)
def load_data():
    series_list = []

    for name, (code, unit) in SERIES.items():
        try:
            url = (
                "https://fred.stlouisfed.org/"
                f"graph/fredgraph.csv?id={code}"
            )
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            df = pd.read_csv(StringIO(response.text))
            dates = pd.to_datetime(df.iloc[:, 0])
            values = pd.to_numeric(
                df.iloc[:, 1], errors="coerce"
            )

            s = pd.Series(
                values.to_numpy(),
                index=pd.DatetimeIndex(dates),
                name=name
            )
            series_list.append(s)

        except Exception as e:
            st.warning(f"Could not load {name}: {e}")

    if not series_list:
        return pd.DataFrame()

    return pd.concat(series_list, axis=1).sort_index()


st.title("Macrofinancial Risk Monitor")
st.caption("Public market data | Source: FRED")

with st.spinner("Retrieving market data..."):
    data = load_data()

if data.empty:
    st.error("No data available. Check FRED connectivity.")
    st.stop()

tab1, tab2, tab3 = st.tabs([
    "Market Overview",
    "Historical Charts",
    "Stress Benchmarking"
])

periods = {
    "1D": 1,
    "1W": 7,
    "1M": 30,
    "3M": 90,
    "1Y": 365
}

def historical_change(s, days, unit):
    s = s.dropna().sort_index()
    if s.empty:
        return None

    latest_date = s.index[-1]
    target_date = latest_date - pd.Timedelta(days=days)

    previous = s.loc[s.index <= target_date]
    if previous.empty:
        return None

    current = s.iloc[-1]
    old = previous.iloc[-1]

    if unit == "bp":
        return (current - old) * 100
    if unit == "points":
        return current - old
    if old == 0:
        return None
    return (current / old - 1) * 100


with tab1:
    st.subheader("Latest Market Movements")

    rows = []

    for name, (_, unit) in SERIES.items():
        if name not in data:
            continue

        s = data[name].dropna()
        if s.empty:
            continue

        row = {
            "Indicator": name,
            "Latest": round(s.iloc[-1], 3),
            "As of": s.index[-1].strftime("%Y-%m-%d"),
            "Change unit": unit
        }

        for label, days in periods.items():
            value = historical_change(s, days, unit)
            row[label] = (
                round(value, 2)
                if value is not None else None
            )

        rows.append(row)

    summary = pd.DataFrame(rows)
    st.dataframe(summary, hide_index=True, width="stretch")

    st.caption(
        "Yield and credit spread changes are in bp; "
        "VIX changes are in index points; "
        "equity and FX changes are percentages. "
        "Lookbacks use calendar days."
    )

    st.download_button(
        "Download market movements",
        summary.to_csv(index=False),
        "market_movements.csv",
        "text/csv"
    )


with tab2:
    st.subheader("Historical Trends")

    indicator = st.selectbox(
        "Select indicator",
        list(data.columns)
    )

    lookback = st.selectbox(
        "Chart period",
        ["3M", "1Y", "5Y", "10Y", "All"],
        index=2
    )

    s = data[indicator].dropna()

    if lookback != "All":
        days = {
            "3M": 90,
            "1Y": 365,
            "5Y": 365 * 5,
            "10Y": 365 * 10
        }[lookback]

        s = s.loc[
            s.index >= s.index[-1] - pd.Timedelta(days=days)
        ]

    chart_data = pd.DataFrame({
        "Date": s.index,
        "Value": s.to_numpy()
    })

    fig = px.line(
        chart_data,
        x="Date",
        y="Value",
        title=indicator,
        template="plotly_white"
    )
    fig.update_layout(
        hovermode="x unified",
        height=500
    )

    st.plotly_chart(fig, width="stretch")


with tab3:
    st.subheader("Historical Stress Benchmarking")

    indicator = st.selectbox(
        "Indicator to benchmark",
        list(data.columns),
        key="stress_indicator"
    )

    horizon = st.selectbox(
        "Movement horizon",
        ["1M", "3M", "1Y"],
        index=1
    )

    days = {"1M": 30, "3M": 90, "1Y": 365}[horizon]

    s = data[indicator].dropna()
    unit = SERIES[indicator][1]

    # Use an as-of observation for calendar lookbacks.
    past = pd.merge_asof(
        pd.DataFrame({
            "Date": s.index,
            "Target": s.index - pd.Timedelta(days=days)
        }).sort_values("Target"),
        pd.DataFrame({
            "PastDate": s.index,
            "PastValue": s.to_numpy()
        }),
        left_on="Target",
        right_on="PastDate",
        direction="backward"
    ).sort_values("Date")

    current = pd.Series(s.to_numpy(), index=s.index)
    previous = pd.Series(
        past["PastValue"].to_numpy(),
        index=pd.DatetimeIndex(past["Date"])
    )

    if unit == "bp":
        changes = (current - previous) * 100
    elif unit == "points":
        changes = current - previous
    else:
        changes = (current / previous - 1) * 100

    changes = changes.replace(
        [float("inf"), float("-inf")], float("nan")
    ).dropna()

    if changes.empty:
        st.warning("Insufficient history for benchmarking.")
    else:
        latest = changes.iloc[-1]
        percentile = (changes <= latest).mean() * 100

        c1, c2 = st.columns(2)
        c1.metric(
            f"Latest {horizon} change",
            f"{latest:.2f} {unit}"
        )
        c2.metric(
            "Historical percentile",
            f"{percentile:.1f}%"
        )

        fig = px.histogram(
            x=changes.to_numpy(),
            nbins=60,
            labels={"x": f"Change ({unit})"},
            title=f"{indicator}: {horizon} changes"
        )
        fig.add_vline(
            x=latest,
            line_dash="dash",
            line_color="red"
        )

        st.plotly_chart(fig, width="stretch")

        st.caption(
            "Percentiles are descriptive, based on "
            "overlapping historical observations. "
            "They are not probabilities of future stress."
        )

st.divider()
st.caption(
    "Data are provided for monitoring and research. "
    "Latest observations may be delayed or revised."
)
