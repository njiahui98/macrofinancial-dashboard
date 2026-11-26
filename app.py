"""Streamlit macrofinancial dashboard reading committed public CSV snapshots."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from update_data import CATALOG

st.set_page_config(page_title='Macrofinancial Risk Monitor', page_icon='📈', layout='wide')
DATA = Path(__file__).resolve().parent / 'data'

@st.cache_data(ttl=600)
def load_snapshots():
    series, errors = {}, {}
    for name, (code, country, category, kind) in CATALOG.items():
        path = DATA / f'{code}.csv'
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            dates = pd.to_datetime(df['Date'], errors='coerce')
            values = pd.to_numeric(df['Value'], errors='coerce')
            s = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(dates), name=name)
            s = s.loc[~s.index.isna()].dropna().sort_index()
            s = s.loc[~s.index.duplicated(keep='last')]
            if not s.empty:
                series[name] = s
        except Exception as exc:
            errors[name] = str(exc)
    return series, errors

def change(s, days, kind):
    s = s.dropna().sort_index()
    if len(s) < 2:
        return np.nan
    target = s.index[-1] - pd.Timedelta(days=days)
    prior = s.loc[s.index <= target]
    if prior.empty or (target - prior.index[-1]).days > 10:
        return np.nan
    a, b = s.iloc[-1], prior.iloc[-1]
    if kind == 'rate':
        return (a-b)*100
    return (a/b-1)*100 if b else np.nan

def rolling(s, days, kind):
    s = s.dropna().sort_index()
    if len(s) < 20:
        return pd.Series(dtype=float)
    current = pd.DataFrame({'Date':s.index, 'Value':s.to_numpy()})
    current['Target'] = current['Date'] - pd.Timedelta(days=days)
    previous = current[['Date','Value']].rename(columns={'Date':'PriorDate','Value':'PriorValue'})
    merged = pd.merge_asof(current.sort_values('Target'), previous, left_on='Target', right_on='PriorDate', direction='backward').sort_values('Date')
    merged.loc[(merged['Target']-merged['PriorDate']).dt.days > 10,'PriorValue'] = np.nan
    if kind == 'rate':
        v = (merged['Value']-merged['PriorValue'])*100
    else:
        v = (merged['Value']/merged['PriorValue']-1)*100
    return pd.Series(v.to_numpy(),index=pd.DatetimeIndex(merged['Date'])).replace([np.inf,-np.inf],np.nan).dropna()

st.title('Macrofinancial Risk Monitor')
st.caption('Public-data snapshots · FRED · updated by GitHub Actions · no live API requests from Streamlit')
with st.sidebar:
    st.header('Coverage')
    economies = st.multiselect('Economies', ['Singapore','United States','China','Japan','Euro area','United Kingdom'], default=['Singapore','United States'])
    categories = st.multiselect('Asset classes', ['Rates','Credit','FX','Equities','Volatility'], default=['Rates','Credit','FX','Equities','Volatility'])
    if st.button('Reload stored data'):
        load_snapshots.clear()
        st.rerun()
    st.caption('New data appear after the scheduled GitHub workflow commits its CSV files and Streamlit redeploys.')

series, errors = load_snapshots()
if errors:
    with st.expander(f'CSV reading warnings ({len(errors)})'):
        st.json(errors)
status_path = DATA / 'update_status.json'
if status_path.exists():
    try:
        status = json.loads(status_path.read_text(encoding='utf-8'))
        st.caption(f"Last refresh attempt (UTC): {status.get('attempt_utc', 'unknown')} · Successful downloads: {status.get('successful', 'unknown')}")
        if status.get('failed'):
            with st.expander('Latest update failures'):
                st.json(status['failed'])
    except Exception:
        pass

meta = {n: {'country':c,'category':a,'kind':k,'source':'FRED snapshot'} for n, (code,c,a,k) in CATALOG.items() if n in series}
if 'US 10Y Treasury' in series and 'US 2Y Treasury' in series:
    series['US 10Y–2Y curve'] = (series['US 10Y Treasury']-series['US 2Y Treasury']).dropna()
    meta['US 10Y–2Y curve'] = {'country':'United States','category':'Rates','kind':'rate','source':'Calculated'}
if 'US HY OAS' in series and 'US IG OAS' in series:
    series['US HY–IG OAS gap'] = (series['US HY OAS']-series['US IG OAS']).dropna()
    meta['US HY–IG OAS gap'] = {'country':'United States','category':'Credit','kind':'rate','source':'Calculated'}

names = [n for n,m in meta.items() if m['country'] in economies and m['category'] in categories]
if not series:
    st.error('No stored data yet. In GitHub, open Actions → Refresh public market data → Run workflow, then refresh this page after the workflow commits the CSV files.')
    st.stop()
if not names:
    st.info('No stored series match these filters. Try another economy or asset class. Singapore SGS/SORA automation is not yet included.')
    st.stop()

rows = []
periods = {'1D':1,'1W':7,'1M':30,'3M':90,'1Y':365}
for name in names:
    s = series[name]
    m = meta[name]
    row = {'Economy':m['country'],'Category':m['category'],'Indicator':name,'Latest':round(float(s.iloc[-1]),4),'As of':s.index[-1].date().isoformat(),'Change unit':'bp' if m['kind']=='rate' else '%','Source':m['source']}
    for label, days in periods.items():
        row[label] = change(s, days, m['kind'])
    rows.append(row)
summary = pd.DataFrame(rows)
market, charts, stress, sources = st.tabs(['Market Overview','Historical Charts','Stress Benchmarking','Data & Sources'])
with market:
    st.subheader('Latest market movements')
    st.caption('Rate and spread changes in bp; FX and equity/index changes in %. Rate levels are in percentage points, VIX in index points. Each series has its own observation date.')
    st.dataframe(summary.drop(columns=['Source']), hide_index=True, use_container_width=True)
    st.download_button('Download market movements CSV', summary.to_csv(index=False), file_name='market_movements.csv', mime='text/csv')
with charts:
    selected = st.multiselect('Indicators', names, default=names[:1])
    window = st.selectbox('Chart period',['3M','1Y','5Y','10Y','All'],index=2)
    rebase = st.checkbox('Rebase positive series to 100')
    frames=[]
    for name in selected:
        s = series[name].copy()
        if window!='All':
            days={'3M':90,'1Y':365,'5Y':1825,'10Y':3650}[window]
            s=s.loc[s.index>=s.index.max()-pd.Timedelta(days=days)]
        if rebase:
            if s.empty or (s<=0).any():
                st.warning(f'Cannot rebase {name}: nonpositive or missing observations.')
                continue
            s=s/s.iloc[0]*100
        frames.append(pd.DataFrame({'Date':s.index,'Value':s.to_numpy(),'Indicator':name}))
    if frames:
        fig=px.line(pd.concat(frames,ignore_index=True),x='Date',y='Value',color='Indicator',template='plotly_white')
        fig.update_layout(height=500,hovermode='x unified')
        st.plotly_chart(fig,use_container_width=True)
with stress:
    indicator=st.selectbox('Indicator',names)
    horizon=st.selectbox('Change horizon',['1M','3M','1Y'],index=1)
    changes=rolling(series[indicator],{'1M':30,'3M':90,'1Y':365}[horizon],meta[indicator]['kind'])
    if len(changes)<20:
        st.warning('Insufficient history for benchmarking.')
    else:
        latest=float(changes.iloc[-1]); pct=float((changes<=latest).mean()*100)
        unit='bp' if meta[indicator]['kind']=='rate' else '%'
        a,b,c=st.columns(3)
        a.metric('Latest movement',f'{latest:+.2f} {unit}')
        b.metric('Signed historical percentile',f'{pct:.1f}%')
        c.metric('Historical windows',f'{len(changes):,}')
        fig=px.histogram(x=changes.to_numpy(),nbins=60,labels={'x':f'Change ({unit})'})
        fig.add_vline(x=latest,line_dash='dash',line_color='red')
        st.plotly_chart(fig,use_container_width=True)
        st.caption('Overlapping historical windows; percentiles are descriptive, not probabilities of future stress.')
with sources:
    availability=pd.DataFrame([{'Indicator':n,'Source':meta[n]['source'],'First':s.index.min().date().isoformat(),'Latest':s.index.max().date().isoformat(),'Observations':len(s)} for n,s in series.items() if not s.empty])
    st.dataframe(availability,hide_index=True,use_container_width=True)
    st.download_button('Download stored observations',pd.concat(series,axis=1).sort_index().to_csv(index_label='Date'),file_name='observations.csv',mime='text/csv')
    st.caption('Source: Federal Reserve Economic Data (FRED). Some series, including S&P 500, have restricted historical coverage. SGS and SORA require a separate verified integration.')
st.divider()
st.caption('Public-data research prototype. Verify data freshness and definitions before official use. Do not upload confidential information.')
