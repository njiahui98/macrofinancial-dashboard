"""Refresh public FRED CSV snapshots without deleting good data on failure."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import json
import os
import time
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data'
OUT.mkdir(exist_ok=True)
CATALOG = {
    'US 2Y Treasury': ('DGS2', 'United States', 'Rates', 'rate'),
    'US 5Y Treasury': ('DGS5', 'United States', 'Rates', 'rate'),
    'US 10Y Treasury': ('DGS10', 'United States', 'Rates', 'rate'),
    'US 30Y Treasury': ('DGS30', 'United States', 'Rates', 'rate'),
    'US IG OAS': ('BAMLC0A0CM', 'United States', 'Credit', 'rate'),
    'US BBB OAS': ('BAMLC0A4CBBB', 'United States', 'Credit', 'rate'),
    'US HY OAS': ('BAMLH0A0HYM2', 'United States', 'Credit', 'rate'),
    'VIX': ('VIXCLS', 'United States', 'Volatility', 'index'),
    'S&P 500': ('SP500', 'United States', 'Equities', 'index'),
    'USD/SGD': ('DEXSIUS', 'Singapore', 'FX', 'fx'),
    'USD/CNY': ('DEXCHUS', 'China', 'FX', 'fx'),
    'USD/JPY': ('DEXJPUS', 'Japan', 'FX', 'fx'),
    'EUR/USD': ('DEXUSEU', 'Euro area', 'FX', 'fx'),
    'GBP/USD': ('DEXUSUK', 'United Kingdom', 'FX', 'fx'),
}

def fetch(code):
    url = 'https://fred.stlouisfed.org/graph/fredgraph.csv'
    error = None
    for attempt in range(2):
        try:
            r = requests.get(url, params={'id': code}, timeout=(8, 15), headers={'User-Agent': 'MacrofinancialMonitor/1.0 (public research)'})
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.text))
            if len(df.columns) < 2:
                raise ValueError('Unexpected CSV format')
            dates = pd.to_datetime(df.iloc[:, 0], errors='coerce')
            values = pd.to_numeric(df.iloc[:, 1], errors='coerce')
            clean = pd.DataFrame({'Date': dates, 'Value': values}).dropna().sort_values('Date')
            clean = clean.drop_duplicates('Date', keep='last')
            if len(clean) < 5:
                raise ValueError('Too few valid observations')
            return clean
        except (requests.RequestException, ValueError, pd.errors.ParserError) as exc:
            error = exc
            if attempt == 0:
                time.sleep(2)
    raise RuntimeError(str(error))

def refresh(item):
    name, (code, country, category, kind) = item
    new = fetch(code)
    target = OUT / f'{code}.csv'
    if target.exists():
        try:
            old = pd.read_csv(target, parse_dates=['Date'])
            new = pd.concat([old, new], ignore_index=True)
            new['Date'] = pd.to_datetime(new['Date'], errors='coerce')
            new['Value'] = pd.to_numeric(new['Value'], errors='coerce')
            new = new.dropna().sort_values('Date').drop_duplicates('Date', keep='last')
        except Exception as exc:
            print(f'Existing {code} snapshot unreadable; replacing with validated download: {exc}')
    temp = target.with_suffix('.tmp')
    new.to_csv(temp, index=False, date_format='%Y-%m-%d')
    os.replace(temp, target)
    return name, code, len(new), new['Date'].max().strftime('%Y-%m-%d')

def main():
    results, failures = [], {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(refresh, item): item[0] for item in CATALOG.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                results.append(result)
                print('OK', *result, flush=True)
            except Exception as exc:
                failures[name] = str(exc)
                print('FAILED', name, str(exc), flush=True)
    status = {'attempt_utc': datetime.now(timezone.utc).isoformat(), 'successful': len(results), 'failed': failures}
    (OUT / 'update_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    print(f'Completed: {len(results)} success, {len(failures)} failure')
    # Fail if nothing was refreshed, while preserving existing snapshots.
    if not results:
        raise SystemExit('All downloads failed. Existing CSV snapshots remain unchanged.')

if __name__ == '__main__':
    main()
