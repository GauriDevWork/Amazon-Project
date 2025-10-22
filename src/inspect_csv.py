# src/preprocess_dates_and_write.py
"""
Read amazon_cleaned.csv in chunks, normalize 'order_date' into YYYY-MM-DD,
write amazon_cleaned_dates_fixed.csv (same columns, order preserved).
"""

import os, sys
import pandas as pd
from dateutil import parser

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IN_CSV = os.path.join(BASE, "outputs", "amazon_cleaned.csv")
OUT_CSV = os.path.join(BASE, "outputs", "amazon_cleaned_dates_fixed.csv")
CHUNK = 200_000

def parse_date_safe(val):
    if pd.isna(val): return pd.NaT
    s = str(val).strip()
    if s == "" or s.lower() in ("nan","none","null","0","0000-00-00"):
        return pd.NaT
    # try common formats quickly
    for fmt in ("%d-%m-%Y","%d/%m/%Y","%Y-%m-%d","%d-%b-%Y","%d %b %Y"):
        try:
            return pd.to_datetime(s, format=fmt, dayfirst=True, errors='raise')
        except Exception:
            pass
    # fallback to dateutil
    try:
        return pd.to_datetime(parser.parse(s, dayfirst=True))
    except Exception:
        return pd.NaT

if not os.path.exists(IN_CSV):
    print("Input CSV not found:", IN_CSV); sys.exit(1)

first = True
total = 0
parsed = 0
unparsed = 0

print("Reading", IN_CSV)
for i, chunk in enumerate(pd.read_csv(IN_CSV, chunksize=CHUNK, low_memory=False), start=1):
    print(f"Chunk {i}: rows={len(chunk)}")
    total += len(chunk)
    if 'order_date' in chunk.columns:
        parsed_dates = chunk['order_date'].apply(parse_date_safe)
        parsed_mask = parsed_dates.notna()
        parsed += parsed_mask.sum()
        unparsed += (~parsed_mask).sum()
        # convert to ISO string or empty
        chunk['order_date'] = parsed_dates.dt.strftime('%Y-%m-%d')
        chunk['order_date'] = chunk['order_date'].fillna('')
    else:
        # if column missing, add empty column (but header shows it exists)
        chunk['order_date'] = ''

    # write out
    if first:
        chunk.to_csv(OUT_CSV, index=False, mode='w', encoding='utf-8')
        first = False
    else:
        chunk.to_csv(OUT_CSV, index=False, mode='a', header=False, encoding='utf-8')

print("Done.")
print(f"Total rows processed: {total:,}; Parsed: {parsed:,}; Unparsed/empty: {unparsed:,}")
print("Cleaned CSV at:", OUT_CSV)
