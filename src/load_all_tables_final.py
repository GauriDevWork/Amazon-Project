# src/load_all_tables_and_enrich.py
"""
Final loader:
- Reads cleaned CSV in chunks
- Aggregates products/customers/time
- Writes products/customers/time_dimension to MySQL in safe chunks
- Updates transactions table to backfill product/customer/time fields via SQL JOINs
"""

import os
import time
import math
import glob
from collections import defaultdict
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy import types as sqltypes
import pymysql
from sqlalchemy.exc import SQLAlchemyError, DBAPIError

# ------------------- CONFIG -------------------
DB_HOST = "127.0.0.1"
DB_PORT = 3306
DB_USER = "root"
DB_PASS = ""             # change if you set password
DB_NAME = "amazon_analytics"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
# prefer cleaned file with fixed dates if available
CSV_CANDIDATES = [
    os.path.join(OUTPUTS_DIR, "amazon_cleaned_dates_fixed.csv"),
    os.path.join(OUTPUTS_DIR, "amazon_cleaned.csv"),
    os.path.join(OUTPUTS_DIR, "amazon_india_2015_2025_raw.csv")
]
CHUNK_ROWS = 150_000   # chunk size for scanning CSV
WRITE_CHUNK_ROWS = {
    'products': 5000,
    'customers': 10000,
    'time_dimension': 500
}
MAX_RETRIES = 3
# -----------------------------------------------

# find CSV
CSV_PATH = None
for p in CSV_CANDIDATES:
    if os.path.exists(p):
        CSV_PATH = p
        break
if not CSV_PATH:
    raise FileNotFoundError("Cleaned CSV not found in outputs/. Put amazon_cleaned*.csv there.")

print("Using CSV:", CSV_PATH)

# ---------- helper conversions ----------
def parse_bool_like(x):
    if pd.isna(x):
        return 0
    s = str(x).strip().lower()
    if s in ("1","1.0","true","t","y","yes","prime"):
        return 1
    if s in ("0","0.0","false","f","n","no","none","nan","null"):
        return 0
    try:
        n = float(s)
        return 1 if n != 0 else 0
    except Exception:
        return 0

def safe_first_nonnull(ser):
    s = ser.dropna()
    return s.iloc[0] if len(s) else None

# ---------- accumulators ----------
prod_stats = {}   # product_id -> dict
cust_stats = {}   # customer_id -> dict
months_seen = set()
rows_scanned = 0

# read header to know columns
hdr = pd.read_csv(CSV_PATH, nrows=0).columns.tolist()
print("Detected columns:", len(hdr))

# convenience getters
def has(c): return c if c in hdr else None

COL_TXN = has("transaction_id")
COL_DATE = has("order_date")
COL_CUST = has("customer_id")
COL_PROD = has("product_id")
COL_PNAME = has("product_name")
COL_CAT = has("category")
COL_SUBCAT = has("subcategory")
COL_BRAND = has("brand")
COL_ORIG = has("original_price_inr")
COL_FINAL = has("final_amount_inr")
COL_PRATING = has("product_rating")
COL_QTY = has("quantity")
COL_CITY = has("customer_city")
COL_STATE = has("customer_state")
COL_ISPRIME = has("is_prime_member")
COL_SUBTOTAL = has("subtotal_inr")
COL_ORDERYEAR = has("order_year")
COL_ORDERMONTH = has("order_month")
COL_ORDERQUARTER = has("order_quarter")

print("Starting CSV scan in chunks (this builds dimension aggregates)...")
start = time.time()
for i, chunk in enumerate(pd.read_csv(CSV_PATH, chunksize=CHUNK_ROWS, low_memory=False), start=1):
    r = len(chunk)
    rows_scanned += r
    print(f"Chunk {i}: rows={r}  cumulative={rows_scanned:,}")

    # parse date (try dayfirst True as many dates in dataset looked 'DD-MM-YYYY')
    if COL_DATE:
        # try intelligently: if format contains '-' and day likely first, use dayfirst True
        chunk[COL_DATE] = pd.to_datetime(chunk[COL_DATE], errors='coerce', dayfirst=True)

    for col in (COL_ORIG, COL_FINAL, COL_PRATING, COL_QTY, COL_SUBTOTAL):
        if col:
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce')

    if COL_ISPRIME:
        chunk['_is_prime_norm'] = chunk[COL_ISPRIME].apply(parse_bool_like)
    else:
        chunk['_is_prime_norm'] = 0

    # products aggregation by product_id
    if COL_PROD:
        grp = chunk.groupby(COL_PROD, dropna=False)
        for pid, g in grp:
            if pd.isna(pid): continue
            pid = str(pid).strip()
            if pid == "": continue
            rec = prod_stats.setdefault(pid, {
                'product_id': pid,
                'product_name': None,
                'category': None,
                'subcategory': None,
                'brand': None,
                'sum_orig': 0.0,
                'sum_final': 0.0,
                'sum_rating': 0.0,
                'count_rating': 0,
                'txn_count': 0
            })
            # first non-null strings
            if COL_PNAME:
                v = safe_first_nonnull(g[COL_PNAME])
                if v and rec['product_name'] is None: rec['product_name'] = str(v).strip()
            if COL_CAT:
                v = safe_first_nonnull(g[COL_CAT])
                if v and rec['category'] is None: rec['category'] = str(v).strip()
            if COL_SUBCAT:
                v = safe_first_nonnull(g[COL_SUBCAT])
                if v and rec['subcategory'] is None: rec['subcategory'] = str(v).strip()
            if COL_BRAND:
                v = safe_first_nonnull(g[COL_BRAND])
                if v and rec['brand'] is None: rec['brand'] = str(v).strip()
            if COL_ORIG:
                s = g[COL_ORIG].sum(skipna=True)
                rec['sum_orig'] += 0.0 if pd.isna(s) else float(s)
            if COL_FINAL:
                s = g[COL_FINAL].sum(skipna=True)
                rec['sum_final'] += 0.0 if pd.isna(s) else float(s)
            if COL_PRATING:
                ratings = g[COL_PRATING].dropna().astype(float)
                rec['sum_rating'] += ratings.sum() if len(ratings) else 0.0
                rec['count_rating'] += len(ratings)
            rec['txn_count'] += len(g)

    # customers aggregation
    if COL_CUST:
        grp = chunk.groupby(COL_CUST, dropna=False)
        for cid, g in grp:
            if pd.isna(cid): continue
            cid = str(cid).strip()
            if cid == "": continue
            rec = cust_stats.setdefault(cid, {
                'customer_id': cid,
                'customer_city': None,
                'customer_state': None,
                'is_prime_member': 0,
                'total_orders': 0,
                'total_spent': 0.0
            })
            if COL_CITY:
                v = safe_first_nonnull(g[COL_CITY])
                if v and rec['customer_city'] is None: rec['customer_city'] = str(v).strip()
            if COL_STATE:
                v = safe_first_nonnull(g[COL_STATE])
                if v and rec['customer_state'] is None: rec['customer_state'] = str(v).strip()
            # prime
            if g['_is_prime_norm'].sum() > 0: rec['is_prime_member'] = 1
            if COL_FINAL:
                s = g[COL_FINAL].sum(skipna=True)
                rec['total_spent'] += 0.0 if pd.isna(s) else float(s)
            rec['total_orders'] += len(g)

    # collect month periods
    if COL_DATE:
        valid = chunk[COL_DATE].dropna()
        for d in valid:
            try:
                months_seen.add(str(pd.Period(d).asfreq('M')))
            except Exception:
                pass

elapsed = time.time() - start
print("Scan done. Rows scanned:", rows_scanned)
print("Unique products:", len(prod_stats))
print("Unique customers:", len(cust_stats))
print("Unique month-periods:", len(months_seen))
print("Elapsed:", round(elapsed, 1), "s")

# ---------- build DataFrames ----------
print("Building DataFrames...")
prod_rows = []
for pid, r in prod_stats.items():
    avg_orig = r['sum_orig'] / r['txn_count'] if r['txn_count'] else None
    avg_final = r['sum_final'] / r['txn_count'] if r['txn_count'] else None
    avg_rating = r['sum_rating'] / r['count_rating'] if r['count_rating'] else None
    prod_rows.append({
        'product_id': r['product_id'],
        'product_name': r['product_name'],
        'category': r['category'],
        'subcategory': r['subcategory'],
        'brand': r['brand'],
        'avg_original_price': avg_orig,
        'avg_final_price': avg_final,
        'avg_rating': avg_rating,
        'transactions_count': r['txn_count']
    })
df_products = pd.DataFrame(prod_rows)

cust_rows = []
for cid, r in cust_stats.items():
    avg_order_val = r['total_spent'] / r['total_orders'] if r['total_orders'] else None
    cust_rows.append({
        'customer_id': r['customer_id'],
        'customer_city': r['customer_city'],
        'customer_state': r['customer_state'],
        'is_prime_member': int(r['is_prime_member']),
        'total_orders': int(r['total_orders']),
        'total_spent': float(r['total_spent']),
        'avg_order_value': avg_order_val
    })
df_customers = pd.DataFrame(cust_rows)

if months_seen:
    months_sorted = sorted(months_seen)
    time_rows = []
    for m in months_sorted:
        try:
            p = pd.Period(m)
            ts = p.to_timestamp()
            time_rows.append({
                'order_year': ts.year,
                'order_month': ts.month,
                'order_quarter': ((ts.month - 1) // 3) + 1,
                'month_start_date': ts.date(),
                'month_name': ts.strftime('%B'),
                'month_label': ts.strftime('%Y-%m')
            })
        except Exception:
            pass
    df_time = pd.DataFrame(time_rows)
else:
    rng = pd.date_range("2015-01-01", "2025-12-01", freq='MS')
    df_time = pd.DataFrame({
        'order_year': rng.year,
        'order_month': rng.month,
        'order_quarter': rng.quarter,
        'month_start_date': rng.date,
        'month_name': rng.strftime('%B'),
        'month_label': rng.strftime('%Y-%m')
    })

print("Shapes:", df_products.shape, df_customers.shape, df_time.shape)

# ---------- safe chunked writer ----------
def get_engine():
    return create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
                         pool_pre_ping=True, pool_recycle=3600)

engine = get_engine()

def write_table_in_chunks(df, table_name, dtype_map, chunk_size=5000):
    """
    Replace table schema then append data in committed chunks.
    """
    total = len(df)
    print(f"[write] Preparing table '{table_name}' ({total:,} rows) chunk_size={chunk_size}")
    if total == 0:
        # create empty schema
        df.head(0).to_sql(table_name, con=engine, if_exists='replace', index=False, dtype=dtype_map)
        print(f"[write] Created empty table {table_name}")
        return

    # create/replace schema (empty)
    df.head(0).to_sql(table_name, con=engine, if_exists='replace', index=False, dtype=dtype_map)
    start_idx = 0
    chunk_no = 0
    while start_idx < total:
        chunk_no += 1
        end_idx = min(start_idx + chunk_size, total)
        sub = df.iloc[start_idx:end_idx].copy()
        attempt = 0
        while attempt < MAX_RETRIES:
            attempt += 1
            try:
                # fresh engine/connection each chunk
                eng = get_engine()
                with eng.begin() as conn:
                    sub.to_sql(table_name, con=conn, if_exists='append', index=False, dtype=dtype_map, method='multi')
                eng.dispose()
                print(f"[write] {table_name} chunk {chunk_no}: inserted rows {end_idx}/{total}")
                break
            except Exception as e:
                print(f"[write] ERROR writing {table_name} chunk {chunk_no} attempt {attempt}: {e}")
                try:
                    eng.dispose()
                except Exception:
                    pass
                time.sleep(1)
                if attempt >= MAX_RETRIES:
                    debug_path = os.path.join(BASE_DIR, f"debug_failed_{table_name}_chunk{chunk_no}.csv")
                    sub.to_csv(debug_path, index=False)
                    raise RuntimeError(f"Failed to insert chunk {chunk_no} for {table_name}. Dumped to {debug_path}")
        start_idx = end_idx

# ---------- define SQL types ----------
dtypes_products = {
    'product_id': sqltypes.VARCHAR(128),
    'product_name': sqltypes.VARCHAR(512),
    'category': sqltypes.VARCHAR(128),
    'subcategory': sqltypes.VARCHAR(128),
    'brand': sqltypes.VARCHAR(128),
    'avg_original_price': sqltypes.Float,
    'avg_final_price': sqltypes.Float,
    'avg_rating': sqltypes.Float,
    'transactions_count': sqltypes.Integer
}
dtypes_customers = {
    'customer_id': sqltypes.VARCHAR(128),
    'customer_city': sqltypes.VARCHAR(128),
    'customer_state': sqltypes.VARCHAR(128),
    'is_prime_member': sqltypes.Integer,
    'total_orders': sqltypes.Integer,
    'total_spent': sqltypes.Float,
    'avg_order_value': sqltypes.Float
}
dtypes_time = {
    'order_year': sqltypes.Integer,
    'order_month': sqltypes.Integer,
    'order_quarter': sqltypes.Integer,
    'month_start_date': sqltypes.DATE,
    'month_name': sqltypes.VARCHAR(32),
    'month_label': sqltypes.VARCHAR(16)
}

# ---------- write dims ----------
write_table_in_chunks(df_products, 'products', dtypes_products, chunk_size=WRITE_CHUNK_ROWS['products'])
write_table_in_chunks(df_customers, 'customers', dtypes_customers, chunk_size=WRITE_CHUNK_ROWS['customers'])
write_table_in_chunks(df_time, 'time_dimension', dtypes_time, chunk_size=WRITE_CHUNK_ROWS['time_dimension'])

# ---------- Enrich transactions via SQL updates ----------
print("Connecting for enrichment and running JOIN updates...")
conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, db=DB_NAME, port=DB_PORT, autocommit=True)
cur = conn.cursor()

# 1) backfill order_year/month/quarter from order_date where null/0
cur.execute("""
UPDATE transactions
SET order_year = COALESCE(NULLIF(order_year,0), YEAR(order_date)),
    order_month = COALESCE(NULLIF(order_month,0), MONTH(order_date)),
    order_quarter = COALESCE(NULLIF(order_quarter,0), QUARTER(order_date))
WHERE order_date IS NOT NULL;
""")

# 2) trim product_id/customer_id whitespace in transactions (helpful)
cur.execute("UPDATE transactions SET product_id = TRIM(product_id), customer_id = TRIM(customer_id);")

# 3) update transactions from products table
cur.execute("""
UPDATE transactions t
LEFT JOIN products p ON TRIM(t.product_id) = TRIM(p.product_id)
SET
  t.product_name = COALESCE(t.product_name, p.product_name),
  t.category = COALESCE(t.category, p.category),
  t.subcategory = COALESCE(t.subcategory, p.subcategory),
  t.brand = COALESCE(t.brand, p.brand),
  t.product_rating = COALESCE(t.product_rating, p.avg_rating)
;
""")

# 4) update transactions from customers table
cur.execute("""
UPDATE transactions t
LEFT JOIN customers c ON TRIM(t.customer_id) = TRIM(c.customer_id)
SET
  t.customer_city = COALESCE(t.customer_city, c.customer_city),
  t.customer_state = COALESCE(t.customer_state, c.customer_state),
  t.is_prime_member = COALESCE(t.is_prime_member, c.is_prime_member)
;
""")

# 5) fill final_amount if missing from subtotal + delivery_charges
cur.execute("""
UPDATE transactions
SET final_amount_inr = COALESCE(final_amount_inr, subtotal_inr + COALESCE(delivery_charges,0))
WHERE final_amount_inr IS NULL AND subtotal_inr IS NOT NULL;
""")

# add indexes (safe to ignore errors)
for sql in [
    "CREATE INDEX IF NOT EXISTS idx_tx_product ON transactions(product_id);",
    "CREATE INDEX IF NOT EXISTS idx_tx_customer ON transactions(customer_id);",
    "CREATE INDEX IF NOT EXISTS idx_tx_orderdate ON transactions(order_date);"
]:
    try:
        cur.execute(sql)
    except Exception:
        # some MySQL versions don't support IF NOT EXISTS for CREATE INDEX — ignore
        pass

# validation queries
def q_one(q):
    cur.execute(q)
    return cur.fetchone()

total_tx = q_one("SELECT COUNT(*) FROM transactions;")[0]
nulls = q_one("SELECT COUNT(*) FROM transactions WHERE product_name IS NULL OR category IS NULL OR customer_city IS NULL;")[0]
prod_count = q_one("SELECT COUNT(*) FROM products;")[0]
cust_count = q_one("SELECT COUNT(*) FROM customers;")[0]
time_count = q_one("SELECT COUNT(*) FROM time_dimension;")[0]

cur.close()
conn.close()

print("\n=== DONE: Summary ===")
print(f"Total transactions: {total_tx:,}")
print(f"Transactions with product_name/category/customer_city NULL: {nulls:,}")
print(f"Products rows: {prod_count:,}")
print(f"Customers rows: {cust_count:,}")
print(f"Time dimension rows: {time_count:,}")
print("Elapsed total:", round(time.time() - start, 1), "s")
print("All tables loaded and transactions enriched successfully.")