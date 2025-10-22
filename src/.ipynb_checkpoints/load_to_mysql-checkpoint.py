import pandas as pd
from sqlalchemy import create_engine, types
import pymysql
import os

# DB config - EDIT THESE
DB_HOST = "127.0.0.1"        # or your host provided by Namecheap
DB_PORT = 3306
DB_USER = "webtwngh_amazon_analytics"
DB_PASS = "D2{YYxw8,yLA"
DB_NAME = "webtwngh_amazon_analytics"

# path to cleaned CSV
csv_path = "../outputs/amazon_cleaned.csv"

# create engine
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
                       pool_pre_ping=True, pool_recycle=3600)

# dtype mapping to optimize storage
dtype_map = {
    'transaction_id': types.VARCHAR(128),
    'customer_id': types.VARCHAR(128),
    'product_id': types.VARCHAR(128),
    'product_name': types.VARCHAR(512),
    'category': types.VARCHAR(128),
    'subcategory': types.VARCHAR(128),
    'brand': types.VARCHAR(128),
    'order_date': types.DATE,
    'product_rating': types.Float,
    'original_price_inr': types.Float,
    'discount_percent': types.Float,
    'final_amount_inr': types.Float,
    'delivery_charges': types.Float,
    'delivery_days': types.Integer,
    'payment_method': types.VARCHAR(64),
    'is_prime_member': types.Boolean,
    'is_festival_sale': types.Boolean,
    'return_status': types.VARCHAR(64),
    'customer_rating': types.Float,
    'customer_city': types.VARCHAR(128),
    'customer_state': types.VARCHAR(128),
    'order_month': types.Integer,
    'order_year': types.Integer,
    'order_quarter': types.Integer
}

# 1) Create a staging table (transactions_staging) to allow validation
with engine.connect() as conn:
    conn.execute("DROP TABLE IF EXISTS transactions_staging;")
    conn.execute("""
        CREATE TABLE transactions_staging LIKE transactions;
    """)
print("Staging table prepared.")

# 2) Read CSV in chunks and insert
chunksize = 100_000
reader = pd.read_csv(csv_path, low_memory=False, parse_dates=['order_date'], chunksize=chunksize)

count = 0
for chunk in reader:
    # minor preprocessing: ensure date and numeric casting
    chunk['order_date'] = pd.to_datetime(chunk['order_date'], errors='coerce').dt.date
    chunk['order_year'] = pd.to_datetime(chunk['order_date']).dt.year
    chunk['order_month'] = pd.to_datetime(chunk['order_date']).dt.month
    chunk['order_quarter'] = pd.to_datetime(chunk['order_date']).dt.quarter

    # ensure booleans are 0/1
    for b in ['is_prime_member','is_festival_sale']:
        if b in chunk.columns:
            chunk[b] = chunk[b].map({True:1, False:0, 'True':1, 'False':0, '1':1, '0':0}).fillna(0).astype(int)

    # write chunk to staging table
    chunk.to_sql('transactions_staging', con=engine, if_exists='append', index=False, dtype=dtype_map, method='multi')
    count += len(chunk)
    print(f"Inserted {count} rows so far...")

print("All chunks inserted into transactions_staging.")
