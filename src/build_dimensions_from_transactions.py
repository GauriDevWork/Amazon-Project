import os
import pandas as pd
import pymysql
import calendar
from sqlalchemy import create_engine, text

DB_HOST = "127.0.0.1"
DB_USER = "root"
DB_PASS = ""
DB_NAME = "amazon_analytics"
DB_PORT = 3306

engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME, local_infile=1)

# Load from parquet or CSV (choose your clean file)
file_path = r"C:\Users\gauri\Python-Projects\sales project\outputs\amazon_cleaned.csv"
df = pd.read_csv(file_path, low_memory=False)

print(f"Loaded {len(df):,} rows from cleaned file")

# --- Build tables in Python ---
# Products
df_products = (
    df.groupby(['product_id', 'product_name', 'category', 'subcategory', 'brand'])
      .agg(
          count_transactions=('transaction_id', 'count'),
          avg_original_price=('original_price_inr', 'mean'),
          avg_final_price=('final_amount_inr', 'mean'),
          avg_rating=('product_rating', 'mean')
      )
      .reset_index()
)
# Customers
df_customers = (
    df.groupby(['customer_id', 'customer_city', 'customer_state'])
      .agg(
          total_orders=('transaction_id', 'count'),
          total_spent=('final_amount_inr', 'sum'),
          avg_order_value=('final_amount_inr', 'mean'),
          is_prime_member=('is_prime_member', 'max')
      )
      .reset_index()
)
# Time Dimension
if 'order_date' in df.columns:
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    df_time = df.dropna(subset=['order_date']).copy()
    df_time['order_year'] = df_time['order_date'].dt.year
    df_time['order_month'] = df_time['order_date'].dt.month
    df_time['order_quarter'] = df_time['order_date'].dt.quarter
    df_time = (
        df_time[['order_year', 'order_month', 'order_quarter']]
        .drop_duplicates()
        .sort_values(['order_year', 'order_month'])
        .assign(month_start_date=lambda d: pd.to_datetime(d['order_year'].astype(str) + '-' + d['order_month'].astype(str) + '-01'))
    )
    df_time['month_name'] = df_time['order_month'].apply(lambda m: calendar.month_name[int(m)])
else:
    # fallback 2015–2025
    months = pd.date_range(start="2015-01-01", end="2025-12-01", freq='MS')
    df_time = pd.DataFrame({
        'order_year': [m.year for m in months],
        'order_month': [m.month for m in months],
        'order_quarter': [(m.month - 1)//3 + 1 for m in months],
        'month_start_date': [m.date() for m in months],
        'month_name': [calendar.month_name[m.month] for m in months],
    })

print("✅ DataFrames built:")
print("products:", df_products.shape)
print("customers:", df_customers.shape)
print("time_dimension:", df_time.shape)

# --- Write to MySQL (truncate + insert) ---
with conn.cursor() as cur:
    cur.execute("DROP TABLE IF EXISTS products;")
    cur.execute("""
        CREATE TABLE products (
            product_id VARCHAR(128), product_name VARCHAR(512),
            category VARCHAR(128), subcategory VARCHAR(128), brand VARCHAR(128),
            count_transactions INT, avg_original_price FLOAT,
            avg_final_price FLOAT, avg_rating FLOAT
        );
    """)

    cur.execute("DROP TABLE IF EXISTS customers;")
    cur.execute("""
        CREATE TABLE customers (
            customer_id VARCHAR(128), customer_city VARCHAR(128),
            customer_state VARCHAR(128), is_prime_member TINYINT(1),
            total_orders INT, total_spent FLOAT, avg_order_value FLOAT
        );
    """)

    cur.execute("DROP TABLE IF EXISTS time_dimension;")
    cur.execute("""
        CREATE TABLE time_dimension (
            order_year INT, order_month INT, order_quarter INT,
            month_start_date DATE, month_name VARCHAR(32)
        );
    """)

    conn.commit()

# Write via cursor.executemany
def insert_many(df, table, cols):
    placeholders = ", ".join(["%s"] * len(cols))
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
    data = [tuple(x) for x in df[cols].values.tolist()]
    with conn.cursor() as cur:
        cur.executemany(sql, data)
    conn.commit()
    print(f"✅ Inserted {len(df):,} rows into {table}")

insert_many(df_products, 'products', list(df_products.columns))
insert_many(df_customers, 'customers', list(df_customers.columns))
insert_many(df_time, 'time_dimension', list(df_time.columns))

print("🎉 All dimension tables loaded successfully!")
conn.close()
