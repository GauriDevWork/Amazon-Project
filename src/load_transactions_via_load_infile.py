# src/load_cleaned_csv_to_mysql.py
"""
Load the preprocessed CSV (with ISO dates) into transactions using LOAD DATA LOCAL INFILE.
This script uses the header order you provided to map columns exactly.
"""

import os, sys, time
import pymysql

# ---------- CONFIG ----------
DB_HOST = "127.0.0.1"
DB_PORT = 3306
DB_USER = "root"
DB_PASS = ""
DB_NAME = "amazon_analytics"

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_FILE = os.path.join(BASE, "outputs", "amazon_cleaned_dates_fixed.csv")

if not os.path.exists(CSV_FILE):
    print("Cleaned CSV not found:", CSV_FILE)
    sys.exit(1)

# --- Column list exactly matching the CSV header you provided ---
columns = [
 'transaction_id','order_date','customer_id','product_id','product_name','category','subcategory','brand',
 'original_price_inr','discount_percent','discounted_price_inr','quantity','subtotal_inr','delivery_charges',
 'final_amount_inr','customer_city','customer_state','customer_tier','customer_spending_tier','customer_age_group',
 'payment_method','delivery_days','delivery_type','is_prime_member','is_festival_sale','festival_name',
 'customer_rating','return_status','order_month','order_year','order_quarter','product_weight_kg','is_prime_eligible',
 'product_rating','is_prime_member_clean','is_prime_eligible_clean','is_festival_sale_clean','dup_key','is_duplicate',
 'price_median_cat','original_price_inr_clean_adj'
]

# Use the column names in LOAD DATA. MySQL expects those columns to exist in the target table; if transactions has fewer columns it's okay,
# but the order must match or you can load into @vars and set explicitly (we assume same names or that extra CSV cols will be ignored).
col_sql = ", ".join(f"`{c}`" for c in columns)

print("Connecting to MySQL (local_infile=1 required)...")
conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, db=DB_NAME,
                       port=DB_PORT, local_infile=1, autocommit=True)
try:
    cur = conn.cursor()
    print("Truncating transactions table (replace mode)...")
    cur.execute("SET FOREIGN_KEY_CHECKS=0;")
    cur.execute("TRUNCATE TABLE transactions;")
    cur.execute("SET FOREIGN_KEY_CHECKS=1;")

    print("Starting LOAD DATA LOCAL INFILE ... This may take a few minutes.")
    start = time.time()

    # Here we give a column list matching the CSV header (if transactions table doesn't have some
    # of these columns, MySQL will error; in that case we should use @vars and SET to match).
    # If transactions has fewer columns or different names, we will fall back to loading via @vars.
    try:
        load_sql = f"""
        LOAD DATA LOCAL INFILE %s
        INTO TABLE transactions
        FIELDS TERMINATED BY ',' ENCLOSED BY '"' 
        LINES TERMINATED BY '\\n'
        IGNORE 1 LINES
        ({col_sql});
        """
        cur.execute(load_sql, (CSV_FILE,))
    except Exception as e:
        # fallback: load into user variables and then set matching columns we have in transactions
        print("Direct LOAD failed (likely column mismatch). Falling back to variable LOAD with SET mapping.")
        # build @vars list equal to number of columns
        var_list = ", ".join(f"@v{i}" for i in range(len(columns)))
        # build SET mapping only for columns that likely exist in transactions (common subset)
        # Minimal safe mapping: set order_date, transaction_id, customer_id, product_id, final_amount_inr, order_year, order_month
        set_map = []
        mapping = {
            'transaction_id': "@v0",
            'order_date': "@v1",
            'customer_id': "@v2",
            'product_id': "@v3",
            'final_amount_inr': "@v14",
            'order_month': "@v27",
            'order_year': "@v28",
            'order_quarter': "@v29"
        }
        for col, var in mapping.items():
            if col == 'order_date':
                set_map.append(f"order_date = NULLIF(@v1, '')")
            else:
                set_map.append(f"{col} = NULLIF({var}, '')")
        set_sql = ",\n    ".join(set_map)
        load_sql2 = f"""
        LOAD DATA LOCAL INFILE %s
        INTO TABLE transactions
        FIELDS TERMINATED BY ',' ENCLOSED BY '"' 
        LINES TERMINATED BY '\\n'
        IGNORE 1 LINES
        ({var_list})
        SET
        {set_sql};
        """
        cur.execute(load_sql2, (CSV_FILE,))

    duration = time.time() - start
    print(f"LOAD DATA completed in {duration:.1f} seconds.")

    cur.execute("SELECT COUNT(*) FROM transactions;")
    total = cur.fetchone()[0]
    print("Rows in transactions after load:", total)

    # Post-load: set order_year/month/quarter from order_date if missing
    print("Backfilling order_year/order_month/order_quarter from order_date where missing...")
    cur.execute("""
        UPDATE transactions
        SET order_year = COALESCE(NULLIF(order_year,0), YEAR(order_date)),
            order_month = COALESCE(NULLIF(order_month,0), MONTH(order_date)),
            order_quarter = COALESCE(NULLIF(order_quarter,0), QUARTER(order_date))
        WHERE order_date IS NOT NULL;
    """)
    conn.commit()

    # Quick sanity counts
    cur.execute("SELECT COUNT(*) FROM transactions WHERE order_date IS NULL OR order_date='0000-00-00';")
    null_dates = cur.fetchone()[0]
    print("Rows with NULL/0000-00-00 order_date after load:", null_dates)

    # sample
    print("Sample rows with non-null order_date (limit 10):")
    cur.execute("SELECT transaction_id, order_date, order_year, order_month, final_amount_inr FROM transactions WHERE order_date IS NOT NULL LIMIT 10;")
    for r in cur.fetchall():
        print(r)

finally:
    cur.close()
    conn.close()
print("✅ Load process completed.")