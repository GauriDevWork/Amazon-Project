# streamlit_exec.py
"""
Executive Summary page for Amazon India (Q1)
- Connects to MySQL transactions table
- Calculates: Total Revenue, Latest Year Revenue, Active Customers, AOV, YoY growth, Top Categories
- Defensive: skips invalid year/month rows and handles missing money values
"""

import streamlit as st
import pandas as pd
import sqlalchemy
from sqlalchemy import text
import altair as alt
import datetime

# ---------- CONFIG ----------
DB_HOST = "127.0.0.1"
DB_PORT = 3306
DB_USER = "root"        # change if needed
DB_PASS = ""            # change if needed
DB_NAME = "amazon_analytics"

# set page
st.set_page_config(page_title="Executive Summary — Amazon India", layout="wide")
st.title("Executive Summary — Amazon India (2015–2025)")

# DB engine
@st.cache_resource
def get_engine():
    url = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return sqlalchemy.create_engine(url, pool_pre_ping=True, pool_recycle=3600)

engine = get_engine()

# helper to safely run SQL and return DataFrame
def run_sql(sql, params=None):
    with engine.connect() as conn:
        df = pd.read_sql(sql=text(sql), con=conn, params=params)
    return df

# ---------- KPIs ----------
# total revenue (all time)
q_total_rev = "SELECT SUM(final_amount_inr) AS total_revenue FROM transactions WHERE final_amount_inr IS NOT NULL;"
total_rev = run_sql(q_total_rev).iloc[0,0] or 0.0

# revenue by year (filter to sensible range)
q_year = """
SELECT order_year AS year,
       SUM(final_amount_inr) AS revenue
FROM transactions
WHERE order_year BETWEEN 2015 AND 2025
  AND final_amount_inr IS NOT NULL
GROUP BY order_year
ORDER BY order_year;
"""
df_year = run_sql(q_year)

# latest year revenue
if not df_year.empty:
    latest_year = int(df_year['year'].max())
    latest_rev = float(df_year.loc[df_year['year'] == latest_year, 'revenue'].sum())
else:
    latest_year = None
    latest_rev = 0.0

# active customers
q_customers = "SELECT COUNT(DISTINCT customer_id) AS active_customers FROM transactions WHERE customer_id IS NOT NULL;"
active_customers = int(run_sql(q_customers).iloc[0,0] or 0)

# AOV
q_aov = "SELECT ROUND(AVG(final_amount_inr),2) AS aov FROM transactions WHERE final_amount_inr IS NOT NULL;"
aov = float(run_sql(q_aov).iloc[0,0] or 0.0)

# Top categories
q_top_cat = """
SELECT COALESCE(NULLIF(category,''),'(Unknown)') AS category,
       SUM(final_amount_inr) AS revenue,
       COUNT(*) AS orders
FROM transactions
WHERE final_amount_inr IS NOT NULL
GROUP BY category
ORDER BY revenue DESC
LIMIT 15;
"""
df_top_cat = run_sql(q_top_cat)

# show KPIs in row
col1, col2, col3, col4 = st.columns([1,1,1,1])
col1.metric("Total Revenue (₹)", f"{total_rev:,.0f}")
if latest_year:
    col2.metric(f"Latest Year Revenue ({latest_year})", f"{latest_rev:,.0f}")
else:
    col2.metric("Latest Year Revenue", "N/A")
col3.metric("Active Customers", f"{active_customers:,}")
col4.metric("Average Order Value (₹)", f"{aov:,.2f}")

st.markdown("---")

# show warnings if monthly/year data looks invalid
if df_year.empty or df_year['year'].max() < 2015:
    st.warning("No valid yearly revenue rows found in `order_year` (filtered 2015-2025). Check data cleaning step or `order_year` population.")
else:
    # Yearly revenues chart + YoY table
    df_year['revenue'] = df_year['revenue'].astype(float)
    df_year['yoy_pct'] = df_year['revenue'].pct_change()*100
    df_year['yoy_pct'] = df_year['yoy_pct'].round(2)

    st.subheader("Yearly Revenue & YoY Growth")
    left, right = st.columns([2,1])

    # line chart
    left.altair_chart(
        alt.Chart(df_year).mark_line(point=True).encode(
            x=alt.X('year:O', title='Year'),
            y=alt.Y('revenue:Q', title='Revenue (₹)'),
            tooltip=['year','revenue']
        ).properties(height=300),
        use_container_width=True
    )

    # YoY table on right
    right.write("YoY table")
    st.dataframe(df_year.rename(columns={'year':'Year','revenue':'Revenue','yoy_pct':'YoY Growth (%)'}).fillna('-'), height=300)

st.markdown("---")

# Top performing categories bar chart
st.subheader("Top Performing Categories (by Revenue)")
if df_top_cat.empty:
    st.info("No category data to show.")
else:
    chart = alt.Chart(df_top_cat).mark_bar().encode(
        x=alt.X('revenue:Q', title='Revenue (₹)'),
        y=alt.Y('category:N', sort='-x', title='Category'),
        tooltip=['category','revenue','orders']
    ).properties(height=450)
    st.altair_chart(chart, use_container_width=True)

# Top products table
st.subheader("Top Products (by Revenue)")
q_top_prod = """
SELECT product_id, product_name, SUM(final_amount_inr) AS revenue, COUNT(*) AS orders
FROM transactions
GROUP BY product_id, product_name
ORDER BY revenue DESC
LIMIT 20;
"""
df_top_prod = run_sql(q_top_prod)
st.dataframe(df_top_prod.fillna('-'), height=300)

st.markdown("---")
st.caption(f"Data source: transactions table. Last refreshed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
