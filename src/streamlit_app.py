import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import plotly.express as px
import plotly.graph_objects as go
from functools import lru_cache
from datetime import datetime

# ------------------ CONFIG ------------------
DB_USER = "root"
DB_PASS = ""         # set if you use password
DB_HOST = "127.0.0.1"
DB_PORT = 3306
DB_NAME = "amazon_analytics"

# Default date window for UI (will clamp to data range)
DEFAULT_START = "2015-01-01"
DEFAULT_END = "2025-12-31"
# --------------------------------------------

st.set_page_config(layout="wide", page_title="Amazon India — Decade Analytics")

# ---------- DB engine & helpers ----------
@st.cache_resource(show_spinner=False)
def get_engine():
    url = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url, pool_pre_ping=True, pool_recycle=3600)

engine = get_engine()

# Generic safe SQL reader with caching
@st.cache_data(ttl=300, show_spinner=False)
def run_sql(query, params=None):
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)
        return df
    except Exception as e:
        st.error("SQL error: " + str(e))
        return pd.DataFrame()

# Small helper to fetch min/max date in transactions
@st.cache_data(ttl=600, show_spinner=False)
def get_date_range():
    q = "SELECT MIN(order_date) AS min_date, MAX(order_date) AS max_date FROM transactions;"
    df = run_sql(q)
    if df.empty or pd.isna(df.loc[0, "min_date"]):
        return DEFAULT_START, DEFAULT_END

    def to_str_date(x):
        if x is None or pd.isna(x):
            return None
        if hasattr(x, "date"):  # pandas.Timestamp or datetime
            return str(x.date())
        return str(x)  # already a date

    min_d = to_str_date(df.loc[0, "min_date"])
    max_d = to_str_date(df.loc[0, "max_date"])
    return min_d or DEFAULT_START, max_d or DEFAULT_END

# ---------- UI: Sidebar filters ----------
st.sidebar.title("Filters")
min_date, max_date = get_date_range()
start_date = st.sidebar.date_input("Start date", pd.to_datetime(min_date))
end_date = st.sidebar.date_input("End date", pd.to_datetime(max_date))
if start_date > end_date:
    st.sidebar.error("Start date must be <= end date")

# category list (small lookup table)
@st.cache_data(ttl=600)
def get_categories():
    q = "SELECT DISTINCT category FROM products WHERE category IS NOT NULL;"
    df = run_sql(q)
    return sorted(df['category'].dropna().astype(str).tolist())

cats = get_categories()
selected_cat = st.sidebar.multiselect("Category (filter)", options=cats, default=[])

prime_opt = st.sidebar.selectbox("Prime filter", ["All", "Only Prime", "Only Non-Prime"])

# helper to build WHERE clause and params
def build_filters():
    where = ["order_date BETWEEN :start AND :end"]
    params = {"start": start_date.strftime("%Y-%m-%d"), "end": end_date.strftime("%Y-%m-%d")}
    if selected_cat:
        where.append("category IN :cats")
        params["cats"] = tuple(selected_cat)
    if prime_opt == "Only Prime":
        where.append("is_prime_member = 1")
    elif prime_opt == "Only Non-Prime":
        where.append("(is_prime_member = 0 OR is_prime_member IS NULL)")
    return " AND ".join(where), params

# ---------- Pages navigation ----------
pages = [
    "Executive",
    "Revenue Analytics",
    "Customer Analytics",
    "Product & Inventory",
    "Operations & Logistics",
    "Advanced Analytics"
]
page = st.sidebar.radio("Choose page", pages)

# ---------- Utility charts & KPI helpers ----------
def kpi_row(kpis):
    cols = st.columns(len(kpis))
    for col, (label, value, delta, fmt) in zip(cols, kpis):
        col.metric(label, fmt(value), fmt(delta) if delta is not None else "")

def safe_fill_dates(df):
    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    return df

# ---------- Executive page (Q1-Q5) ----------
if page == "Executive":
    st.title("Executive Summary — Amazon India (2015–2025)")
    where_clause, params = build_filters()

    # KPI fetches: total revenue, active customers, avg order value, YoY growth
    q_kpis = f"""
    SELECT 
      SUM(final_amount_inr) AS total_revenue,
      COUNT(DISTINCT customer_id) AS active_customers,
      AVG(final_amount_inr) AS avg_order_value,
      YEAR(order_date) AS year,
      SUM(final_amount_inr) AS year_revenue
    FROM transactions
    WHERE {where_clause}
    GROUP BY YEAR(order_date)
    ORDER BY YEAR(order_date) ASC;
    """
    df_year = run_sql(q_kpis, params)
    if df_year.empty:
        st.info("No data for selected filters/date range.")
    else:
        # compute totals from filtered set
        total_rev = df_year['year_revenue'].sum()
        active_customers_q = f"SELECT COUNT(DISTINCT customer_id) AS c FROM transactions WHERE {where_clause};"
        active_customers = int(run_sql(active_customers_q, params).loc[0, 'c'] or 0)

        avg_order_q = f"SELECT AVG(final_amount_inr) AS aov FROM transactions WHERE {where_clause};"
        aov = float(run_sql(avg_order_q, params).loc[0, 'aov'] or 0)

        # yoy growth - take last two years if available
        df_year_sorted = df_year.sort_values('year')
        if len(df_year_sorted) >= 2:
            last = df_year_sorted['year_revenue'].iloc[-1]
            prev = df_year_sorted['year_revenue'].iloc[-2]
            yoy = (last - prev) / prev * 100 if prev and prev != 0 else None
        else:
            yoy = None

        # Top categories
        q_cat = f"""
        SELECT p.category, SUM(t.final_amount_inr) AS revenue, COUNT(*) AS txns
        FROM transactions t
        LEFT JOIN products p ON TRIM(t.product_id) = TRIM(p.product_id)
        WHERE {where_clause}
        GROUP BY p.category
        ORDER BY revenue DESC
        LIMIT 8;
        """
        df_cat = run_sql(q_cat, params)

        # KPI row
        kpis = [
            ("Total Revenue (₹)", total_rev, yoy, lambda v: f"₹{v:,.0f}"),
            ("Active Customers", active_customers, None, lambda v: f"{int(v):,}"),
            ("Average Order Value (₹)", aov, None, lambda v: f"₹{v:,.2f}"),
            ("YoY Growth %", yoy if yoy is not None else 0, None, lambda v: f"{v:.1f}%" if v is not None else "N/A")
        ]
        kpi_row(kpis)

        # Yearly revenue time series with trendline and annotations
        st.subheader("Yearly Revenue & YOY Growth")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_year_sorted['year'].astype(int), y=df_year_sorted['year_revenue'], name='Revenue'))
        fig.add_trace(go.Scatter(x=df_year_sorted['year'].astype(int), y=df_year_sorted['year_revenue'].rolling(2).mean(), 
                                 mode='lines+markers', name='Rolling mean'))
        fig.update_layout(yaxis_title="Revenue (₹)", xaxis_title="Year", template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

        # Top categories chart
        if not df_cat.empty:
            st.subheader("Top Categories (Revenue)")
            fig2 = px.treemap(df_cat, path=['category'], values='revenue', title="Revenue by Category")
            st.plotly_chart(fig2, use_container_width=True)

        # Table: last 5 months performance (monthly trend)
        st.subheader("Recent Monthly Snapshot")
        q_monthly = f"""
        SELECT YEAR(order_date) AS yr, MONTH(order_date) AS m, SUM(final_amount_inr) AS revenue, COUNT(DISTINCT customer_id) AS customers
        FROM transactions
        WHERE {where_clause}
        GROUP BY YEAR(order_date), MONTH(order_date)
        ORDER BY YEAR(order_date) DESC, MONTH(order_date) DESC
        LIMIT 12;
        """
        df_month = run_sql(q_monthly, params)
        if not df_month.empty:
            df_month['period'] = df_month.apply(lambda r: f"{int(r.yr)}-{int(r.m):02d}", axis=1)
            st.dataframe(df_month[['period','revenue','customers']].sort_values('period', ascending=False).reset_index(drop=True))
        else:
            st.info("No recent monthly data.")

# ---------- Revenue Analytics (Q6-10) ----------
elif page == "Revenue Analytics":
    st.title("Revenue Analytics (Trends, Seasonality, Price/Demand)")

    where_clause, params = build_filters()
    # Revenue trend monthly
    q = f"""
    SELECT YEAR(order_date) AS yr, MONTH(order_date) AS m, SUM(final_amount_inr) AS revenue, COUNT(*) AS orders
    FROM transactions
    WHERE {where_clause}
    GROUP BY YEAR(order_date), MONTH(order_date)
    ORDER BY YEAR(order_date), MONTH(order_date);
    """
    df = run_sql(q, params)
    if df.empty:
        st.info("No data for selected filters")
    else:
        df['period'] = pd.to_datetime(df['yr'].astype(str) + "-" + df['m'].astype(str) + "-01")
        st.subheader("Monthly revenue trend")
        fig = px.line(df, x='period', y='revenue', title="Monthly Revenue", markers=True)
        st.plotly_chart(fig, use_container_width=True)

        # Seasonality heatmap - pivot year vs month
        st.subheader("Seasonality heatmap (Year vs Month)")
        pivot = df.pivot(index='yr', columns='m', values='revenue').fillna(0)
        pivot = pivot.sort_index()
        fig2 = px.imshow(pivot, labels=dict(x="Month", y="Year", color="Revenue"), 
                         x=[calendar_month for calendar_month in pivot.columns], 
                         y=pivot.index.astype(str))
        st.plotly_chart(fig2, use_container_width=True)

        # Price vs demand scatter (sample)
        st.subheader("Price vs Demand (sampled)")
        q_price = f"""
        SELECT final_amount_inr AS price, quantity, product_id, category
        FROM transactions
        WHERE {where_clause} AND final_amount_inr IS NOT NULL
        LIMIT 20000;
        """
        df_price = run_sql(q_price, params)
        if not df_price.empty:
            df_price = df_price.sample(min(8000, len(df_price)), random_state=42)
            fig3 = px.scatter(df_price, x='price', y='quantity', color='category', hover_data=['product_id'], title="Price vs Quantity")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Not enough price data.")

# ---------- Customer Analytics (Q11-15) ----------
elif page == "Customer Analytics":
    st.title("Customer Analytics (RFM, Segments, CLV)")
    where_clause, params = build_filters()

    st.subheader("RFM Snapshot (sample customers)")
    # Compute RFM from transactions: recency, frequency, monetary
    q_rfm = f"""
    SELECT customer_id, MAX(order_date) AS last_order, COUNT(*) AS frequency, SUM(final_amount_inr) AS monetary
    FROM transactions
    WHERE {where_clause} AND customer_id IS NOT NULL
    GROUP BY customer_id
    LIMIT 200000;
    """
    df_rfm = run_sql(q_rfm, params)
    if df_rfm.empty:
        st.info("No customer transactions")
    else:
        ref_date = pd.to_datetime(end_date)
        df_rfm['last_order'] = pd.to_datetime(df_rfm['last_order'], errors='coerce')
        df_rfm['recency_days'] = (ref_date - df_rfm['last_order']).dt.days
        # simple RFM quantiles
        df_rfm['r_score'] = pd.qcut(df_rfm['recency_days'].rank(method='first'), 4, labels=[4,3,2,1]).astype(int)
        df_rfm['f_score'] = pd.qcut(df_rfm['frequency'].rank(method='first'), 4, labels=[1,2,3,4]).astype(int)
        df_rfm['m_score'] = pd.qcut(df_rfm['monetary'].rank(method='first'), 4, labels=[1,2,3,4]).astype(int)
        df_rfm['rfm_score'] = df_rfm['r_score'].astype(str) + df_rfm['f_score'].astype(str) + df_rfm['m_score'].astype(str)
        top_customers = df_rfm.sort_values('monetary', ascending=False).head(20)
        st.dataframe(top_customers[['customer_id','recency_days','frequency','monetary']].head(50))

        # RFM scatter
        fig = px.scatter(df_rfm.sample(min(5000,len(df_rfm))), x='recency_days', y='monetary', size='frequency', hover_data=['customer_id'], title="RFM scatter (sample)")
        fig.update_layout(xaxis_title="Recency (days)", yaxis_title="Monetary (₹)")
        st.plotly_chart(fig, use_container_width=True)

# ---------- Product & Inventory (Q16-20) ----------
elif page == "Product & Inventory":
    st.title("Product & Inventory Analytics")
    where_clause, params = build_filters()

    st.subheader("Top Products by Revenue")
    q_prod = f"""
    SELECT t.product_id, COALESCE(p.product_name, t.product_name) AS product_name,
           SUM(t.final_amount_inr) AS revenue, COUNT(*) AS qty, AVG(t.product_rating) AS avg_rating
    FROM transactions t
    LEFT JOIN products p ON TRIM(t.product_id) = TRIM(p.product_id)
    WHERE {where_clause}
    GROUP BY t.product_id
    ORDER BY revenue DESC
    LIMIT 50;
    """
    df_prod = run_sql(q_prod, params)
    if df_prod.empty:
        st.info("No product data")
    else:
        st.dataframe(df_prod[['product_id','product_name','revenue','qty','avg_rating']])

    st.subheader("Category Performance (bar)")
    q_cat2 = f"""
    SELECT COALESCE(p.category, t.category) AS category, SUM(t.final_amount_inr) AS revenue, COUNT(*) AS orders
    FROM transactions t
    LEFT JOIN products p ON TRIM(t.product_id) = TRIM(p.product_id)
    WHERE {where_clause}
    GROUP BY category
    ORDER BY revenue DESC;
    """
    df_cat2 = run_sql(q_cat2, params)
    if not df_cat2.empty:
        fig = px.bar(df_cat2, x='category', y='revenue', title="Revenue by Category")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No category data")

# ---------- Operations & Logistics (Q21-25) ----------
elif page == "Operations & Logistics":
    st.title("Operations & Logistics (Delivery, Returns, Payments)")
    where_clause, params = build_filters()

    st.subheader("Delivery performance distribution")
    q_del = f"""
    SELECT delivery_days, COUNT(*) AS cnt, AVG(customer_rating) AS avg_rating
    FROM transactions
    WHERE {where_clause} AND delivery_days IS NOT NULL
    GROUP BY delivery_days
    ORDER BY delivery_days;
    """
    df_del = run_sql(q_del, params)
    if not df_del.empty:
        fig = px.bar(df_del, x='delivery_days', y='cnt', hover_data=['avg_rating'], title="Delivery days distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No delivery data")

    st.subheader("Payment methods evolution (Top methods)")
    q_pay = f"""
    SELECT payment_method, SUM(final_amount_inr) AS revenue, COUNT(*) AS txns
    FROM transactions
    WHERE {where_clause}
    GROUP BY payment_method
    ORDER BY txns DESC
    LIMIT 10;
    """
    df_pay = run_sql(q_pay, params)
    if not df_pay.empty:
        fig = px.pie(df_pay, names='payment_method', values='txns', title="Payment method share")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No payment method data")

# ---------- Advanced Analytics (Q26-30) ----------
elif page == "Advanced Analytics":
    st.title("Advanced Analytics (Forecasting, CLV, Cross-sell ideas)")
    where_clause, params = build_filters()

    st.subheader("Cohort retention (monthly)")
    # build cohort table aggregated by cohort_month and months since
    q_cohort = f"""
    SELECT customer_id, MIN(DATE_FORMAT(order_date, '%%Y-%%m-01')) AS cohort_month, DATE_FORMAT(order_date, '%%Y-%%m-01') AS order_month
    FROM transactions
    WHERE {where_clause} AND customer_id IS NOT NULL
    GROUP BY customer_id, DATE_FORMAT(order_date, '%%Y-%%m-01');
    """
    df_cohort = run_sql(q_cohort, params)
    if df_cohort.empty:
        st.info("No cohort data")
    else:
        df_cohort['cohort_month'] = pd.to_datetime(df_cohort['cohort_month'])
        df_cohort['order_month'] = pd.to_datetime(df_cohort['order_month'])
        df_cohort['months_since'] = ((df_cohort['order_month'].dt.year - df_cohort['cohort_month'].dt.year) * 12 +
                                     (df_cohort['order_month'].dt.month - df_cohort['cohort_month'].dt.month))
        cohort_pivot = df_cohort.groupby(['cohort_month','months_since']).agg(active_customers=('customer_id','nunique')).reset_index()
        # pivot small sample of recent cohorts
        recent = sorted(cohort_pivot['cohort_month'].dt.strftime('%Y-%m').unique())[-8:]
        pt = cohort_pivot[cohort_pivot['cohort_month'].dt.strftime('%Y-%m').isin(recent)]
        heat = pt.pivot(index='cohort_month', columns='months_since', values='active_customers').fillna(0)
        fig = px.imshow(heat, labels=dict(x="Months since cohort", y="Cohort month", color="Active customers"))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top cross-sell pairs (co-purchase) — sample")
    q_pairs = f"""
    SELECT p1.product_id AS prod_a, p2.product_id AS prod_b, COUNT(*) AS cnt
    FROM transactions t1
    JOIN transactions t2 ON t1.customer_id = t2.customer_id AND t1.transaction_id <> t2.transaction_id
    JOIN products p1 ON TRIM(t1.product_id)=TRIM(p1.product_id)
    JOIN products p2 ON TRIM(t2.product_id)=TRIM(p2.product_id)
    WHERE {where_clause}
    GROUP BY prod_a, prod_b
    ORDER BY cnt DESC
    LIMIT 50;
    """
    try:
        df_pairs = run_sql(q_pairs, params)
    except Exception:
        df_pairs = pd.DataFrame()
    if not df_pairs.empty:
        st.dataframe(df_pairs.head(20))
    else:
        st.info("Cross-sell pair extraction not available (dataset size or joins).")

# ---------- End pages ----------

