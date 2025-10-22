import streamlit as st, pandas as pd, plotly.express as px
from sqlalchemy import text
kpi_sql = """SELECT SUM(final_amount_inr) AS total_revenue, COUNT(DISTINCT customer_id) AS active_customers, AVG(final_amount_inr) AS avg_order_value FROM transactions;"""
trend_sql = """SELECT order_year, SUM(final_amount_inr) AS revenue FROM transactions GROUP BY order_year ORDER BY order_year;"""
k = run_query(kpi_sql).iloc[0]
st.metric("Total Revenue (₹)", f"{k.total_revenue:,.0f}")
st.metric("Active Customers", f"{k.active_customers:,}")
st.metric("Avg Order Value (₹)", f"{k.avg_order_value:,.0f}")
df_tr = run_query(trend_sql)
df_tr['period'] = pd.to_datetime(df_tr['order_year'].astype(str)+'-01-01')
fig = px.line(df_tr, x='period', y='revenue', title='Yearly Revenue')
st.plotly_chart(fig, use_container_width=True)
