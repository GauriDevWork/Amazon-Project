import pandas as pd
import numpy as np
import re
from dateutil import parser
from datetime import datetime

# --- Q1: Date standardization ---
def parse_date_safe(d):
    """Parse multiple date formats to YYYY-MM-DD safely."""
    if pd.isna(d):
        return pd.NaT
    s = str(d).strip()
    for fmt in ("%d/%m/%Y", "%d-%m-%y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt).date()
        except:
            pass
    try:
        dt = parser.parse(s, dayfirst=True, fuzzy=True)
        return dt.date()
    except:
        return pd.NaT

def clean_order_date(df, col='order_date'):
    df[col + '_clean'] = df[col].apply(parse_date_safe)
    df[col + '_clean'] = pd.to_datetime(df[col + '_clean'], errors='coerce').dt.date
    return df

# --- Q2: Price cleaning ---
def clean_price_col(x):
    """Remove ₹, commas, text like 'Price on Request' and keep numeric value."""
    if pd.isna(x): 
        return np.nan
    s = str(x)
    if re.search(r'price on request', s, re.I):
        return np.nan
    s = re.sub(r'[^\d.\-]', '', s)
    if s.count('.') > 1:
        parts = s.split('.')
        s = ''.join(parts[:-1]) + '.' + parts[-1]
    try:
        return float(s)
    except:
        return np.nan

def clean_original_price(df, col='original_price_inr'):
    df[col + '_clean'] = df[col].apply(clean_price_col)
    return df

# --- Q3: Rating normalization ---
def normalize_rating(x):
    """Convert ratings like '4 stars', '3/5', '80%' to numeric 1-5 scale."""
    if pd.isna(x): 
        return np.nan
    s = str(x).strip().lower()
    m = re.search(r'(\d+(\.\d+)?)', s)
    if not m:
        return np.nan
    val = float(m.group(1))
    if val > 5 and val <= 100:
        return round((val / 100) * 5, 2)
    if val > 5:
        return np.nan
    return round(val, 2)

def clean_ratings(df, col='product_rating'):
    df[col + '_clean'] = df[col].apply(normalize_rating)
    return df

# --- Q4: City standardization ---
CITY_MAP = {
    'Bengaluru': ['Bangalore', 'Bengaluru', 'Bengalooru', 'Bengalore'],
    'Mumbai': ['Mumbai', 'Bombay'],
    'New Delhi': ['Delhi', 'New Delhi', 'N.Delhi'],
    'Hyderabad': ['Hyderabad', 'Hydrabad'],
    'Chennai': ['Chennai', 'Madras']
}

VARIANT_TO_CANON = {}
for canon, variants in CITY_MAP.items():
    for v in variants:
        VARIANT_TO_CANON[v.lower()] = canon

def clean_city(x):
    """Normalize city names to canonical form."""
    if pd.isna(x): 
        return np.nan
    s = str(x).strip().lower()
    s = re.sub(r'[^a-z\s]', '', s)
    s = ' '.join(s.split())
    return VARIANT_TO_CANON.get(s, s.title())

def clean_customer_city(df, col='customer_city'):
    df[col + '_clean'] = df[col].apply(clean_city)
    return df

# --- Q5: Boolean normalization ---
def bool_to_bool(x):
    """Convert mixed boolean forms (True/False, Y/N, 1/0, Yes/No) to Python True/False."""
    if pd.isna(x): 
        return np.nan
    s = str(x).strip().lower()
    if s in ('true', 't', '1', 'yes', 'y'):
        return True
    if s in ('false', 'f', '0', 'no', 'n'):
        return False
    try:
        v = float(s)
        return bool(v)
    except:
        return np.nan

def clean_boolean_cols(df, cols):
    for c in cols:
        df[c + '_clean'] = df[c].apply(bool_to_bool)
    return df

# --- Q6: Category normalization ---
def normalize_category(cat):
    """Standardize category variants (case, typos, symbols)."""
    if pd.isna(cat): 
        return np.nan
    s = str(cat).lower()
    s = s.replace('&', 'and').replace('/', ' ')
    s = re.sub(r'[^a-z0-9\s]', '', s).strip()
    if 'electronic' in s:
        return 'Electronics'
    if 'fashion' in s or 'clothing' in s:
        return 'Fashion'
    if 'home' in s:
        return 'Home & Kitchen'
    if 'beauty' in s:
        return 'Beauty & Personal Care'
    return s.title()

def clean_category(df, col='category'):
    df[col + '_clean'] = df[col].apply(normalize_category)
    return df

# --- Q7: Delivery days cleaning ---
def parse_delivery_days(x):
    """Convert text like 'Same Day', '1-2 days', '5 days' into numeric."""
    if pd.isna(x): 
        return np.nan
    s = str(x).strip().lower()
    if 'same' in s:
        return 0
    m = re.search(r'(\d+)\s*-\s*(\d+)', s)
    if m:
        return (int(m.group(1)) + int(m.group(2))) / 2.0
    m = re.search(r'(\d+)', s)
    if m:
        val = int(m.group(1))
        if val < 0 or val > 30:
            return np.nan
        return val
    return np.nan

def clean_delivery_days(df, col='delivery_days'):
    df[col + '_clean'] = df[col].apply(parse_delivery_days)
    return df

# --- Q8: Duplicates ---
def mark_duplicates(df):
    """Mark possible duplicate transactions."""
    df['dup_key'] = (
        df['customer_id'].astype(str) + '|' +
        df['product_id'].astype(str) + '|' +
        df['order_date'].astype(str) + '|' +
        df['final_amount_inr'].astype(str)
    )
    df['is_duplicate'] = df.duplicated('dup_key', keep=False)
    return df

# --- Q9: Outlier correction ---
def correct_price_outliers(df, price_col='original_price_inr_clean'):
    """Adjust extremely high prices (e.g., 100x error) using category medians."""
    if 'category_clean' not in df.columns:
        print("Warning: category_clean not found; skipping outlier correction.")
        return df
    df['price_median_cat'] = df.groupby('category_clean')[price_col].transform('median')

    def fix(x, med):
        if pd.isna(x) or pd.isna(med):
            return x
        if x > 50 * med:
            return x / 100
        if x > 20 * med:
            return x / 10
        return x

    df[price_col + '_adj'] = df.apply(lambda r: fix(r[price_col], r['price_median_cat']), axis=1)
    return df

# --- Q10: Payment methods ---
PAYMENT_MAP = {
    'UPI': ['upi', 'phonepe', 'googlepay', 'gpay', 'paytm'],
    'Credit Card': ['credit card', 'cc', 'mastercard', 'visa'],
    'Debit Card': ['debit card'],
    'Netbanking': ['netbanking', 'net banking', 'internet banking'],
    'COD': ['cod', 'cash on delivery', 'c.o.d'],
    'Wallet': ['wallet', 'amazon pay', 'paytm wallet']
}

def standardize_payment(x):
    """Normalize payment method names."""
    if pd.isna(x):
        return np.nan
    s = str(x).lower()
    for canon, variants in PAYMENT_MAP.items():
        for v in variants:
            if v in s:
                return canon
    return s.title()

def clean_payment_method(df, col='payment_method'):
    df[col + '_clean'] = df[col].apply(standardize_payment)
    return df
