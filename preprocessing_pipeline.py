"""
Week 2 - Data Collection, Cleaning & Preprocessing
FreshCart Supplies - Logistics Data Analysis

Reference dataset: DataCo Smart Supply Chain for Big Data Analysis (Kaggle)
Simulates the WMS + POS + telematics merge FreshCart would pull from.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# ---------------------------------------------------------------------------
# 1. Load & inspect
# ---------------------------------------------------------------------------
df = pd.read_csv("dataco_supply_chain.csv", encoding="latin-1")
print(df.shape)
print(df.isnull().sum().sort_values(ascending=False).head(10))

# ---------------------------------------------------------------------------
# 2. Handle missing values
# ---------------------------------------------------------------------------
# Standardize and coerce bad date strings instead of dropping outright
df["order_date"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
df["shipping_date"] = pd.to_datetime(df["shipping date (DateOrders)"], errors="coerce")

# Region-level imputation for missing zip codes (avoids losing ~2-3% of rows)
region_mode_zip = df.groupby("Order Region")["Order Zipcode"].transform(
    lambda s: s.mode().iat[0] if not s.mode().empty else pd.NA
)
df["Order Zipcode"] = df["Order Zipcode"].fillna(region_mode_zip)

# ---------------------------------------------------------------------------
# 3. Outlier detection
# ---------------------------------------------------------------------------
def flag_iqr_outliers(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (series < lower) | (series > upper)


df["profit_outlier"] = flag_iqr_outliers(df["Order Profit Per Order"])

# Business-rule errors: negative quantity, ship date before order date
bad_qty = df["Order Item Quantity"] < 0
bad_dates = df["shipping_date"] < df["order_date"]
df_clean = df[~(bad_qty | bad_dates)].copy()

# ---------------------------------------------------------------------------
# 4. Normalization & encoding
# ---------------------------------------------------------------------------
num_cols = ["avg_weekly_demand", "demand_variability", "order_frequency"]
scaler = StandardScaler()
df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])

delivery_order = [["On Time", "Late", "Very Late"]]
df_clean["delivery_status_enc"] = OrdinalEncoder(categories=delivery_order).fit_transform(
    df_clean[["Delivery Status"]]
)

df_clean = pd.get_dummies(df_clean, columns=["Shipping Mode"], drop_first=True)

print("Cleaned shape:", df_clean.shape)
