"""
Data Loading & Cleaning
FreshCart Supplies - Logistics Data Analysis (Week 1)
"""
import pandas as pd

sales = pd.read_csv("sales_transactions.csv", parse_dates=["order_date"])


def clean_sales(df):
    df = df.drop_duplicates(subset=["order_id", "sku"])
    df = df[df["quantity"] > 0]
    df["revenue"] = df["quantity"] * df["unit_price"]
    df["demand_censored"] = df["on_hand_qty"] == 0  # stockout flag
    return df


sales_clean = clean_sales(sales)
