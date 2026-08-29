"""
EDA - Seasonality & ABC Classification
FreshCart Supplies - Logistics Data Analysis (Week 1)
"""
import pandas as pd
import matplotlib.pyplot as plt

from data_cleaning import sales_clean

weekly = (
    sales_clean.groupby([pd.Grouper(key="order_date", freq="W"), "sku"])["quantity"]
    .sum()
    .unstack(fill_value=0)
)

weekly[weekly.sum().nlargest(5).index].plot(figsize=(12, 5))
plt.title("Weekly Demand - Top 5 SKUs")
plt.show()

rev = sales_clean.groupby("sku")["revenue"].sum().sort_values(ascending=False)
cum_pct = rev.cumsum() / rev.sum()
abc = pd.cut(cum_pct, bins=[0, 0.8, 0.95, 1.0], labels=["A", "B", "C"])
