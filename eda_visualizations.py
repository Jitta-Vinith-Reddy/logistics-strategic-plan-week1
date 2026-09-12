"""
Week 3 - Exploratory Data Analysis & Visualizations
FreshCart Supplies - Logistics Data Analysis
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
df = pd.read_csv("freshcart_shipments.csv", parse_dates=["order_date"])

# ---------------------------------------------------------------------------
# Central tendency / summary stats
# ---------------------------------------------------------------------------
summary = df[["delivery_time_days", "shipment_volume_units", "transport_cost", "cost_per_km"]].describe()
summary.to_csv("summary_stats.csv")
print(summary)

corr = df[["distance_km", "shipment_volume_units", "delivery_time_days", "transport_cost", "cost_per_km"]].corr()
print(corr)

on_time_rate_by_region = df.groupby("region")["on_time"].mean().sort_values()
print(on_time_rate_by_region)

# ---------------------------------------------------------------------------
# 1. Distribution of delivery times (histogram + KDE)
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 4.2))
sns.histplot(df["delivery_time_days"], bins=30, kde=True, color="#2f6fed")
plt.axvline(df["delivery_time_days"].mean(), color="#d62728", linestyle="--", label=f"Mean = {df['delivery_time_days'].mean():.2f} days")
plt.axvline(df["delivery_time_days"].median(), color="#2ca02c", linestyle="--", label=f"Median = {df['delivery_time_days'].median():.2f} days")
plt.title("Distribution of Delivery Times")
plt.xlabel("Delivery Time (days)")
plt.ylabel("Number of Shipments")
plt.legend()
plt.tight_layout()
plt.savefig("img/01_delivery_time_hist.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 2. Delivery time by region (boxplot) - bottleneck identification
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 4.2))
order = df.groupby("region")["delivery_time_days"].median().sort_values().index
sns.boxplot(data=df, x="region", y="delivery_time_days", order=order, palette="Blues")
plt.title("Delivery Time by Region")
plt.xlabel("Region")
plt.ylabel("Delivery Time (days)")
plt.tight_layout()
plt.savefig("img/02_delivery_time_by_region_box.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 3. Shipment volume vs transport cost (scatter, colored by mode)
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 4.2))
sns.scatterplot(data=df, x="shipment_volume_units", y="transport_cost", hue="transport_mode", alpha=0.55, s=28)
plt.title("Shipment Volume vs Transport Cost")
plt.xlabel("Shipment Volume (units)")
plt.ylabel("Transport Cost ($)")
plt.legend(title="Mode")
plt.tight_layout()
plt.savefig("img/03_volume_vs_cost_scatter.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 4. Correlation heatmap
# ---------------------------------------------------------------------------
plt.figure(figsize=(6.5, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
plt.title("Correlation Matrix - Key Logistics Variables")
plt.tight_layout()
plt.savefig("img/04_correlation_heatmap.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 5. Daily average delivery time trend (line chart) - shows disruption
# ---------------------------------------------------------------------------
daily = df.groupby(df["order_date"].dt.date)["delivery_time_days"].mean()
plt.figure(figsize=(8.5, 4.2))
plt.plot(daily.index, daily.values, color="#2f6fed", linewidth=1.4)
plt.axvspan(pd.Timestamp("2026-06-12"), pd.Timestamp("2026-06-19"), color="red", alpha=0.15, label="Disruption window")
plt.title("Average Delivery Time Trend (Daily)")
plt.xlabel("Date")
plt.ylabel("Avg Delivery Time (days)")
plt.xticks(rotation=40)
plt.legend()
plt.tight_layout()
plt.savefig("img/05_delivery_time_trend.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 6. Cost per km by transport mode (bar chart)
# ---------------------------------------------------------------------------
plt.figure(figsize=(6.5, 4.2))
mode_cost = df.groupby("transport_mode")["cost_per_km"].mean().sort_values()
sns.barplot(x=mode_cost.index, y=mode_cost.values, palette="mako")
plt.title("Average Cost per Km by Transport Mode")
plt.xlabel("Transport Mode")
plt.ylabel("Cost per Km ($)")
plt.tight_layout()
plt.savefig("img/06_cost_per_km_by_mode.png", dpi=150)
plt.close()

print("All charts saved.")
