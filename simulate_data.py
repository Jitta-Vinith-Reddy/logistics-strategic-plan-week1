"""
Week 3 - Simulate hypothetical logistics dataset
FreshCart Supplies - 90 days of shipments across 5 regions
"""
import numpy as np
import pandas as pd

np.random.seed(42)

n = 3000
regions = ["North", "South", "East", "West", "Central"]
region_weights = [0.22, 0.18, 0.25, 0.15, 0.20]
modes = ["Road", "Rail", "Air"]
mode_weights = [0.72, 0.20, 0.08]

region = np.random.choice(regions, size=n, p=region_weights)
mode = np.random.choice(modes, size=n, p=mode_weights)

# Base delivery time by region (some regions structurally slower - e.g. farther from DC)
region_base_time = {"North": 2.2, "South": 3.4, "East": 1.8, "West": 4.1, "Central": 1.4}
mode_time_adj = {"Road": 0.0, "Rail": 0.6, "Air": -1.0}

delivery_days = np.array([
    max(0.3, np.random.normal(region_base_time[r] + mode_time_adj[m], 0.9))
    for r, m in zip(region, mode)
])

shipment_volume = np.random.gamma(shape=4.0, scale=35, size=n)  # units per shipment

mode_cost_per_km = {"Road": 0.9, "Rail": 0.55, "Air": 2.8}
distance_km = np.random.uniform(20, 420, size=n)
transport_cost = np.array([
    distance_km[i] * mode_cost_per_km[mode[i]] * (1 + 0.15 * np.random.randn())
    for i in range(n)
])
transport_cost = np.clip(transport_cost, 15, None)

order_date = pd.to_datetime("2026-05-01") + pd.to_timedelta(
    np.random.randint(0, 90, size=n), unit="D"
)

df = pd.DataFrame({
    "order_date": order_date,
    "region": region,
    "transport_mode": mode,
    "distance_km": distance_km.round(1),
    "shipment_volume_units": shipment_volume.round(0),
    "delivery_time_days": delivery_days.round(2),
    "transport_cost": transport_cost.round(2),
})

# Inject a delivery-time disruption in week 7 (simulating a monsoon/road-closure event)
disruption_mask = (df["order_date"] >= "2026-06-12") & (df["order_date"] <= "2026-06-19")
df.loc[disruption_mask, "delivery_time_days"] += np.random.uniform(1.5, 3.0, disruption_mask.sum())

df["on_time"] = df["delivery_time_days"] <= (df["region"].map(region_base_time) + 1.5)
df["cost_per_km"] = (df["transport_cost"] / df["distance_km"]).round(3)

df.to_csv("freshcart_shipments.csv", index=False)
print(df.shape)
print(df.head())
print(df.describe(numeric_only=True))
