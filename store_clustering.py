"""
Store Clustering (K-Means)
FreshCart Supplies - Logistics Data Analysis (Week 1)
"""
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from data_cleaning import sales_clean

features = sales_clean.groupby("store_id").agg(
    avg_weekly_demand=("quantity", "mean"),
    demand_variability=("quantity", "std"),
    order_frequency=("order_id", "nunique"),
)

X = StandardScaler().fit_transform(features)
features["cluster"] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)
