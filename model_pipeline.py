"""
Week 4 - Predictive Modeling and Optimization in Logistics Systems
FreshCart Supplies - forecasting delivery_time_days

Dataset schema matches Week 3: order_date, region, transport_mode,
distance_km, shipment_volume_units, delivery_time_days, transport_cost,
cost_per_km, on_time.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------------
# 1. Recreate the Week 3 dataset (same generative structure)
# ---------------------------------------------------------------------------
N = 3000
regions = ["North", "South", "East", "West", "Central"]
region_dist = {"Central": 60, "East": 140, "North": 170, "South": 260, "West": 330}
modes = ["Road", "Rail", "Air"]
mode_probs = [0.62, 0.23, 0.15]
mode_cost_per_km = {"Road": 0.90, "Rail": 0.55, "Air": 2.80}
mode_speed_kmpd = {"Road": 90, "Rail": 70, "Air": 550}

dates = pd.date_range("2026-05-01", "2026-07-29", periods=N)
df = pd.DataFrame({
    "order_date": dates,
    "region": np.random.choice(regions, N, p=[0.22, 0.20, 0.18, 0.20, 0.20]),
    "transport_mode": np.random.choice(modes, N, p=mode_probs),
})
df["distance_km"] = df["region"].map(region_dist) + np.random.normal(0, 20, N)
df["distance_km"] = df["distance_km"].clip(lower=15)
df["shipment_volume_units"] = np.random.gamma(4, 35, N).round().astype(int)

base_speed = df["transport_mode"].map(mode_speed_kmpd)
df["delivery_time_days"] = (df["distance_km"] / base_speed + np.random.normal(0.3, 0.35, N)).clip(lower=0.2)

# mid-June disruption event, matching Week 3
disruption = (df["order_date"] >= "2026-06-12") & (df["order_date"] <= "2026-06-19")
df.loc[disruption, "delivery_time_days"] += np.random.uniform(1.5, 3.5, disruption.sum())

df["transport_cost"] = (
    df["distance_km"] * df["transport_mode"].map(mode_cost_per_km)
    + df["shipment_volume_units"] * 0.15
    + np.random.normal(0, 15, N)
).clip(lower=10)
df["cost_per_km"] = df["transport_cost"] / df["distance_km"]
df["on_time"] = df["delivery_time_days"] <= 3.0

df.to_csv("logistics_week4.csv", index=False)

# ---------------------------------------------------------------------------
# 2. Problem definition: forecast delivery_time_days
# ---------------------------------------------------------------------------
features = ["distance_km", "shipment_volume_units", "region", "transport_mode"]
target = "delivery_time_days"

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cat_cols = ["region", "transport_mode"]
num_cols = ["distance_km", "shipment_volume_units"]
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first"), cat_cols),
], remainder="passthrough")

# ---------------------------------------------------------------------------
# 3. Model 1: Linear Regression (baseline)
# ---------------------------------------------------------------------------
lin_pipe = Pipeline([("prep", preprocess), ("model", LinearRegression())])
lin_pipe.fit(X_train, y_train)
pred_lin = lin_pipe.predict(X_test)

# ---------------------------------------------------------------------------
# 4. Model 2: Random Forest (ensemble)
# ---------------------------------------------------------------------------
rf_pipe = Pipeline([("prep", preprocess), ("model", RandomForestRegressor(random_state=42))])
rf_pipe.fit(X_train, y_train)
pred_rf = rf_pipe.predict(X_test)

# ---------------------------------------------------------------------------
# 5. Model 3: Gradient Boosting (ensemble, tuned)
# ---------------------------------------------------------------------------
gb_pipe = Pipeline([("prep", preprocess), ("model", GradientBoostingRegressor(random_state=42))])

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [2, 3, 4],
    "model__learning_rate": [0.05, 0.1],
}
grid = GridSearchCV(gb_pipe, param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
grid.fit(X_train, y_train)
best_gb = grid.best_estimator_
pred_gb = best_gb.predict(X_test)

# ---------------------------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------------------------
def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    print(f"{name:22s} MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")
    return mae, rmse, r2

print("\n--- Test set performance ---")
m_lin = evaluate("Linear Regression", y_test, pred_lin)
m_rf = evaluate("Random Forest", y_test, pred_rf)
m_gb = evaluate("Gradient Boosting (tuned)", y_test, pred_gb)

print("\nBest GB params:", grid.best_params_)

# 5-fold CV on the best model
cv_scores = cross_val_score(best_gb, X, y, cv=5, scoring="neg_root_mean_squared_error")
print("5-fold CV RMSE (best GB):", (-cv_scores).round(3), "mean:", round(-cv_scores.mean(), 3))

# ---------------------------------------------------------------------------
# 7. Feature importance
# ---------------------------------------------------------------------------
ohe = best_gb.named_steps["prep"].named_transformers_["cat"]
cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
all_feature_names = cat_feature_names + num_cols
importances = best_gb.named_steps["model"].feature_importances_
imp_df = pd.DataFrame({"feature": all_feature_names, "importance": importances}).sort_values(
    "importance", ascending=False
)
print("\nFeature importances:\n", imp_df)

# ---------------------------------------------------------------------------
# 8. Charts
# ---------------------------------------------------------------------------
# 8.1 Actual vs predicted
plt.figure(figsize=(6, 5.5))
plt.scatter(y_test, pred_gb, alpha=0.4, s=18, color="#2E5C8A")
lims = [0, max(y_test.max(), pred_gb.max()) + 0.5]
plt.plot(lims, lims, "r--", label="Perfect prediction")
plt.xlabel("Actual Delivery Time (days)")
plt.ylabel("Predicted Delivery Time (days)")
plt.title("Actual vs Predicted Delivery Time (Gradient Boosting)")
plt.legend()
plt.tight_layout()
plt.savefig("chart1_actual_vs_predicted.png", dpi=150)
plt.close()

# 8.2 Residuals
residuals = y_test.values - pred_gb
plt.figure(figsize=(6.5, 4.5))
sns.histplot(residuals, bins=30, kde=True, color="#4C8C4A")
plt.axvline(0, color="crimson", linestyle="--")
plt.title("Residual Distribution (Gradient Boosting)")
plt.xlabel("Residual (Actual - Predicted, days)")
plt.tight_layout()
plt.savefig("chart2_residuals.png", dpi=150)
plt.close()

# 8.3 Feature importance
plt.figure(figsize=(6.5, 4.5))
sns.barplot(data=imp_df, x="importance", y="feature", color="#2E5C8A")
plt.title("Feature Importance (Gradient Boosting)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("chart3_feature_importance.png", dpi=150)
plt.close()

# 8.4 Model comparison bar chart
comp = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "Gradient Boosting (tuned)"],
    "RMSE": [m_lin[1], m_rf[1], m_gb[1]],
    "MAE": [m_lin[0], m_rf[0], m_gb[0]],
})
comp_melt = comp.melt(id_vars="Model", var_name="Metric", value_name="Days")
plt.figure(figsize=(7, 4.5))
sns.barplot(data=comp_melt, x="Model", y="Days", hue="Metric", palette=["#2E5C8A", "#8AA9C9"])
plt.title("Model Comparison: MAE and RMSE")
plt.ylabel("Error (days)")
plt.tight_layout()
plt.savefig("chart4_model_comparison.png", dpi=150)
plt.close()

print("\nCharts saved.")
