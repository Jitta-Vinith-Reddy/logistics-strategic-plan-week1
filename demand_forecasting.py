"""
Demand Forecasting (XGBoost with lag features)
FreshCart Supplies - Logistics Data Analysis (Week 1)

Note: X and y below refer to the feature matrix and target series
built from the cleaned/lagged sales data (see add_lags output).
"""
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error


def add_lags(df):
    for lag in [1, 2, 4, 52]:
        df[f"lag_{lag}"] = df.groupby("sku")["quantity"].shift(lag)
    df["roll_mean_4w"] = df.groupby("sku")["quantity"].shift(1).rolling(4).mean()
    return df.dropna()


model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05)
tscv = TimeSeriesSplit(n_splits=5)

# X, y = feature matrix / target built from add_lags(sales_clean)
for tr, te in tscv.split(X):
    model.fit(X.iloc[tr], y.iloc[tr])
    print("MAPE:", mean_absolute_percentage_error(y.iloc[te], model.predict(X.iloc[te])))
