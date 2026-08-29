# Logistics Data Analysis — Week 1: Strategic Planning

**Intern:** Jitta Vinith Reddy
**Task:** Week 1 — Strategic Planning and Data Exploration in Logistics (Yuva Intern)

## Scenario

**FreshCart Supplies**, a mid-sized e-commerce company, operates one regional
distribution center (DC) serving ~120 retail stores within a 400 km radius.
Key challenges:

1. Stockouts at some stores and overstock at others — inflating holding costs ~18% annually
2. Manually planned delivery routes — rising fuel costs (+12% YoY)
3. Unpredictable seasonal demand handled reactively via expensive emergency shipments

**Objective:** Use historical sales, inventory, and delivery data to (a) forecast
demand per store, (b) segment SKUs/stores for tailored inventory policies, and
(c) optimize delivery routes.

## Key Performance Indicators (KPIs)

| KPI | Definition | Baseline | Target |
|---|---|---|---|
| Inventory Turnover Ratio | COGS ÷ Average Inventory | 6.2 | 8.0 |
| Order Fill Rate | % lines shipped complete from stock | 89% | 96% |
| Cost per km Delivered | Total delivery cost ÷ total km | $1.85 | $1.50 |
| On-Time Delivery Rate | % deliveries within window | 84% | 95% |
| Forecast Accuracy (MAPE) | Mean Absolute % Error of forecasts | N/A | <15% |

## Strategic Roadmap

| Phase | Weeks | Focus |
|---|---|---|
| 1. Data Collection | 1–2 | POS sales, WMS inventory, GPS/telematics delivery logs, master data |
| 2. Data Cleaning | 2–3 | Missing values, dedup, unit/date standardization, geocoding, stockout handling |
| 3. EDA & Segmentation | 3–4 | Time-series decomposition, ABC analysis, store clustering, route heatmaps |
| 4. Modeling | 4–6 | Demand forecasting (baseline vs XGBoost), safety stock, VRP route optimization |
| 5. Evaluation & Deployment | 6–8 | Rolling backtest, MAPE/fill rate/route savings, dashboard + recommendations |

## Repository Contents

| File | Purpose |
|---|---|
| `data_cleaning.py` | Load and clean raw sales transactions |
| `eda_abc_classification.py` | Weekly demand trends + ABC (Pareto) SKU classification |
| `store_clustering.py` | K-Means clustering of stores by demand behavior |
| `demand_forecasting.py` | XGBoost demand forecasting with lag features, time-series CV |
| `route_optimization.py` | OR-Tools vehicle routing problem (VRP) skeleton |

## Data Science Techniques Used

- **Regression / Gradient Boosting (XGBoost):** demand prediction
- **Time-series forecasting:** 4-week replenishment planning
- **K-Means clustering:** store segmentation
- **ABC/XYZ analysis:** inventory policy design
- **OR-Tools VRP optimization:** delivery route planning
- **EDA:** seasonality and anomaly detection

## Expected Outcomes

- Forecast MAPE below 15%, lifting fill rate from 89% to ~95%
- ABC/XYZ-driven inventory policies reducing holding costs 20–25%, raising turnover toward 8 turns/year
- 10–15% route distance savings, cutting cost/km from $1.85 to ~$1.50

Full write-up: see `Week1_Logistics_Strategic_Plan.docx` in this repo.
