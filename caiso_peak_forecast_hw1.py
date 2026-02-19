"""
INDENG 290 HW 1 - Forecasting CAISO Peak Load
March 2026 Forecast
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv("CAISOHourlyLoadCSV(in).csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.rename(columns={'CAISO Load (MW)': 'load'})
df = df.sort_values(['Date', 'Hour']).reset_index(drop=True)

print("Data loaded from", df['Date'].min().date(), "to", df['Date'].max().date())

# Params
amp_factor = 1.038
growth_rate = 0.008
ensemble_weight = 0.5

# US Holidays
holidays = [
    '2024-11-11', '2024-11-28', '2024-12-25',
    '2025-01-01', '2025-01-20', '2025-02-17',
    '2025-05-26', '2025-06-19', '2025-07-04',
    '2025-09-01', '2025-10-13', '2025-11-11', '2025-11-27', '2025-12-25',
    '2026-01-01', '2026-01-19', '2026-02-16'
]

features = [
    'hour', 'dow', 'month', 'week_of_year',
    'is_weekend', 'is_holiday', 'year_trend',
    'hour_sin', 'hour_cos',
    'month_sin', 'month_cos',
    'dow_sin', 'dow_cos'
]

def create_features(d):
    d = d.copy()
    d['hour'] = d['Hour']
    d['dow'] = d['Date'].dt.dayofweek
    d['month'] = d['Date'].dt.month
    d['week_of_year'] = d['Date'].dt.isocalendar().week.astype(int)
    d['is_weekend'] = (d['dow'] >= 5).astype(int)
    d['is_holiday'] = d['Date'].dt.strftime('%Y-%m-%d').isin(holidays).astype(int)
    d['year_trend'] = d['Date'].dt.year - 2024
    
    # cyclical encoding for time features
    d['hour_sin'] = np.sin(2 * np.pi * d['hour'] / 24)
    d['hour_cos'] = np.cos(2 * np.pi * d['hour'] / 24)
    d['month_sin'] = np.sin(2 * np.pi * d['month'] / 12)
    d['month_cos'] = np.cos(2 * np.pi * d['month'] / 12)
    d['dow_sin'] = np.sin(2 * np.pi * d['dow'] / 7)
    d['dow_cos'] = np.cos(2 * np.pi * d['dow'] / 7)
    return d

df = create_features(df)

# --- Backtest: March 2025 ---
# train on Nov 24 - Feb 25 to predict Mar 25
train_bt = df[df['Date'] < '2025-03-01']
test_bt = df[(df['Date'].dt.year == 2025) & (df['Date'].dt.month == 3)]

rf = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf.fit(train_bt[features], train_bt['load'])
pred_bt = rf.predict(test_bt[features])

raw_peak = pred_bt.max()
adj_peak = raw_peak * amp_factor
actual_peak = test_bt['load'].max()

mae = mean_absolute_error(test_bt['load'].values, pred_bt)
peak_err = abs(adj_peak - actual_peak)

test_copy = test_bt.copy()
test_copy['pred_adj'] = pred_bt * amp_factor
pidx = test_copy['pred_adj'].idxmax()

print("\n--- Backtest (March 2025) ---")
print(f"Actual peak: {actual_peak} MW")
print(f"Raw RF prediction: {round(raw_peak)} MW")
print(f"Adjusted prediction (x{amp_factor}): {round(adj_peak)} MW")
print(f"Peak error: {round(peak_err)} MW ({round(peak_err/actual_peak*100, 1)}%)")
print(f"Hourly MAE: {round(mae)} MW")

# --- Final Forecast: March 2026 ---
# Method A: Historical Trend
method_a = actual_peak * (1 + growth_rate)

# Method B: RF on full data
rf_final = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf_final.fit(df[features], df['load'])

# generate March 2026 dummy data
dates = []
for d in range(1, 32):
    for h in range(1, 25):
        dates.append({'Date': pd.Timestamp(f'2026-03-{d:02d}'), 'Hour': h, 'load': np.nan})

mar26 = create_features(pd.DataFrame(dates))
mar26['pred_raw'] = rf_final.predict(mar26[features])
mar26['pred_adj'] = mar26['pred_raw'] * amp_factor

method_b_peak = mar26['pred_adj'].max()
pidx_26 = mar26['pred_adj'].idxmax()
peak_date = mar26.loc[pidx_26, 'Date'].strftime('%Y-%m-%d')
peak_hour = mar26.loc[pidx_26, 'Hour']

# Ensemble
final_forecast = (ensemble_weight * method_b_peak) + ((1 - ensemble_weight) * method_a)

print("\n--- Final Forecast (March 2026) ---")
print(f"Method A: {round(method_a)} MW")
print(f"Method B: {round(method_b_peak)} MW")
print(f"Ensemble Forecast: {round(final_forecast)} MW")
print(f"Predicted peak date: {peak_date} at Hour {peak_hour}")

# --- Task 3: Shape Factors ---
peak_day_df = mar26[mar26['Date'] == mar26.loc[pidx_26, 'Date']].sort_values('Hour').copy()
l_star = peak_day_df['pred_adj'].max()
peak_day_df['shape_factor'] = peak_day_df['pred_adj'] / l_star

print(f"\n--- Shape Factors for {peak_date} ---")

# create and save plot
plt.figure(figsize=(10, 4))
sns.barplot(x='Hour', y='shape_factor', data=peak_day_df, palette='viridis')
plt.ylim(0, 1.05)
plt.title(f"Shape Factors for {peak_date}")
plt.xlabel('Hour')
plt.ylabel('Shape Factor')

# plt.show() # commented out to avoid blocking execution 

out_file = f"shape_factors_{peak_date}.png"
plt.savefig(out_file, bbox_inches='tight')
print(f"Saved plot to {out_file}")

print("\nKey hours shape factors:")
for h in [2, 8, 9, 11, 13, 19]:
    val = peak_day_df[peak_day_df['Hour'] == h]['shape_factor'].values[0]
    print(f"Hour {h}: {round(val, 4)}")
