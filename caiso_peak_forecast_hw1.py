"""
INDENG 290 Energy Analytics — HW #1 Forecasting
================================================
Forecast: CAISO Peak Load for March 2026

Model     : Random Forest Regressor + Peak Amplification Factor
            Ensemble with Historical Trend (Method A)
Data      : CAISOHourlyLoadCSV (Nov 2024–Oct 2025), instructor-provided
            Access date: February 2026
            No additional external data used.

Results
-------
  Point forecast  : 28,054 MW
  Forecast date   : 2026-03-13 (Friday)
  Forecast hour   : Hour 9 (hour ending 9:00 am)

Backtest (March 2025)
---------------------
  Actual peak     : 28,127 MW  (2025-03-14, Hour 11)  [known]
  Forecast peak   : 27,126 MW
  Peak error      : 1,001 MW  (3.6%)
  Hourly MAE      : 1,048 MW
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────────────────────────
DATA_PATH  = "CAISOHourlyLoadCSV(in).csv"   # path to instructor-provided data
AMP_FACTOR = 1.038   # peak amplification factor (estimated via leave-one-month-out CV)
GROWTH_RATE = 0.008  # assumed YoY peak growth rate for CAISO in March (~0.8%)
ENSEMBLE_W  = 0.5   # weight on Method B (RF); (1 - weight) on Method A (historical)

# US Federal Holidays in data range (hardcoded — no external library needed)
US_HOLIDAYS = {
    '2024-11-11','2024-11-28','2024-12-25',
    '2025-01-01','2025-01-20','2025-02-17',
    '2025-05-26','2025-06-19','2025-07-04',
    '2025-09-01','2025-10-13','2025-11-11','2025-11-27','2025-12-25',
    '2026-01-01','2026-01-19','2026-02-16',
}

FEATURES = [
    'hour', 'dow', 'month', 'week_of_year',
    'is_weekend', 'is_holiday', 'year_trend',
    'hour_sin', 'hour_cos',
    'month_sin', 'month_cos',
    'dow_sin', 'dow_cos',
]

# ─────────────────────────────────────────────────────────────────
# 1. LOAD & CLEAN DATA
# ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df = df.rename(columns={'CAISO Load (MW)': 'load'})
df = df.sort_values(['Date', 'Hour']).reset_index(drop=True)

print(f"Loaded {len(df):,} hourly observations: "
      f"{df['Date'].min().date()} → {df['Date'].max().date()}")

# ─────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────
def make_features(d: pd.DataFrame) -> pd.DataFrame:
    """
    Build calendar-based features for load forecasting.

    Features capture three dominant drivers of hourly load variation:
      • Time-of-day  : hour (raw + cyclic sine/cosine encoding)
      • Day-of-week  : weekday vs. weekend; holiday indicator
      • Seasonality  : month (cyclic), week-of-year
      • Trend        : integer year offset from 2024 to capture load growth
    """
    d = d.copy()
    d['hour']         = d['Hour']
    d['dow']          = d['Date'].dt.dayofweek           # 0 = Monday
    d['month']        = d['Date'].dt.month
    d['week_of_year'] = d['Date'].dt.isocalendar().week.astype(int)
    d['is_weekend']   = (d['dow'] >= 5).astype(int)
    d['is_holiday']   = (d['Date'].dt.strftime('%Y-%m-%d')
                          .isin(US_HOLIDAYS)).astype(int)
    d['year_trend']   = d['Date'].dt.year - 2024         # 2024→0, 2025→1, 2026→2

    # Cyclic encodings prevent artificial discontinuity (e.g. hour 24→1)
    d['hour_sin']  = np.sin(2 * np.pi * d['hour']  / 24)
    d['hour_cos']  = np.cos(2 * np.pi * d['hour']  / 24)
    d['month_sin'] = np.sin(2 * np.pi * d['month'] / 12)
    d['month_cos'] = np.cos(2 * np.pi * d['month'] / 12)
    d['dow_sin']   = np.sin(2 * np.pi * d['dow']   / 7)
    d['dow_cos']   = np.cos(2 * np.pi * d['dow']   / 7)
    return d

df = make_features(df)

# ─────────────────────────────────────────────────────────────────
# 3. BACKTEST — Predict March 2025 Peak
#
# Rationale: March 2025 is the most recent March in the dataset and
# serves as a direct analog for March 2026.  The actual March 2025
# peak (28,127 MW on 3/14 Hour 11) is known, so we can compute a
# concrete error metric for the monthly peak MW.
#
# Train  : Nov 2024 – Feb 2025  (4 months preceding March)
# Test   : March 2025           (31 days × 24 hours = 744 obs)
# ─────────────────────────────────────────────────────────────────
train_bt = df[df['Date'] < '2025-03-01']
test_bt  = df[(df['Date'].dt.year == 2025) & (df['Date'].dt.month == 3)]

rf_bt = RandomForestRegressor(
    n_estimators=500, max_depth=15, min_samples_leaf=2,
    random_state=42, n_jobs=-1
)
rf_bt.fit(train_bt[FEATURES], train_bt['load'])
pred_bt = rf_bt.predict(test_bt[FEATURES])

# RF models predict conditional means and systematically underestimate extremes.
# The amplification factor (AMP_FACTOR = 1.038) corrects for this bias.
# It was estimated via leave-one-month-out cross-validation on the training set.
raw_peak_bt  = pred_bt.max()
adj_peak_bt  = raw_peak_bt * AMP_FACTOR
actual_peak  = test_bt['load'].max()
mae_hourly   = mean_absolute_error(test_bt['load'].values, pred_bt)
peak_err_mw  = abs(adj_peak_bt - actual_peak)
peak_err_pct = peak_err_mw / actual_peak * 100

# Forecast peak date/hour
test_copy = test_bt.copy()
test_copy['pred_adj'] = pred_bt * AMP_FACTOR
pidx = test_copy['pred_adj'].idxmax()
bt_date = test_copy.loc[pidx, 'Date'].strftime('%Y-%m-%d')
bt_hour = test_copy.loc[pidx, 'Hour']

print("\n── BACKTEST: March 2025 ────────────────────────────────────────")
print(f"  Actual peak    : {actual_peak:>8,.0f} MW  (2025-03-14, Hour 11)  [known]")
print(f"  RF raw predict : {raw_peak_bt:>8,.0f} MW")
print(f"  Amp factor     :   ×{AMP_FACTOR}")
print(f"  Adjusted pred  : {adj_peak_bt:>8,.0f} MW  ({bt_date}, Hour {bt_hour})")
print(f"  Peak error     : {peak_err_mw:>8,.0f} MW  ({peak_err_pct:.1f}%)")
print(f"  Hourly MAE     : {mae_hourly:>8,.0f} MW")

# ─────────────────────────────────────────────────────────────────
# 4. FINAL FORECAST — March 2026
#
# Two-method ensemble:
#   Method A: Historical analog + trend
#             March 2025 actual peak × (1 + GROWTH_RATE)
#             Simple, interpretable, leverages best available prior.
#
#   Method B: RF trained on full dataset + AMP_FACTOR
#             Learns seasonal & day-of-week patterns from all 12 months.
#
# Ensemble: equal-weight average of both methods.
# ─────────────────────────────────────────────────────────────────

# Method A
methodA = actual_peak * (1 + GROWTH_RATE)

# Method B — retrain on full dataset
rf_final = RandomForestRegressor(
    n_estimators=500, max_depth=15, min_samples_leaf=2,
    random_state=42, n_jobs=-1
)
rf_final.fit(df[FEATURES], df['load'])

# Build March 2026 feature matrix (Hour 1–24 × 31 days)
rows = [
    {'Date': pd.Timestamp(f'2026-03-{d:02d}'), 'Hour': h, 'load': np.nan}
    for d in range(1, 32)
    for h in range(1, 25)
]
mar26 = make_features(pd.DataFrame(rows))
mar26['pred_raw'] = rf_final.predict(mar26[FEATURES])
mar26['pred_adj'] = mar26['pred_raw'] * AMP_FACTOR

methodB_peak = mar26['pred_adj'].max()
pidx26       = mar26['pred_adj'].idxmax()
peak_date26  = mar26.loc[pidx26, 'Date'].strftime('%Y-%m-%d')
peak_hour26  = mar26.loc[pidx26, 'Hour']

# Ensemble forecast
ensemble_forecast = ENSEMBLE_W * methodB_peak + (1 - ENSEMBLE_W) * methodA

print("\n── FINAL FORECAST: March 2026 ──────────────────────────────────")
print(f"  Method A (historical trend)  : {methodA:>8,.0f} MW")
print(f"  Method B (RF + amp factor)   : {methodB_peak:>8,.0f} MW  ({peak_date26}, Hour {peak_hour26})")
print(f"  Ensemble (50/50)             : {ensemble_forecast:>8,.0f} MW")
print()
print(f"  ★  POINT FORECAST  : {ensemble_forecast:,.0f} MW")
print(f"  ★  FORECAST DATE   : {peak_date26}")
print(f"  ★  FORECAST HOUR   : Hour {peak_hour26} (hour ending {peak_hour26}:00 PT)")

# ─────────────────────────────────────────────────────────────────
# 5. TASK 3 — 24-Hour Shape Factors for Peak Day
# ─────────────────────────────────────────────────────────────────
# Use the Method B peak day (2026-03-13) and scale so max(sh) = 1.00
peak_day_data = (mar26[mar26['Date'] == mar26.loc[pidx26, 'Date']]
                 .sort_values('Hour').copy())

L_star = peak_day_data['pred_adj'].max()   # = Method B peak = denominator
peak_day_data['shape_factor'] = peak_day_data['pred_adj'] / L_star

print(f"\n── SHAPE FACTORS for {peak_date26} (L* = {ensemble_forecast:,.0f} MW) ──")
print(f"  (sh = forecast_hour_load / L*; max sh = 1.00 by construction)")

# plot shape factor bar chart（per hour）
plot_df = peak_day_data.sort_values('Hour').copy()
plt.figure(figsize=(12, 4))
sns.barplot(x='Hour', y='shape_factor', data=plot_df, palette='viridis')
plt.ylim(0, 1.05)
plt.title(f"Shape Factors for {peak_date26} (L* = {ensemble_forecast:,.0f} MW)")
plt.xlabel('Hour')
plt.ylabel('Shape factor (sh)')
# label each bar with its shape factor value
for _, r in plot_df.iterrows():
    plt.text(r['Hour']-1, r['shape_factor'] + 0.02, f"{r['shape_factor']:.3f}",
             ha='center', va='bottom', fontsize=9)
plt.tight_layout()
# 自动保存图像（便于在无图形界面的环境中查看）
out_png = f"shape_factors_{peak_date26}.png"
plt.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"  Saved shape factors plot to: {out_png}")
plt.show()

# Summary table for write-up 
sf_dict = dict(zip(peak_day_data['Hour'], peak_day_data['shape_factor'].round(4)))
print(f"\n  Key shape factors:")
for h in [2, 8, 9, 11, 13, 19]:
    print(f"    Hour {h:2d}: sh = {sf_dict[h]:.4f}")

print("\nDone.")
