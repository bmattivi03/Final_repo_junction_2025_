# This script:
# 1) Trains and tunes LightGBM models on HOURLY data (for 48-hour forecast).
# 2) Trains and tunes LightGBM models on DAILY data (for 12-month forecast).
# 3) Produces two CSVs in the required format:
#       forecast_48h.csv  (hourly, 48 rows)
#       forecast_12m.csv  (monthly totals, 12 rows)
# Assumptions:
#   - Hourly dataset:  full_hourly_dataset.csv
#   - Daily dataset:   dataset_daily_consumption.csv

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error
import optuna

# =========================================
# 1. HOURLY DATA: load and prepare
# =========================================

hourly_df = pd.read_csv('full_hourly_dataset.csv')

numeric_cols_hourly = [
    'group_id', 'hourly_consumption',
    'season', 'is_weekend', 'is_holiday',
    'hour', 'day_of_week', 'month',
    'ae_h_1', 'ae_h_2', 'ae_h_3', 'ae_h_4',
    'ae_h_5', 'ae_h_6', 'ae_h_7', 'ae_h_8'
]
for col in numeric_cols_hourly:
    if col in hourly_df.columns:
        hourly_df[col] = pd.to_numeric(hourly_df[col], errors='coerce')

hourly_df = hourly_df.dropna(subset=['group_id', 'hourly_consumption'])
hourly_df['group_id'] = hourly_df['group_id'].astype(int)

base_features_hourly = [
    'season', 'is_weekend', 'is_holiday',
    'hour', 'day_of_week', 'month',
    'ae_h_1', 'ae_h_2', 'ae_h_3', 'ae_h_4',
    'ae_h_5', 'ae_h_6', 'ae_h_7', 'ae_h_8'
]

# =========================================
# 2. HOURLY: Optuna tuning (single group)
# =========================================

hourly_counts = hourly_df['group_id'].value_counts()
hourly_gid = hourly_counts.index[0]

hourly_sub = hourly_df[hourly_df['group_id'] == hourly_gid].copy()
hourly_sub = hourly_sub.sort_values(['month', 'day_of_week', 'hour']).reset_index(drop=True)

split_idx_h = int(len(hourly_sub) * 0.8)
X_h = hourly_sub[base_features_hourly]
y_h = hourly_sub['hourly_consumption']

Xh_train, Xh_val = X_h.iloc[:split_idx_h], X_h.iloc[split_idx_h:]
yh_train, yh_val = y_h.iloc[:split_idx_h], y_h.iloc[split_idx_h:]

print("HOURLY: train size:", len(Xh_train))
print("HOURLY: val size:", len(Xh_val))

def objective_hourly(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 128),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'random_state': 42,
        'verbose': -1
    }
    model = LGBMRegressor(**params)
    model.fit(Xh_train, yh_train)
    pred = model.predict(Xh_val)
    mape = mean_absolute_percentage_error(yh_val, pred) * 100.0
    return mape

study_hourly = optuna.create_study(direction='minimize')
study_hourly.optimize(objective_hourly, n_trials=50)

best_params_hourly = study_hourly.best_params
best_params_hourly['random_state'] = 42
best_params_hourly['verbose'] = -1

print("HOURLY best MAPE:", study_hourly.best_value)
print("HOURLY best params:", best_params_hourly)

# =========================================
# 3. HOURLY: train per-group models
# =========================================

unique_groups_hourly = sorted(hourly_df['group_id'].unique())
models_by_group_hourly = {}

for gid in unique_groups_hourly:
    sub = hourly_df[hourly_df['group_id'] == gid].copy()
    sub = sub.sort_values(['month', 'day_of_week', 'hour']).reset_index(drop=True)

    if len(sub) < 100:
        continue

    split_idx_g = int(len(sub) * 0.8)
    Xg = sub[base_features_hourly]
    yg = sub['hourly_consumption']
    Xg_train, Xg_val = Xg.iloc[:split_idx_g], Xg.iloc[split_idx_g:]
    yg_train, yg_val = yg.iloc[:split_idx_g], yg.iloc[split_idx_g:]

    model_g = LGBMRegressor(**best_params_hourly)
    model_g.fit(Xg_train, yg_train)
    models_by_group_hourly[gid] = model_g

print("HOURLY: models trained for groups:", len(models_by_group_hourly))

# =========================================
# 4. HOURLY: build 48-hour forecast
# =========================================

start_ts_48 = datetime(2024, 10, 1, 0, 0)
timestamps_48 = [start_ts_48 + timedelta(hours=i) for i in range(48)]

ae_cols = ['ae_h_1', 'ae_h_2', 'ae_h_3', 'ae_h_4', 'ae_h_5', 'ae_h_6', 'ae_h_7', 'ae_h_8']
mean_ae_by_group = hourly_df.groupby('group_id')[ae_cols].mean()

def build_hourly_feature_row(dt, template_row):
    row = template_row.copy()
    row['month'] = dt.month
    row['day_of_week'] = dt.weekday()
    row['hour'] = dt.hour
    if dt.month in [12, 1, 2]:
        row['season'] = 0
    elif dt.month in [3, 4, 5]:
        row['season'] = 1
    elif dt.month in [6, 7, 8]:
        row['season'] = 2
    else:
        row['season'] = 3
    row['is_weekend'] = 1 if dt.weekday() >= 5 else 0
    row['is_holiday'] = 0
    return row

hourly_48_df = pd.DataFrame()
hourly_48_df['measured_at'] = [dt.strftime('%Y-%m-%dT%H:%M:%S.000Z') for dt in timestamps_48]

for gid in unique_groups_hourly:
    model = models_by_group_hourly.get(gid)
    col_name = str(int(gid))

    if model is None:
        hourly_48_df[col_name] = 0.0
        continue

    ae_vals = mean_ae_by_group.loc[gid]
    template = pd.Series(index=base_features_hourly, dtype=float)
    template[ae_cols] = ae_vals.values

    preds = []
    for dt in timestamps_48:
        feat_row = build_hourly_feature_row(dt, template)
        X_row = pd.DataFrame([feat_row[base_features_hourly]])
        pred_val = model.predict(X_row)[0]
        preds.append(pred_val)

    hourly_48_df[col_name] = preds

hourly_48_df.to_csv('forecast_48h_raw.csv', index=False, sep=';')

with open('forecast_48h_raw.csv', 'r', encoding='utf-8') as f:
    content_48 = f.read()

lines_48 = content_48.split('\n')
new_lines_48 = []
for i, line in enumerate(lines_48):
    if i == 0 or line.strip() == '':
        new_lines_48.append(line)
        continue
    parts = line.split(';')
    ts = parts[0]
    nums = parts[1:]
    nums = [n.replace('.', ',') for n in nums]
    new_line = ';'.join([ts] + nums)
    new_lines_48.append(new_line)

with open('forecast_48h.csv', 'w', encoding='utf-8') as f:
    f.write('\n'.join(new_lines_48))

print("48-hour forecast CSV created: forecast_48h.csv")

# =========================================
# 5. DAILY DATA: load and prepare
# =========================================

daily_df = pd.read_csv('dataset_daily_consumption.csv')

# Try to parse date, adapt to your schema if needed
if 'date' in daily_df.columns:
    daily_df['date'] = pd.to_datetime(daily_df['date'], errors='coerce')

else:
    raise ValueError('No date')

if 'daily_consumption' not in daily_df.columns:
    raise ValueError("Expected 'daily_consumption' column in daily dataset.")

numeric_daily_cols = ['group_id', 'daily_consumption']
for col in numeric_daily_cols:
    if col in daily_df.columns:
        daily_df[col] = pd.to_numeric(daily_df[col], errors='coerce')

daily_df = daily_df.dropna(subset=['group_id', 'daily_consumption', 'date'])
daily_df['group_id'] = daily_df['group_id'].astype(int)

if 'month' not in daily_df.columns:
    daily_df['month'] = daily_df['date'].dt.month
if 'day_of_week' not in daily_df.columns:
    daily_df['day_of_week'] = daily_df['date'].dt.weekday
if 'is_weekend' not in daily_df.columns:
    daily_df['is_weekend'] = daily_df['day_of_week'].isin([5, 6]).astype(int)
if 'season' not in daily_df.columns:
    def month_to_season(m):
        if m in [12, 1, 2]:
            return 0
        elif m in [3, 4, 5]:
            return 1
        elif m in [6, 7, 8]:
            return 2
        else:
            return 3
    daily_df['season'] = daily_df['month'].apply(month_to_season)
if 'is_holiday' not in daily_df.columns:
    daily_df['is_holiday'] = 0

base_features_daily = [
    'season', 'is_weekend', 'is_holiday',
    'day_of_week', 'month'
]

# =========================================
# 6. DAILY: Optuna tuning (single group)
# =========================================

daily_counts = daily_df['group_id'].value_counts()
daily_gid = daily_counts.index[0]

daily_sub = daily_df[daily_df['group_id'] == daily_gid].copy()
daily_sub = daily_sub.sort_values('date').reset_index(drop=True)

split_idx_d = int(len(daily_sub) * 0.8)
X_d = daily_sub[base_features_daily]
y_d = daily_sub['daily_consumption']

Xd_train, Xd_val = X_d.iloc[:split_idx_d], X_d.iloc[split_idx_d:]
yd_train, yd_val = y_d.iloc[:split_idx_d], y_d.iloc[split_idx_d:]

print("DAILY: train size:", len(Xd_train))
print("DAILY: val size:", len(Xd_val))

def objective_daily(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 128),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'random_state': 42,
        'verbose': -1
    }
    model = LGBMRegressor(**params)
    model.fit(Xd_train, yd_train)
    pred = model.predict(Xd_val)
    mape = mean_absolute_percentage_error(yd_val, pred) * 100.0
    return mape

study_daily = optuna.create_study(direction='minimize')
study_daily.optimize(objective_daily, n_trials=50)

best_params_daily = study_daily.best_params
best_params_daily['random_state'] = 42
best_params_daily['verbose'] = -1

print("DAILY best MAPE:", study_daily.best_value)
print("DAILY best params:", best_params_daily)

# =========================================
# 7. DAILY: train per-group models
# =========================================

unique_groups_daily = sorted(daily_df['group_id'].unique())
models_by_group_daily = {}

for gid in unique_groups_daily:
    sub = daily_df[daily_df['group_id'] == gid].copy()
    sub = sub.sort_values('date').reset_index(drop=True)

    if len(sub) < 60:
        continue

    split_idx_gd = int(len(sub) * 0.8)
    Xgd = sub[base_features_daily]
    ygd = sub['daily_consumption']
    Xgd_train, Xgd_val = Xgd.iloc[:split_idx_gd], Xgd.iloc[split_idx_gd:]
    ygd_train, ygd_val = ygd.iloc[:split_idx_gd], ygd.iloc[split_idx_gd:]

    model_d = LGBMRegressor(**best_params_daily)
    model_d.fit(Xgd_train, ygd_train)
    models_by_group_daily[gid] = model_d

print("DAILY: models trained for groups:", len(models_by_group_daily))

# =========================================
# 8. DAILY: build 12-month monthly forecast
# =========================================

month_starts = []
for m in range(10, 13):
    month_starts.append(datetime(2024, m, 1, 0, 0))
for m in range(1, 10):
    month_starts.append(datetime(2025, m, 1, 0, 0))

monthly_forecast_df = pd.DataFrame()
monthly_forecast_df['measured_at'] = [
    dt.strftime('%Y-%m-%dT%H:%M:%S.000Z') for dt in month_starts
]

for gid in unique_groups_daily:
    model_d = models_by_group_daily.get(gid)
    col_name = str(int(gid))

    if model_d is None:
        monthly_forecast_df[col_name] = 0.0
        continue

    monthly_totals = []
    for dt in month_starts:

        year = dt.year
        month = dt.month

        if month == 12:
            next_month = 1
            next_year = year + 1
        else:
            next_month = month + 1
            next_year = year
        mask = (
            (daily_df['group_id'] == gid) &
            (daily_df['date'] >= datetime(year, month, 1)) &
            (daily_df['date'] < datetime(next_year, next_month, 1))
        )
        days_in_month = daily_df.loc[mask, 'date']
        if len(days_in_month) == 0:
            days_in_month = pd.date_range(
                start=datetime(year, month, 1),
                end=datetime(next_year, next_month, 1) - pd.Timedelta(days=1),
                freq="D"
            )

        daily_preds = []
        for cur_date in days_in_month:
            
            month_val = cur_date.month
            dow_val = cur_date.weekday()
            is_weekend = int(dow_val >= 5)

            if month_val in [12, 1, 2]:
                season_val = 0
            elif month_val in [3, 4, 5]:
                season_val = 1
            elif month_val in [6, 7, 8]:
                season_val = 2
            else:
                season_val = 3

            feat = pd.DataFrame([{
                'season': season_val,
                'is_weekend': is_weekend,
                'is_holiday': 0,
                'day_of_week': dow_val,
                'month': month_val
            }])

            pred_day = model_d.predict(feat)[0]
            daily_preds.append(pred_day)

        monthly_totals.append(float(np.sum(daily_preds)))

    monthly_forecast_df[col_name] = monthly_totals

monthly_forecast_df.to_csv('forecast_12m_raw.csv', index=False, sep=';')

with open('forecast_12m_raw.csv', 'r', encoding='utf-8') as f:
    content_12m = f.read()

lines_12m = content_12m.split('\n')
new_lines_12m = []
for i, line in enumerate(lines_12m):
    if i == 0 or line.strip() == '':
        new_lines_12m.append(line)
        continue
    parts = line.split(';')
    ts = parts[0]
    nums = parts[1:]
    nums = [n.replace('.', ',') for n in nums]
    new_line = ';'.join([ts] + nums)
    new_lines_12m.append(new_line)

with open('forecast_12m.csv', 'w', encoding='utf-8') as f:
    f.write('\n'.join(new_lines_12m))

print("12-month forecast CSV created: forecast_12m.csv")