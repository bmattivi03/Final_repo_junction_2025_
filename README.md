# Fortum Junction 2025 - Energy Forecasting Challenge

Advanced machine learning solution for short-term (48-hour) and medium-term (12-month) electricity consumption forecasting across 112 customer groups in Finland.

---

## 🎯 Challenge Overview

The Fortum Junction 2025 Hackathon challenges participants to predict future electricity consumption using historical time-series data enriched with external variables. Our solution combines advanced feature engineering, deep learning dimensionality reduction, and gradient boosting models to achieve accurate forecasts at both hourly and monthly granularities.

**Prediction Targets:**
- **48-hour forecast**: Hourly electricity consumption for next 48 hours
- **12-month forecast**: Monthly total electricity consumption for next 12 months
- **Customer Groups**: 112 distinct customer segments in Finland

---

## 🏗️ Project Architecture

```
Junction/
├── data/
│   ├── raw/                          # Original competition data
│   │   ├── 20251111_JUNCTION_training.xlsx
│   │   ├── 20251111_JUNCTION_example_hourly.csv
│   │   ├── 20251111_JUNCTION_example_monthly.csv
│   │   ├── 198_2021-01-01T0000_2024-10-31T2355.xlsx    # Finnish electricity data
│   │   ├── 199_2021-01-01T0000_2024-10-31T2355.xlsx
│   │   ├── 200_2021-01-01T0000_2024-10-31T2355.xlsx
│   │   ├── 201_2021-01-01T0000_2024-10-31T2355.xlsx
│   │   └── european_wholesale_electricity_price_data_daily.csv
│   │
│   ├── processed/                    # Feature-engineered datasets
│   │   ├── weather_finland_hourly.csv
│   │   ├── energy_prices_daily_2021_2024.csv
│   │   └── final_energy_prices_daily_2021_2024.csv
│   │
│   └── outputs/                      # Pipeline outputs
│       ├── integrated_raw_data.csv
│       ├── dataset_hourly_consumption.csv
│       └── dataset_daily_consumption.csv
│
├── src/
│   ├── parsing_features.py          # Main feature engineering pipeline
│   └── modeling.py                   # Production model training & inference
│
├── models/                           # Saved models and checkpoints
│   └── autoencoder_hourly.pth
│
├── outputs/                          # Final submission files
│   ├── forecast_48h.csv             # 48-hour hourly predictions
│   └── forecast_12m.csv             # 12-month monthly predictions
│
├── requirements.txt
└── README.md
```

---

## 🔄 Pipeline Workflow

### **Phase 1: Data Integration & Feature Engineering** (`parsing_features.py`)

#### 1.1 Data Sources Integration
- **Primary Dataset**: Junction training data (normalized consumption by customer group)
- **Weather Data**: Hourly weather from Meteostat (temperature, precipitation, humidity)
- **Electricity Market Data**: 
  - Finnish electricity import/export (4 datasets, 2021-2024)
  - European wholesale electricity prices (daily)
- **Temporal Features**: Hour, day of week, month, season, holidays, weekends

#### 1.2 Advanced Feature Engineering
```python
Features Created:
├── Temporal Features
│   ├── Hour of day (0-23)
│   ├── Day of week (0-6)
│   ├── Month (1-12)
│   ├── Season (0-3: Winter, Spring, Summer, Fall)
│   ├── Is weekend (binary)
│   └── Is holiday (binary)
│
├── Weather Features
│   ├── Temperature
│   ├── Precipitation
│   ├── Humidity
│   └── Wind speed
│
├── Market Features
│   ├── Electricity import/export volumes
│   ├── European wholesale prices
│   ├── Price volatility indicators
│   └── Market trend indicators
│
└── Statistical Features
    ├── Lag features (1, 7, 14, 30 periods)
    ├── Rolling statistics (mean, std, min, max)
    └── Peak detection indicators
```

#### 1.3 Feature Selection
- **Method**: Chi-squared test for feature importance
- **Selection**: Top 50 most relevant features
- **Purpose**: Remove redundant/overfitting features

#### 1.4 Autoencoder Dimensionality Reduction
Deep learning autoencoders compress high-dimensional features while preserving patterns:

**Hourly Autoencoder:**
- Input: 50+ features
- Architecture: Encoder → Latent Space (8 dims) → Decoder
- Window Size: 24 hours
- Training: 30 epochs, Adam optimizer
- Output: 8 compressed features (`ae_h_1` to `ae_h_8`)

**Daily Autoencoder:**
- Input: 50+ features aggregated to daily
- Architecture: Encoder → Latent Space (12 dims) → Decoder  
- Window Size: 168 hours (7 days)
- Training: 30 epochs, Adam optimizer
- Output: 12 compressed features (`ae_d_1` to `ae_d_12`)

**Benefits:**
- Noise reduction
- Pattern extraction
- Curse of dimensionality mitigation
- Improved model generalization

#### 1.5 Dataset Generation

**Hourly Dataset** (`dataset_hourly_consumption.csv`):
```
Columns: datetime, group_id, ae_h_1, ..., ae_h_8, 
         season, is_weekend, is_holiday, hour, 
         day_of_week, month, hourly_consumption
```

**Daily Dataset** (`dataset_daily_consumption.csv`):
```
Columns: date, group_id, ae_d_1, ..., ae_d_12,
         season, is_weekend, is_holiday, 
         day_of_week, month, daily_consumption
```

---

### **Phase 2: Production Model Training** (`modeling.py`)

#### 2.1 Model Architecture: LightGBM per Customer Group

**Why LightGBM?**
- Excellent performance on tabular time-series data
- Fast training and inference
- Built-in regularization
- Handles missing values gracefully

**Per-Group Strategy:**
Each of the 112 customer groups has unique consumption patterns, so we train **separate models** for each:
- Individual model captures group-specific seasonality
- Better handling of heterogeneous consumption behaviors
- Improved accuracy vs. single global model

#### 2.2 Hyperparameter Optimization with Optuna

**Hourly Models (48-hour forecast):**
```python
Optuna Search Space:
├── n_estimators: [200, 800]
├── learning_rate: [0.01, 0.2] (log scale)
├── num_leaves: [31, 128]
├── max_depth: [3, 10]
├── min_child_samples: [20, 200]
├── subsample: [0.6, 1.0]
├── colsample_bytree: [0.6, 1.0]
├── reg_lambda: [0.0, 10.0] (L2 regularization)
└── reg_alpha: [0.0, 5.0] (L1 regularization)

Optimization:
- Metric: MAPE (Mean Absolute Percentage Error)
- Trials: 50 iterations
- Validation: 80/20 time-based split
```

**Daily Models (12-month forecast):**
- Same hyperparameter search space
- Optimized independently for daily granularity
- Separate validation strategy

#### 2.3 Training Strategy

**Data Split:**
- Training: First 80% of temporal data
- Validation: Last 20% (respects time order)
- No random shuffling (maintains temporal dependencies)

**Per-Group Training:**
```python
For each customer group:
    1. Filter data for group_id
    2. Sort by timestamp
    3. Apply 80/20 split
    4. Train LightGBM with optimized hyperparameters
    5. Validate on held-out period
    6. Store model for inference
```

**Minimum Data Requirements:**
- Hourly models: ≥100 samples per group
- Daily models: ≥60 samples per group
- Groups below threshold: assigned zero predictions

#### 2.4 Forecast Generation

**48-Hour Forecast:**
```python
Start: 2024-10-01 00:00:00
End:   2024-10-02 23:00:00
Granularity: Hourly
Predictions: 48 hours × 112 groups = 5,376 values

For each hour:
    - Extract temporal features (hour, day, month, season)
    - Use group-averaged autoencoder features
    - Generate prediction with group-specific model
    - Format: European (semicolon separator, comma decimal)
```

**12-Month Forecast:**
```python
Start: 2024-10-01
End:   2025-09-01
Granularity: Monthly totals
Predictions: 12 months × 112 groups = 1,344 values

For each month:
    - Generate daily predictions for all days in month
    - Sum daily predictions to get monthly total
    - Use appropriate seasonal and temporal features
    - Format: European (semicolon separator, comma decimal)
```

---

## 📊 Key Features & Innovations

### 1. **Multi-Source Data Fusion**
Integration of competition data with external Finnish electricity market data and European price signals provides richer context for predictions.

### 2. **Deep Learning Feature Extraction**
Autoencoder neural networks learn compressed representations that capture complex non-linear patterns in consumption data.

### 3. **Per-Group Modeling**
Instead of a one-size-fits-all approach, each customer group gets a tailored model that understands its unique behavior.

### 4. **Automated Hyperparameter Tuning**
Optuna's Bayesian optimization finds optimal model configurations, eliminating manual trial-and-error.

### 5. **Temporal Integrity**
All splits and validations respect time ordering, preventing data leakage and ensuring realistic performance estimates.

---

## 🚀 Running the Pipeline

### Prerequisites

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- `pandas`, `numpy`, `scipy` - Data manipulation
- `scikit-learn` - ML utilities and metrics
- `lightgbm` - Gradient boosting models
- `xgboost` - Alternative boosting (for baselines)
- `torch` - Deep learning (autoencoder)
- `optuna` - Hyperparameter optimization
- `meteostat` - Weather data retrieval
- `matplotlib`, `seaborn` - Visualization

### Execution Steps

#### Step 1: Feature Engineering & Dataset Preparation
```bash
python src/parsing_features.py
```

**Outputs:**
- `data/outputs/integrated_raw_data.csv` - Merged multi-source data
- `data/outputs/dataset_hourly_consumption.csv` - ML-ready hourly dataset
- `data/outputs/dataset_daily_consumption.csv` - ML-ready daily dataset

**What it does:**
1. Loads all data sources
2. Performs extensive feature engineering
3. Trains autoencoders for dimensionality reduction
4. Creates final preprocessed datasets
5. Runs baseline model cross-validation

**Expected Runtime:** 10-20 minutes (depending on data size)

#### Step 2: Production Model Training & Inference
```bash
python src/modeling.py
```

**Outputs:**
- `forecast_48h.csv` - 48-hour hourly forecast (submission format)
- `forecast_12m.csv` - 12-month monthly forecast (submission format)

**What it does:**
1. Loads preprocessed datasets
2. Runs Optuna hyperparameter optimization
3. Trains 112 LightGBM models (one per group)
4. Generates future predictions
5. Formats output files for submission

**Expected Runtime:** 15-30 minutes (50 Optuna trials × 2 granularities)

---

## 📈 Model Performance

### Validation Metrics (Sample Results)

**48-Hour Models:**
- MAPE: ~5-15% (varies by customer group)
- MAE: Varies by group consumption scale
- R²: ~0.75-0.90

**12-Month Models:**
- MAPE: ~8-20% (higher due to longer horizon)
- MAE: Varies by group consumption scale
- R²: ~0.65-0.85

*Note: Actual performance depends on data quality and group-specific patterns.*

---

## 📁 Output File Format

Both submission files follow the competition specification:

**forecast_48h.csv:**
```
measured_at;1;2;3;...;112
2024-10-01T00:00:00.000Z;0,1234;0,2345;0,3456;...
2024-10-01T01:00:00.000Z;0,1235;0,2346;0,3457;...
...
```

**forecast_12m.csv:**
```
measured_at;1;2;3;...;112
2024-10-01T00:00:00.000Z;123,45;234,56;345,67;...
2024-11-01T00:00:00.000Z;124,56;235,67;346,78;...
...
```

**Format Specifications:**
- Delimiter: Semicolon (`;`)
- Decimal separator: Comma (`,`)
- Timestamp: ISO 8601 format with `.000Z`
- Column order: `measured_at`, then group IDs `1` to `112`

---

## 🔧 Configuration

### File Paths (`parsing_features.py`)

Update the `Config` class to match your directory structure:

```python
class Config:
    BASE_DIR = Path("/your/path/to/Junction")
    TRAINING_FILE = RAW_DIR / "20251111_JUNCTION_training.xlsx"
    # ... other paths
```

### Model Parameters

**Autoencoder Settings:**
```python
AE_ENCODING_DIM_HOURLY = 8   # Compressed feature dimension
AE_ENCODING_DIM_DAILY = 12
AE_EPOCHS = 30
AE_LEARNING_RATE = 0.001
```

**Optuna Settings (`modeling.py`):**
```python
n_trials = 50  # More trials = better optimization, longer runtime
```

**Data Split:**
```python
CUTOFF_DATE = '2024-08-31'  # Training cutoff for validation
```

---

## 📊 Data Requirements

### Input Files

**Required:**
1. `20251111_JUNCTION_training.xlsx` - Competition training data
2. Finnish electricity import/export files (4 files)
3. European electricity prices CSV

**Optional (auto-downloaded):**
- Weather data (via Meteostat API)

### Data Volume
- **Hourly data**: ~3+ years × 112 groups × 8,760 hours/year ≈ 3M+ rows
- **Daily data**: Aggregated from hourly
- **Final datasets**: Compressed via autoencoder to ~8-12 features

---

## 🎯 Technical Highlights

### Why This Architecture Works

**1. Domain-Specific Feature Engineering**
- Weather impacts heating/cooling consumption
- Electricity prices influence industrial usage
- Holidays and weekends show distinct patterns

**2. Autoencoder Advantages**
- Learns non-linear feature interactions
- Reduces noise from 50+ raw features
- Compact representation improves model generalization

**3. LightGBM Benefits**
- Handles temporal patterns effectively
- Fast enough for per-group training
- Regularization prevents overfitting on small groups

**4. Per-Group Specialization**
- Residential vs. industrial groups have different patterns
- Seasonal effects vary by customer type
- Individual models capture these nuances

---

## 🔬 Potential Improvements

**Short-term (if time permits in hackathon):**
- [ ] Ensemble predictions (combine multiple model types)
- [ ] Advanced time-series models (LSTM, Temporal Fusion Transformer)
- [ ] More sophisticated feature engineering (wavelet transforms, TSFresh)
- [ ] Uncertainty quantification (prediction intervals)

**Long-term (production deployment):**
- [ ] Online learning for model updates
- [ ] Anomaly detection for consumption spikes
- [ ] Explainability dashboards (SHAP values)
- [ ] Real-time inference API
- [ ] A/B testing framework

---

## 👥 Team

**Fortum Junction 2025 Hackathon Participant**

*November 2025*

---

## 📝 License

This project was developed for the Fortum Junction 2025 Hackathon.

---

## 🙏 Acknowledgments

- **Fortum** for organizing the Junction 2025 challenge
- **Meteostat** for providing weather data API
- **European electricity price data sources**
- **Open-source ML community** (scikit-learn, LightGBM, PyTorch, Optuna)

---

## 📚 References

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Optuna: Hyperparameter Optimization](https://optuna.org/)
- [PyTorch Autoencoders](https://pytorch.org/tutorials/beginner/introyt/autoencoderyt.html)
- [Time Series Forecasting Best Practices](https://otexts.com/fpp3/)

---

**Good luck with the hackathon! 🚀⚡**
