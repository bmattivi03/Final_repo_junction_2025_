#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FORTUM JUNCTION 2025 - ENERGY FORECASTING PIPELINE
================================================================================
Complete ML pipeline for hourly and monthly electricity consumption prediction
across 112 customer groups in Finland.

This pipeline includes:
1. Data loading and preprocessing from multiple sources
2. External feature engineering (weather, electricity prices, market data)
3. Advanced feature creation (temporal, seasonal, market indicators)
4. Feature selection using statistical methods
5. Model training with cross-validation
6. Autoencoder-based dimensionality reduction
7. Final dataset preparation for production models

Author: Team [Your Team Name]
Date: November 2025
Competition: Fortum Junction 2025 Hackathon
================================================================================
"""

# ============================================================================
# SECTION 1: IMPORTS AND DEPENDENCIES
# ============================================================================

# Standard library imports
import os
import sys
import time
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union

# Data manipulation and numerical computing
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks

# Machine Learning - Scikit-learn
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import chi2, SelectKBest, mutual_info_regression
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    mean_absolute_percentage_error,
    r2_score
)
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)

# Advanced ML Models
try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
except ImportError:
    print("Warning: LightGBM not installed. Installing...")
    os.system(f"{sys.executable} -m pip install lightgbm")
    import lightgbm as lgb
    from lightgbm import LGBMRegressor

try:
    import xgboost as xgb
    from xgboost import XGBRegressor
except ImportError:
    print("Warning: XGBoost not installed. Installing...")
    os.system(f"{sys.executable} -m pip install xgboost")
    import xgboost as xgb
    from xgboost import XGBRegressor

# Deep Learning - PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("Warning: PyTorch not installed. Installing...")
    os.system(f"{sys.executable} -m pip install torch")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader

# Weather data
try:
    from meteostat import Hourly, Point
except ImportError:
    print("Warning: Meteostat not installed. Installing...")
    os.system(f"{sys.executable} -m pip install meteostat")
    from meteostat import Hourly, Point

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 2: CONFIGURATION AND FILE PATHS
# ============================================================================

print("=" * 80)
print("FORTUM JUNCTION 2025 - ENERGY FORECASTING PIPELINE")
print("=" * 80)
print()

# -----------------------------------------------------------------------------
# USER-CONFIGURABLE PATHS - CHANGE THESE TO MATCH YOUR DIRECTORY STRUCTURE
# -----------------------------------------------------------------------------

class Config:
    """
    Central configuration class for all file paths and parameters.
    Modify these paths to match your local directory structure.
    """
    
    # Base directories
    BASE_DIR = Path("/Users/brandomattivi/Desktop/hackathon/Junction")  # CHANGE THIS
    DATA_DIR = BASE_DIR / "data"
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    MODEL_DIR = BASE_DIR / "models"
    OUTPUT_DIR = BASE_DIR / "outputs"
    
    # Main hackathon dataset
    TRAINING_FILE = RAW_DIR / "20251111_JUNCTION_training.xlsx"
    
    # External data sources - Finnish electricity import/export
    ELECTRICITY_FILES = [
        RAW_DIR / "198_2021-01-01T0000_2024-10-31T2355.xlsx",
        RAW_DIR / "199_2021-01-01T0000_2024-10-31T2355.xlsx",
        RAW_DIR / "200_2021-01-01T0000_2024-10-31T2355.xlsx",
        RAW_DIR / "201_2021-01-01T0000_2024-10-31T2355.xlsx"
    ]
    
    # European electricity prices
    EU_PRICES_RAW = DATA_DIR / "european_wholesale_electricity_price_data_daily.csv"
    EU_PRICES_PROCESSED = PROCESSED_DIR / "energy_prices_daily_2021_2024.csv"
    EU_PRICES_FINAL = PROCESSED_DIR / "final_energy_prices_daily_2021_2024.csv"
    
    # Weather data
    WEATHER_HOURLY = PROCESSED_DIR / "weather_finland_hourly.csv"
    
    # Model parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_SPLITS = 5
    
    # Training cutoff date (for time series validation)
    CUTOFF_DATE = '2024-08-31'
    
    # Autoencoder parameters
    AE_ENCODING_DIM_HOURLY = 8
    AE_ENCODING_DIM_DAILY = 12
    AE_WINDOW_HOURLY = 24
    AE_WINDOW_DAILY = 168
    AE_EPOCHS = 30
    AE_LEARNING_RATE = 0.001
    
    # Feature engineering parameters
    ROLLING_WINDOWS = [7, 14, 30]
    LAG_PERIODS = [1, 7, 14, 30]
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        for directory in [cls.PROCESSED_DIR, cls.MODEL_DIR, cls.OUTPUT_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
            
    @classmethod
    def validate_paths(cls):
        """Validate that required input files exist"""
        missing_files = []
        
        if not cls.TRAINING_FILE.exists():
            missing_files.append(str(cls.TRAINING_FILE))
            
        for file in cls.ELECTRICITY_FILES:
            if not file.exists():
                missing_files.append(str(file))
                
        if missing_files:
            print("⚠️  Warning: The following files are missing:")
            for file in missing_files:
                print(f"   - {file}")
            print("\nPlease update the paths in the Config class or ensure files exist.")
            
        return len(missing_files) == 0

# Initialize configuration
Config.create_directories()
config_valid = Config.validate_paths()

if not config_valid:
    print("\n❌ Critical files missing. Please fix the paths and rerun.")
    # Continue anyway for demonstration purposes

# ============================================================================
# SECTION 3: UTILITY FUNCTIONS
# ============================================================================

def print_section_header(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)

def print_subsection_header(title: str):
    """Print a formatted subsection header"""
    print("\n" + "-" * 60)
    print(f"{title}")
    print("-" * 60)

def safe_read_excel(filepath: Path, sheet_name: str = None) -> pd.DataFrame:
    """
    Safely read Excel file with error handling
    
    Args:
        filepath: Path to Excel file
        sheet_name: Name of sheet to read (optional)
        
    Returns:
        DataFrame or empty DataFrame if error
    """
    try:
        if sheet_name:
            return pd.read_excel(filepath, sheet_name=sheet_name)
        else:
            return pd.read_excel(filepath)
    except FileNotFoundError:
        print(f"⚠️  File not found: {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"⚠️  Error reading {filepath}: {e}")
        return pd.DataFrame()

def safe_read_csv(filepath: Path) -> pd.DataFrame:
    """
    Safely read CSV file with error handling
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame or empty DataFrame if error
    """
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"⚠️  File not found: {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"⚠️  Error reading {filepath}: {e}")
        return pd.DataFrame()

# ============================================================================
# SECTION 4: DATA LOADING AND PREPROCESSING
# ============================================================================

class DataLoader:
    """
    Main class for loading and preprocessing the hackathon dataset
    """
    
    @staticmethod
    def load_training_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load the main training dataset from the hackathon organizers
        
        Returns:
            Tuple of (consumption_wide, groups, prices) DataFrames
        """
        print_section_header("LOADING HACKATHON TRAINING DATA")
        
        consumption_wide = safe_read_excel(
            Config.TRAINING_FILE, 
            sheet_name="training_consumption"
        )
        groups = safe_read_excel(
            Config.TRAINING_FILE, 
            sheet_name="groups"
        )
        prices = safe_read_excel(
            Config.TRAINING_FILE, 
            sheet_name="training_prices"
        )
        
        print(f"✓ Consumption data shape: {consumption_wide.shape}")
        print(f"✓ Groups data shape: {groups.shape}")
        print(f"✓ Prices data shape: {prices.shape}")
        
        return consumption_wide, groups, prices
    
    @staticmethod
    def reshape_hourly_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
        """
        Convert wide format hourly consumption data to long format
        
        Args:
            df_wide: Wide format DataFrame with columns as group IDs
            
        Returns:
            Long format DataFrame with group_id as a column
        """
        print("\n📊 Reshaping consumption data from wide to long format...")
        
        # Fix comma decimal separators (European format)
        for col in df_wide.columns:
            if col == "measured_at":
                continue
            df_wide[col] = (
                df_wide[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .astype(float)
            )
        
        # Convert datetime
        df_wide["measured_at"] = pd.to_datetime(df_wide["measured_at"])
        
        # Melt to long format
        df_long = df_wide.melt(
            id_vars=["measured_at"],
            var_name="group_id",
            value_name="consumption"
        )
        
        df_long["group_id"] = df_long["group_id"].astype(int)
        
        print(f"✓ Reshaped to long format: {df_long.shape}")
        
        return df_long
    
    @staticmethod
    def parse_group_metadata(df_groups: pd.DataFrame) -> pd.DataFrame:
        """
        Parse the group label into separate metadata columns
        
        Group label format: "Region|Province|City|CustomerType|PricingType|PricingLevel"
        
        Args:
            df_groups: DataFrame with group_id and group_label columns
            
        Returns:
            DataFrame with parsed metadata columns
        """
        print("\n🏷️  Parsing group metadata...")
        
        # Split the pipe-delimited label
        parts = df_groups["group_label"].str.split("|", expand=True)
        
        df_groups["region"] = parts[0].str.strip()
        df_groups["province"] = parts[1].str.strip()
        df_groups["city"] = parts[2].str.strip()
        df_groups["customer_type"] = parts[3].str.strip()
        df_groups["pricing_type"] = parts[4].str.strip()
        df_groups["pricing_level"] = parts[5].str.strip()
        
        print(f"✓ Parsed {len(df_groups)} group metadata records")
        print(f"  Regions: {df_groups['region'].nunique()}")
        print(f"  Customer types: {df_groups['customer_type'].nunique()}")
        print(f"  Pricing types: {df_groups['pricing_type'].nunique()}")
        
        return df_groups

# ============================================================================
# SECTION 5: EXTERNAL DATA - ELECTRICITY IMPORT/EXPORT
# ============================================================================

class ElectricityMarketData:
    """
    Process Finnish electricity import/export and production data
    """
    
    NUMERIC_COLUMNS = [
        "Electricity production, surplus/deficit - real-time data",
        "Net import/export of electricity - real-time data"
    ]
    
    @classmethod
    def load_and_aggregate(cls) -> pd.DataFrame:
        """
        Load and aggregate electricity market data from multiple files
        
        Returns:
            Aggregated daily electricity market DataFrame
        """
        print_section_header("PROCESSING ELECTRICITY IMPORT/EXPORT DATA")
        
        all_data = []
        
        for filepath in Config.ELECTRICITY_FILES:
            if not filepath.exists():
                print(f"⚠️  Skipping missing file: {filepath}")
                continue
                
            print(f"📁 Loading {filepath.name}...")
            
            df = safe_read_excel(filepath)
            if df.empty:
                continue
                
            # Parse datetime and set index
            df["startTime"] = pd.to_datetime(df["startTime"], errors="coerce")
            df = df.set_index("startTime").sort_index()
            
            # Convert to numeric
            df[cls.NUMERIC_COLUMNS] = df[cls.NUMERIC_COLUMNS].apply(
                pd.to_numeric, errors="coerce"
            )
            
            # Resample to daily
            df_daily = df[cls.NUMERIC_COLUMNS].resample("D").sum()
            all_data.append(df_daily)
        
        if not all_data:
            print("⚠️  No electricity data loaded")
            return pd.DataFrame()
        
        # Aggregate all files
        aggregated = pd.concat(all_data).groupby(level=0).sum()
        
        # Rename columns for clarity
        aggregated.columns = ['elec_production_deficit', 'elec_net_import_export']
        
        print(f"✓ Aggregated electricity data shape: {aggregated.shape}")
        print(f"✓ Date range: {aggregated.index.min()} to {aggregated.index.max()}")
        
        # Save processed data
        output_path = Config.PROCESSED_DIR / "electricity_market_daily.csv"
        aggregated.to_csv(output_path)
        print(f"💾 Saved to: {output_path}")
        
        return aggregated

# ============================================================================
# SECTION 6: EXTERNAL DATA - WEATHER
# ============================================================================

class WeatherDataProcessor:
    """
    Fetch and process weather data for Finnish cities using Meteostat API
    """
    
    # Major Finnish cities/regions with coordinates
    FINLAND_LOCATIONS = {
        "Etelä-Savo": (61.6945, 27.2723),
        "Joensuu": (62.6010, 29.7632),
        "Pohjois-Karjala": (62.8924, 30.1306),
        "Pohjois-Savo": (63.0, 27.0),
        "Lappi": (66.5, 25.7),
        "Rovaniemi": (66.5039, 25.7294),
        "Oulu": (65.0121, 25.4651),
        "Pohjois-Pohjanmaa": (65.6, 26.0),
        "Lappeenranta": (61.0583, 28.1863),
        "Kanta-Häme": (60.9, 24.3),
        "Lahti": (60.9827, 25.6615),
        "Päijät-Häme": (61.0, 25.5),
        "Espoo": (60.2055, 24.6559),
        "Uusimaa": (60.25, 25.0),
        "Vantaa": (60.2941, 25.0400),
        "Pori": (61.4850, 21.7970),
        "Varsinais-Suomi": (60.4545, 22.2648),
        "Etelä-Pohjanmaa": (62.8, 23.0),
        "Jyväskylä": (62.2415, 25.7209),
        "Keski-Suomi": (62.4, 25.7),
        "Pirkanmaa": (61.5, 23.8),
        "Tampere": (61.4978, 23.7610),
        "Pohjanmaa": (63.0833, 21.6167)
    }
    
    @staticmethod
    def fetch_weather_chunked(
        lat: float, 
        lon: float, 
        start_date: str, 
        end_date: str, 
        chunk_days: int = 7
    ) -> pd.DataFrame:
        """
        Download weather data in chunks to avoid API limits
        
        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            chunk_days: Number of days per API request
            
        Returns:
            DataFrame with hourly weather data
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        all_data = []
        current = start
        location = Point(lat, lon)
        
        while current <= end:
            chunk_start = current
            chunk_end = min(current + timedelta(days=chunk_days), end)
            
            try:
                data = Hourly(location, chunk_start, chunk_end)
                df = data.fetch().reset_index()
                df.rename(columns={"time": "timestamp"}, inplace=True)
                all_data.append(df)
                
                print(f"  ✓ Fetched {chunk_start.date()} → {chunk_end.date()}")
                
            except Exception as e:
                print(f"  ⚠️ Failed: {chunk_start} to {chunk_end}: {e}")
            
            current = chunk_end + timedelta(days=1)
            time.sleep(0.3)  # Rate limiting
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
    
    @classmethod
    def fetch_all_finland_weather(
        cls, 
        start_date: str = "2021-01-01",
        end_date: str = "2024-09-30"
    ) -> pd.DataFrame:
        """
        Fetch weather data for all major Finnish cities
        
        Args:
            start_date: Start date for weather data
            end_date: End date for weather data
            
        Returns:
            Combined DataFrame with weather data for all locations
        """
        print_section_header("FETCHING WEATHER DATA FOR FINLAND")
        
        # Check if processed file already exists
        if Config.WEATHER_HOURLY.exists():
            print("📁 Loading existing weather data...")
            df_weather = safe_read_csv(Config.WEATHER_HOURLY)
            if not df_weather.empty:
                print(f"✓ Loaded weather data: {df_weather.shape}")
                return df_weather
        
        print("🌤️  Fetching new weather data from Meteostat API...")
        print(f"  Period: {start_date} to {end_date}")
        print(f"  Locations: {len(cls.FINLAND_LOCATIONS)}")
        
        all_dfs = []
        
        for name, (lat, lon) in cls.FINLAND_LOCATIONS.items():
            print(f"\n📍 {name} ({lat:.4f}, {lon:.4f})")
            
            df = cls.fetch_weather_chunked(
                lat, lon, start_date, end_date
            )
            
            if not df.empty:
                df["region"] = name
                all_dfs.append(df)
        
        if not all_dfs:
            print("⚠️  No weather data fetched")
            return pd.DataFrame()
        
        # Combine all locations
        weather_finland = pd.concat(all_dfs, ignore_index=True)
        
        print(f"\n✓ Combined weather data shape: {weather_finland.shape}")
        
        # Save to file
        weather_finland.to_csv(Config.WEATHER_HOURLY, index=False)
        print(f"💾 Saved to: {Config.WEATHER_HOURLY}")
        
        return weather_finland

# ============================================================================
# SECTION 7: EXTERNAL DATA - EUROPEAN ELECTRICITY PRICES
# ============================================================================

class EuropeanElectricityPrices:
    """
    Process European wholesale electricity price data
    """
    
    @staticmethod
    def process_eu_prices() -> pd.DataFrame:
        """
        Process and filter European electricity price data
        
        Returns:
            DataFrame with daily electricity prices for relevant countries
        """
        print_section_header("PROCESSING EUROPEAN ELECTRICITY PRICES")
        
        # Check if final processed file exists
        if Config.EU_PRICES_FINAL.exists():
            print("📁 Loading existing processed EU prices...")
            df = safe_read_csv(Config.EU_PRICES_FINAL)
            if not df.empty:
                df.index = pd.to_datetime(df.index) if 'Date' in df.columns else df.index
                print(f"✓ Loaded EU prices: {df.shape}")
                return df
        
        # Load raw data
        print("📊 Loading raw European price data...")
        df_raw = safe_read_csv(Config.EU_PRICES_RAW)
        
        if df_raw.empty:
            print("⚠️  No EU price data available")
            return pd.DataFrame()
        
        print(f"  Raw shape: {df_raw.shape}")
        
        # Convert date column
        df_raw['Date'] = pd.to_datetime(df_raw['Date'])
        
        # Pivot to wide format (countries as columns)
        print("🔄 Pivoting data...")
        df_pivot = df_raw.pivot_table(
            index='Date',
            columns='Country',
            values='Price (EUR/MWhe)',
            aggfunc='mean'
        )
        
        # Filter date range
        print("📅 Filtering to 2021-2024...")
        df_filtered = df_pivot.loc['2021-01-01':'2024-10-01'].copy()
        
        print(f"  Filtered shape: {df_filtered.shape}")
        
        # Select relevant countries for Finnish market
        print("\n🌍 Selecting relevant countries...")
        
        # Countries directly relevant to Finnish electricity market
        nordic_countries = ['Finland', 'Sweden', 'Norway', 'Denmark', 'Estonia']
        connected_markets = ['Germany', 'Poland', 'Lithuania', 'Latvia']
        major_eu_markets = ['France', 'Netherlands', 'Belgium']
        
        selected_countries = nordic_countries + connected_markets + major_eu_markets
        
        # Keep only available countries
        available_countries = [c for c in selected_countries if c in df_filtered.columns]
        df_clean = df_filtered[available_countries].copy()
        
        print(f"  Selected {len(available_countries)} countries")
        
        # Fill missing values
        df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
        
        # Add aggregate features
        print("📊 Creating aggregate price indicators...")
        
        # Nordic average (most relevant for Finland)
        nordic_cols = [c for c in nordic_countries if c in df_clean.columns]
        if nordic_cols:
            df_clean['price_nordic_avg'] = df_clean[nordic_cols].mean(axis=1)
        
        # All selected countries average
        df_clean['price_eu_avg'] = df_clean[available_countries].mean(axis=1)
        
        # Price spreads
        if 'Finland' in df_clean.columns:
            if 'price_nordic_avg' in df_clean.columns:
                df_clean['price_fi_vs_nordic'] = df_clean['Finland'] - df_clean['price_nordic_avg']
            if 'Germany' in df_clean.columns:
                df_clean['price_fi_vs_germany'] = df_clean['Finland'] - df_clean['Germany']
        
        # Rename columns with prefix
        rename_dict = {col: f'elec_price_{col.lower().replace(" ", "_")}' 
                      for col in df_clean.columns}
        df_clean.rename(columns=rename_dict, inplace=True)
        
        print(f"✓ Final EU prices shape: {df_clean.shape}")
        print(f"✓ Features created: {df_clean.shape[1]}")
        
        # Save processed data
        df_clean.to_csv(Config.EU_PRICES_FINAL)
        print(f"💾 Saved to: {Config.EU_PRICES_FINAL}")
        
        return df_clean

# ============================================================================
# SECTION 8: FEATURE ENGINEERING
# ============================================================================

class FeatureEngineer:
    """
    Create advanced features for energy consumption prediction
    """
    
    @staticmethod
    def create_temporal_features(df: pd.DataFrame, datetime_col: str = 'datetime') -> pd.DataFrame:
        """
        Create temporal features from datetime column
        
        Features include:
        - Basic: hour, day, month, year, day of week, week of year
        - Cyclical: sine/cosine transformations for periodicity
        - Special: weekend, quarter, season, etc.
        
        Args:
            df: Input DataFrame
            datetime_col: Name of datetime column
            
        Returns:
            DataFrame with temporal features added
        """
        print_subsection_header("Creating Temporal Features")
        
        if datetime_col not in df.columns:
            print(f"⚠️  DateTime column '{datetime_col}' not found")
            return df
        
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        
        # Basic temporal features
        df['hour'] = df[datetime_col].dt.hour
        df['day'] = df[datetime_col].dt.day
        df['month'] = df[datetime_col].dt.month
        df['year'] = df[datetime_col].dt.year
        df['dayofweek'] = df[datetime_col].dt.dayofweek
        df['weekofyear'] = df[datetime_col].dt.isocalendar().week
        df['dayofyear'] = df[datetime_col].dt.dayofyear
        df['quarter'] = df[datetime_col].dt.quarter
        
        # Binary features
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['is_month_start'] = df[datetime_col].dt.is_month_start.astype(int)
        df['is_month_end'] = df[datetime_col].dt.is_month_end.astype(int)
        
        # Cyclical features (sine/cosine transformations)
        # Hour of day
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week
        df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        # Day of year
        df['doy_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
        df['doy_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
        
        # Month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        print(f"✓ Created {20} temporal features")
        
        return df
    
    @staticmethod
    def create_lag_features(
        df: pd.DataFrame, 
        target_col: str, 
        lag_periods: List[int] = None,
        group_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        Create lag features for time series prediction
        
        Args:
            df: Input DataFrame
            target_col: Column to create lags for
            lag_periods: List of lag periods (default: [1, 7, 14, 30])
            group_cols: Columns to group by for lag calculation
            
        Returns:
            DataFrame with lag features added
        """
        print_subsection_header("Creating Lag Features")
        
        if target_col not in df.columns:
            print(f"⚠️  Target column '{target_col}' not found")
            return df
        
        if lag_periods is None:
            lag_periods = Config.LAG_PERIODS
        
        if group_cols:
            # Create lags within groups
            for lag in lag_periods:
                col_name = f'{target_col}_lag_{lag}'
                df[col_name] = df.groupby(group_cols)[target_col].shift(lag)
                print(f"  ✓ Created {col_name} (grouped)")
        else:
            # Simple lags
            for lag in lag_periods:
                col_name = f'{target_col}_lag_{lag}'
                df[col_name] = df[target_col].shift(lag)
                print(f"  ✓ Created {col_name}")
        
        print(f"✓ Created {len(lag_periods)} lag features")
        
        return df
    
    @staticmethod
    def create_rolling_features(
        df: pd.DataFrame, 
        target_col: str, 
        windows: List[int] = None,
        group_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        Create rolling window statistics
        
        Args:
            df: Input DataFrame
            target_col: Column to calculate rolling stats for
            windows: List of window sizes (default: [7, 14, 30])
            group_cols: Columns to group by
            
        Returns:
            DataFrame with rolling features added
        """
        print_subsection_header("Creating Rolling Window Features")
        
        if target_col not in df.columns:
            print(f"⚠️  Target column '{target_col}' not found")
            return df
        
        if windows is None:
            windows = Config.ROLLING_WINDOWS
        
        for window in windows:
            if group_cols:
                # Grouped rolling features
                df[f'{target_col}_rolling_mean_{window}'] = \
                    df.groupby(group_cols)[target_col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                df[f'{target_col}_rolling_std_{window}'] = \
                    df.groupby(group_cols)[target_col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).std()
                    )
                df[f'{target_col}_rolling_min_{window}'] = \
                    df.groupby(group_cols)[target_col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).min()
                    )
                df[f'{target_col}_rolling_max_{window}'] = \
                    df.groupby(group_cols)[target_col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).max()
                    )
            else:
                # Simple rolling features
                df[f'{target_col}_rolling_mean_{window}'] = \
                    df[target_col].rolling(window=window, min_periods=1).mean()
                df[f'{target_col}_rolling_std_{window}'] = \
                    df[target_col].rolling(window=window, min_periods=1).std()
                df[f'{target_col}_rolling_min_{window}'] = \
                    df[target_col].rolling(window=window, min_periods=1).min()
                df[f'{target_col}_rolling_max_{window}'] = \
                    df[target_col].rolling(window=window, min_periods=1).max()
            
            print(f"  ✓ Created rolling features for window={window}")
        
        print(f"✓ Created {len(windows) * 4} rolling features")
        
        return df
    
    @staticmethod
    def create_finland_specific_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features specific to Finnish energy consumption patterns
        
        Includes:
        - Daylight hours calculation (critical for Nordic countries)
        - Heating/cooling degree days
        - Finnish holidays and special periods
        - Seasonal patterns specific to Finland
        
        Args:
            df: Input DataFrame with datetime column
            
        Returns:
            DataFrame with Finland-specific features
        """
        print_subsection_header("Creating Finland-Specific Features")
        
        # Ensure we have a date column
        if 'datetime' in df.columns:
            df['date'] = pd.to_datetime(df['datetime']).dt.date
        elif 'date' not in df.columns:
            print("⚠️  No datetime/date column found")
            return df
        
        df['date'] = pd.to_datetime(df['date'])
        
        # 1. Daylight hours (extremely important for Finland)
        def calculate_daylight_hours(date, latitude=60.1695):  # Helsinki latitude
            """Calculate daylight hours for given date and latitude"""
            day_of_year = date.dayofyear
            axis_tilt = 23.44
            declination = axis_tilt * np.sin(2 * np.pi * (284 + day_of_year) / 365)
            
            lat_rad = np.radians(latitude)
            dec_rad = np.radians(declination)
            
            cos_hour_angle = -np.tan(lat_rad) * np.tan(dec_rad)
            
            # Handle polar day/night
            if cos_hour_angle > 1:
                return 0  # Polar night
            elif cos_hour_angle < -1:
                return 24  # Midnight sun
            
            hour_angle = np.arccos(cos_hour_angle)
            daylight_hours = (2 * hour_angle * 24) / (2 * np.pi)
            
            return daylight_hours
        
        print("  Calculating daylight hours...")
        df['daylight_hours'] = df['date'].apply(calculate_daylight_hours)
        df['daylight_hours_diff'] = df['daylight_hours'].diff()
        df['is_polar_night'] = (df['daylight_hours'] < 4).astype(int)
        df['is_midnight_sun'] = (df['daylight_hours'] > 20).astype(int)
        
        # 2. Heating and cooling degree days
        if 'temp' in df.columns or 'temperature' in df.columns:
            temp_col = 'temp' if 'temp' in df.columns else 'temperature'
            df['heating_degree_days'] = np.maximum(0, 18 - df[temp_col])
            df['cooling_degree_days'] = np.maximum(0, df[temp_col] - 22)
        
        # 3. Finnish holidays
        finnish_holidays = [
            (1, 1),   # New Year
            (1, 6),   # Epiphany
            (5, 1),   # May Day (Vappu)
            (6, 24),  # Midsummer Eve (Juhannus)
            (12, 6),  # Independence Day
            (12, 24), # Christmas Eve
            (12, 25), # Christmas Day
            (12, 26), # Boxing Day
        ]
        
        def is_finnish_holiday(date):
            return int((date.month, date.day) in finnish_holidays)
        
        df['is_holiday'] = df['date'].apply(is_finnish_holiday)
        
        # 4. Finnish seasons and special periods
        df['is_summer_vacation'] = df['date'].apply(
            lambda x: int(x.month in [6, 7, 8])
        )
        df['is_christmas_season'] = df['date'].apply(
            lambda x: int(x.month == 12)
        )
        df['is_midsummer_week'] = df['date'].apply(
            lambda x: int((x.month == 6) and (20 <= x.day <= 27))
        )
        
        # 5. Heating season (October to April in Finland)
        df['is_heating_season'] = df['date'].apply(
            lambda x: int(x.month in [10, 11, 12, 1, 2, 3, 4])
        )
        
        # 6. Season encoding
        def get_season(month):
            if month in [12, 1, 2]:
                return 0  # Winter
            elif month in [3, 4, 5]:
                return 1  # Spring
            elif month in [6, 7, 8]:
                return 2  # Summer
            else:
                return 3  # Autumn
        
        df['season'] = df['date'].apply(lambda x: get_season(x.month))
        
        print(f"✓ Created 14 Finland-specific features")
        
        return df

# ============================================================================
# SECTION 9: DATA INTEGRATION AND MERGING
# ============================================================================

class DataIntegrator:
    """
    Integrate all data sources into a unified dataset
    """
    
    @staticmethod
    def merge_all_data_sources() -> pd.DataFrame:
        """
        Main function to load and merge all data sources
        
        Returns:
            Unified DataFrame with all features
        """
        print_section_header("DATA INTEGRATION")
        
        # 1. Load main training data
        consumption_wide, groups, prices = DataLoader.load_training_data()
        
        if consumption_wide.empty:
            print("❌ No training data available")
            return pd.DataFrame()
        
        # 2. Reshape to long format
        consumption_long = DataLoader.reshape_hourly_to_long(consumption_wide)
        
        # 3. Parse group metadata
        groups_parsed = DataLoader.parse_group_metadata(groups)
        
        # 4. Merge consumption with group metadata
        print("\n🔗 Merging consumption with group metadata...")
        df = consumption_long.merge(groups_parsed, on="group_id", how="left")
        
        # Add datetime column
        df['datetime'] = df['measured_at']
        df['date'] = df['datetime'].dt.date.astype(str)
        df['hour'] = df['datetime'].dt.hour
        
        print(f"✓ Base dataset shape: {df.shape}")
        
        # 5. Add temporal features
        df = FeatureEngineer.create_temporal_features(df, 'datetime')
        
        # 6. Add prices
        print("\n💰 Adding electricity prices...")
        prices['measured_at'] = pd.to_datetime(prices['measured_at'])
        prices['eur_per_mwh'] = (
            prices['eur_per_mwh']
            .astype(str)
            .str.replace(",", ".", regex=False)
            .astype(float)
        )
        
        df = df.merge(
            prices[['measured_at', 'eur_per_mwh']], 
            left_on='datetime', 
            right_on='measured_at', 
            how='left',
            suffixes=('', '_price')
        )
        
        # 7. Add weather data if available
        weather_path = Config.WEATHER_HOURLY
        if weather_path.exists():
            print("\n🌤️  Adding weather data...")
            weather = safe_read_csv(weather_path)
            
            if not weather.empty:
                weather['timestamp'] = pd.to_datetime(weather['timestamp'])
                weather['date'] = weather['timestamp'].dt.date.astype(str)
                weather['hour'] = weather['timestamp'].dt.hour
                
                # Match regions/cities
                df['city_clean'] = df['city'].str.replace('_Others', '', regex=False)
                
                df = df.merge(
                    weather,
                    left_on=['city_clean', 'date', 'hour'],
                    right_on=['region', 'date', 'hour'],
                    how='left',
                    suffixes=('', '_weather')
                )
                
                print(f"✓ Weather features added")
        
        # 8. Add European electricity prices
        eu_prices_path = Config.EU_PRICES_FINAL
        if eu_prices_path.exists():
            print("\n💶 Adding European electricity prices...")
            eu_prices = safe_read_csv(eu_prices_path)
            
            if not eu_prices.empty:
                eu_prices.index = pd.to_datetime(eu_prices.index)
                eu_prices['date'] = eu_prices.index.date.astype(str)
                
                df = df.merge(
                    eu_prices,
                    on='date',
                    how='left'
                )
                
                print(f"✓ EU price features added")
        
        # 9. Add electricity market data
        elec_market_path = Config.PROCESSED_DIR / "electricity_market_daily.csv"
        if elec_market_path.exists():
            print("\n⚡ Adding electricity market data...")
            elec_market = safe_read_csv(elec_market_path)
            
            if not elec_market.empty:
                elec_market.index = pd.to_datetime(elec_market.index)
                elec_market['date'] = elec_market.index.date.astype(str)
                
                df = df.merge(
                    elec_market,
                    on='date',
                    how='left'
                )
                
                print(f"✓ Electricity market features added")
        
        # 10. Add Finland-specific features
        df = FeatureEngineer.create_finland_specific_features(df)
        
        # 11. Create lag and rolling features
        df = FeatureEngineer.create_lag_features(
            df, 
            'consumption',
            group_cols=['group_id']
        )
        
        df = FeatureEngineer.create_rolling_features(
            df,
            'consumption',
            group_cols=['group_id']
        )
        
        print(f"\n✓ Final integrated dataset shape: {df.shape}")
        print(f"✓ Total features: {df.shape[1]}")
        
        return df

# ============================================================================
# SECTION 10: FEATURE SELECTION
# ============================================================================

class FeatureSelector:
    """
    Advanced feature selection methods
    """
    
    @staticmethod
    def remove_overfitting_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove features that might cause overfitting
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with overfitting features removed
        """
        print_section_header("REMOVING OVERFITTING FEATURES")
        
        # Features that are too specific or might cause data leakage
        features_to_remove = [
            'measured_at',      # Duplicate of datetime
            'timestamp',        # Duplicate of datetime
            'index',           # Row index
            'group_label',     # Already parsed
            'city_clean',      # Temporary column
            'region_weather',  # Duplicate from weather merge
            'measured_at_price',  # Duplicate from price merge
        ]
        
        # Remove if they exist
        existing_to_remove = [col for col in features_to_remove if col in df.columns]
        
        if existing_to_remove:
            df = df.drop(columns=existing_to_remove)
            print(f"✓ Removed {len(existing_to_remove)} overfitting features")
        
        return df
    
    @staticmethod
    def select_features_chi2(
        df: pd.DataFrame, 
        target_col: str = 'consumption',
        k: int = 50
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select top features using Chi-squared test
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            k: Number of top features to select
            
        Returns:
            Tuple of (DataFrame with selected features, list of selected feature names)
        """
        print_section_header("CHI-SQUARED FEATURE SELECTION")
        
        if target_col not in df.columns:
            print(f"❌ Target column '{target_col}' not found")
            return df, []
        
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target from features
        feature_cols = [col for col in numeric_cols if col != target_col]
        
        # Remove columns with all NaN or constant values
        valid_features = []
        for col in feature_cols:
            if df[col].notna().sum() > 0 and df[col].std() > 0:
                valid_features.append(col)
        
        print(f"📋 Valid numeric features: {len(valid_features)}")
        
        if len(valid_features) == 0:
            print("❌ No valid features for selection")
            return df, []
        
        # Prepare data
        X = df[valid_features].fillna(0)
        y = df[target_col].fillna(0)
        
        # Scale features to be non-negative for chi2
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Discretize target for chi2
        y_binned = pd.qcut(y, q=10, labels=False, duplicates='drop')
        
        try:
            # Perform chi2 test
            chi2_scores, p_values = chi2(X_scaled, y_binned)
            
            # Get top k features
            top_indices = np.argsort(chi2_scores)[-k:][::-1]
            top_features = [valid_features[i] for i in top_indices]
            
            print(f"✓ Selected top {len(top_features)} features")
            print("\n📊 Top 10 features by Chi² score:")
            for i in range(min(10, len(top_features))):
                idx = top_indices[i]
                print(f"  {i+1:2d}. {valid_features[idx]:40s} | Chi²: {chi2_scores[idx]:10.2f}")
            
            # Keep target and selected features
            selected_cols = top_features + [target_col]
            
            # Add back important categorical/metadata columns
            important_cols = ['datetime', 'group_id', 'date', 'hour']
            for col in important_cols:
                if col in df.columns and col not in selected_cols:
                    selected_cols.append(col)
            
            return df[selected_cols], top_features
            
        except Exception as e:
            print(f"❌ Error during chi2 selection: {e}")
            return df, []

# ============================================================================
# SECTION 11: AUTOENCODER FOR DIMENSIONALITY REDUCTION
# ============================================================================

class CompactAutoencoder(nn.Module):
    """
    Compact autoencoder for feature compression
    """
    
    def __init__(self, input_dim: int, encoding_dim: int):
        """
        Initialize autoencoder architecture
        
        Args:
            input_dim: Number of input features
            encoding_dim: Size of compressed representation
        """
        super(CompactAutoencoder, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            
            nn.Linear(32, encoding_dim),
            nn.Tanh()  # Bounded activation for stable encoding
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, input_dim)
        )
    
    def encode(self, x):
        """Encode input to compressed representation"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode compressed representation to output"""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through full autoencoder"""
        z = self.encode(x)
        return self.decode(z)

class AutoencoderProcessor:
    """
    Process data using autoencoders for dimensionality reduction
    """
    
    def __init__(self):
        """Initialize autoencoder processor"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
    def train_autoencoder(
        self,
        df: pd.DataFrame,
        encoding_dim: int,
        window_size: int,
        epochs: int = 30,
        batch_size: int = 512
    ) -> Tuple[nn.Module, pd.DataFrame]:
        """
        Train autoencoder and return encoded features
        
        Args:
            df: Input DataFrame (already scaled)
            encoding_dim: Size of encoded representation
            window_size: Window size for temporal context
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Tuple of (trained model, encoded features DataFrame)
        """
        print(f"\n🤖 Training Autoencoder (encoding_dim={encoding_dim}, window={window_size}h)")
        
        input_dim = df.shape[1]
        
        # Create model
        model = CompactAutoencoder(input_dim, encoding_dim).to(self.device)
        
        # Prepare data loader
        tensor_data = torch.FloatTensor(df.values).to(self.device)
        dataset = torch.utils.data.TensorDataset(tensor_data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.AE_LEARNING_RATE)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                batch_data = batch[0]
                
                # Forward pass
                output = model(batch_data)
                loss = criterion(output, batch_data)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(loader)
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # Encode all data
        print("  🔄 Encoding all data...")
        model.eval()
        encoded_data = []
        
        with torch.no_grad():
            for i in range(0, len(df), 10000):
                batch_end = min(i + 10000, len(df))
                batch = tensor_data[i:batch_end]
                encoded = model.encode(batch).cpu().numpy()
                encoded_data.append(encoded)
        
        encoded_array = np.vstack(encoded_data)
        
        # Create DataFrame with encoded features
        col_names = [f'ae_{window_size}h_{i+1}' for i in range(encoding_dim)]
        df_encoded = pd.DataFrame(encoded_array, columns=col_names, index=df.index)
        
        print(f"✓ Encoded shape: {df_encoded.shape}")
        
        return model, df_encoded

# ============================================================================
# SECTION 12: MODEL TRAINING
# ============================================================================

class ModelTrainer:
    """
    Train and evaluate forecasting models
    """
    
    def __init__(self):
        """Initialize model trainer"""
        self.models_48h = {}
        self.models_12m = {}
        self.best_model_48h = None
        self.best_model_12m = None
        
    def train_48hour_models(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> Dict:
        """
        Train models for 48-hour ahead consumption prediction
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary with model performance metrics
        """
        print_section_header("TRAINING 48-HOUR CONSUMPTION MODELS")
        
        # Define models
        models = {
            'LightGBM': LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=8,
                num_leaves=31,
                random_state=Config.RANDOM_STATE,
                verbose=-1
            ),
            'XGBoost': XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=8,
                random_state=Config.RANDOM_STATE,
                verbosity=0
            ),
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=Config.RANDOM_STATE,
                n_jobs=-1
            )
        }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=Config.CV_SPLITS)
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\n🔧 Training {model_name}...")
            
            scores = {'mae': [], 'rmse': [], 'mape': [], 'r2': []}
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_val = y_train.iloc[val_idx]
                
                # Train model
                model.fit(X_fold_train, y_fold_train)
                
                # Predict
                y_pred = model.predict(X_fold_val)
                
                # Calculate metrics
                mae = mean_absolute_error(y_fold_val, y_pred)
                rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
                mape = mean_absolute_percentage_error(y_fold_val, y_pred) * 100
                r2 = r2_score(y_fold_val, y_pred)
                
                scores['mae'].append(mae)
                scores['rmse'].append(rmse)
                scores['mape'].append(mape)
                scores['r2'].append(r2)
                
                print(f"  Fold {fold}: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%, R²={r2:.4f}")
            
            # Store results
            results[model_name] = {
                'mae_mean': np.mean(scores['mae']),
                'mae_std': np.std(scores['mae']),
                'rmse_mean': np.mean(scores['rmse']),
                'rmse_std': np.std(scores['rmse']),
                'mape_mean': np.mean(scores['mape']),
                'mape_std': np.std(scores['mape']),
                'r2_mean': np.mean(scores['r2']),
                'r2_std': np.std(scores['r2'])
            }
            
            # Train final model on all data
            model.fit(X_train, y_train)
            self.models_48h[model_name] = model
            
            print(f"\n  📊 {model_name} CV Results:")
            print(f"    MAE:  {results[model_name]['mae_mean']:.4f} ± {results[model_name]['mae_std']:.4f}")
            print(f"    RMSE: {results[model_name]['rmse_mean']:.4f} ± {results[model_name]['rmse_std']:.4f}")
            print(f"    MAPE: {results[model_name]['mape_mean']:.2f}% ± {results[model_name]['mape_std']:.2f}%")
            print(f"    R²:   {results[model_name]['r2_mean']:.4f} ± {results[model_name]['r2_std']:.4f}")
        
        # Select best model
        best_model_name = min(results, key=lambda x: results[x]['mape_mean'])
        self.best_model_48h = self.models_48h[best_model_name]
        
        print(f"\n✅ Best 48h model: {best_model_name}")
        
        return results
    
    def train_monthly_models(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> Dict:
        """
        Train models for 12-month ahead consumption prediction
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary with model performance metrics
        """
        print_section_header("TRAINING 12-MONTH CONSUMPTION MODELS")
        
        # Define models
        models = {
            'LightGBM': LGBMRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=20,
                random_state=Config.RANDOM_STATE,
                verbose=-1
            ),
            'XGBoost': XGBRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=6,
                random_state=Config.RANDOM_STATE,
                verbosity=0
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=6,
                random_state=Config.RANDOM_STATE
            )
        }
        
        # Time series cross-validation
        n_splits = min(Config.CV_SPLITS, len(X_train) // 12)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\n🔧 Training {model_name}...")
            
            scores = {'mae': [], 'rmse': [], 'mape': [], 'r2': []}
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
                if len(train_idx) < 12:
                    continue
                    
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_val = y_train.iloc[val_idx]
                
                # Train model
                model.fit(X_fold_train, y_fold_train)
                
                # Predict
                y_pred = model.predict(X_fold_val)
                
                # Calculate metrics
                mae = mean_absolute_error(y_fold_val, y_pred)
                rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
                mape = mean_absolute_percentage_error(y_fold_val, y_pred) * 100
                r2 = r2_score(y_fold_val, y_pred)
                
                scores['mae'].append(mae)
                scores['rmse'].append(rmse)
                scores['mape'].append(mape)
                scores['r2'].append(r2)
                
                print(f"  Fold {fold}: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%, R²={r2:.4f}")
            
            if len(scores['mae']) > 0:
                # Store results
                results[model_name] = {
                    'mae_mean': np.mean(scores['mae']),
                    'mae_std': np.std(scores['mae']),
                    'rmse_mean': np.mean(scores['rmse']),
                    'rmse_std': np.std(scores['rmse']),
                    'mape_mean': np.mean(scores['mape']),
                    'mape_std': np.std(scores['mape']),
                    'r2_mean': np.mean(scores['r2']),
                    'r2_std': np.std(scores['r2'])
                }
                
                # Train final model on all data
                model.fit(X_train, y_train)
                self.models_12m[model_name] = model
                
                print(f"\n  📊 {model_name} CV Results:")
                print(f"    MAE:  {results[model_name]['mae_mean']:.4f} ± {results[model_name]['mae_std']:.4f}")
                print(f"    RMSE: {results[model_name]['rmse_mean']:.4f} ± {results[model_name]['rmse_std']:.4f}")
                print(f"    MAPE: {results[model_name]['mape_mean']:.2f}% ± {results[model_name]['mape_std']:.2f}%")
                print(f"    R²:   {results[model_name]['r2_mean']:.4f} ± {results[model_name]['r2_std']:.4f}")
        
        if results:
            # Select best model
            best_model_name = min(results, key=lambda x: results[x]['mape_mean'])
            self.best_model_12m = self.models_12m[best_model_name]
            
            print(f"\n✅ Best monthly model: {best_model_name}")
        
        return results

# ============================================================================
# SECTION 13: MAIN PIPELINE
# ============================================================================

def main():
    """
    Main pipeline execution function
    """
    print("\n" + "=" * 80)
    print("🚀 STARTING FORTUM JUNCTION 2025 PIPELINE")
    print("=" * 80)
    
    # 1. Load and integrate all data sources
    print("\n📊 Step 1: Loading and integrating data...")
    df = DataIntegrator.merge_all_data_sources()
    
    if df.empty:
        print("❌ Failed to load data. Exiting.")
        return
    
    # Save integrated raw data
    raw_output = Config.OUTPUT_DIR / "integrated_raw_data.csv"
    df.to_csv(raw_output, index=False)
    print(f"💾 Saved raw integrated data to: {raw_output}")
    
    # 2. Clean and prepare data
    print("\n🧹 Step 2: Cleaning and preparing data...")
    
    # Remove overfitting features
    df = FeatureSelector.remove_overfitting_features(df)
    
    # Handle missing values
    print("  Handling missing values...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # 3. Feature selection
    print("\n🎯 Step 3: Feature selection...")
    
    # Select features for hourly consumption
    df_selected, selected_features = FeatureSelector.select_features_chi2(
        df, 
        target_col='consumption',
        k=50
    )
    
    # 4. Prepare training data
    print("\n📚 Step 4: Preparing training data...")
    
    # Define target columns
    target_hourly = 'consumption'
    target_daily = 'consumption'  # Will be aggregated
    
    # Create hourly dataset
    hourly_features = [col for col in df_selected.columns 
                      if col not in ['datetime', 'date', 'group_id']]
    
    # Remove target from features
    hourly_features = [col for col in hourly_features if col != target_hourly]
    
    # Scale features
    print("  Scaling features...")
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_selected[hourly_features]),
        columns=hourly_features,
        index=df_selected.index
    )
    
    # 5. Autoencoder dimensionality reduction
    print("\n🤖 Step 5: Autoencoder dimensionality reduction...")
    
    ae_processor = AutoencoderProcessor()
    
    # Hourly autoencoder
    model_hourly, df_encoded_hourly = ae_processor.train_autoencoder(
        df_scaled,
        encoding_dim=Config.AE_ENCODING_DIM_HOURLY,
        window_size=Config.AE_WINDOW_HOURLY
    )
    
    # Daily autoencoder
    model_daily, df_encoded_daily = ae_processor.train_autoencoder(
        df_scaled,
        encoding_dim=Config.AE_ENCODING_DIM_DAILY,
        window_size=Config.AE_WINDOW_DAILY
    )
    
    # 6. Create final datasets
    print("\n📦 Step 6: Creating final datasets...")
    
    # Hourly dataset
    df_hourly_final = pd.concat([
        df_selected[['datetime', 'group_id']],
        df_encoded_hourly,
        df_selected[[target_hourly]]
    ], axis=1)
    
    # Remove rows with NaN in target
    df_hourly_final = df_hourly_final.dropna(subset=[target_hourly])
    
    # Daily dataset (aggregate to daily)
    df_selected['date'] = pd.to_datetime(df_selected['datetime']).dt.date
    daily_agg = df_selected.groupby(['date', 'group_id'])[target_daily].sum().reset_index()
    daily_agg.columns = ['date', 'group_id', 'daily_consumption']
    
    # Add encoded features to daily
    df_daily_encoded = df_encoded_daily.copy()
    df_daily_encoded['date'] = pd.to_datetime(df_selected['datetime']).dt.date
    df_daily_encoded['group_id'] = df_selected['group_id']
    
    daily_encoded_agg = df_daily_encoded.groupby(['date', 'group_id']).mean().reset_index()
    
    df_daily_final = daily_agg.merge(
        daily_encoded_agg,
        on=['date', 'group_id'],
        how='left'
    )
    
    # Save final datasets
    hourly_output = Config.OUTPUT_DIR / "dataset_hourly_consumption.csv"
    daily_output = Config.OUTPUT_DIR / "dataset_daily_consumption.csv"
    
    df_hourly_final.to_csv(hourly_output, index=False)
    df_daily_final.to_csv(daily_output, index=False)
    
    print(f"💾 Saved hourly dataset to: {hourly_output}")
    print(f"💾 Saved daily dataset to: {daily_output}")
    
    # 7. Train models
    print("\n🎓 Step 7: Training forecasting models...")
    
    trainer = ModelTrainer()
    
    # Prepare training data
    cutoff = pd.to_datetime(Config.CUTOFF_DATE)
    
    # Hourly model training
    train_mask = pd.to_datetime(df_hourly_final['datetime']) <= cutoff
    X_train_hourly = df_hourly_final[train_mask].drop(columns=['datetime', 'group_id', target_hourly])
    y_train_hourly = df_hourly_final[train_mask][target_hourly]
    
    if len(X_train_hourly) > 0:
        results_48h = trainer.train_48hour_models(X_train_hourly, y_train_hourly)
    
    # Monthly model training (if enough data)
    if len(df_daily_final) > 30:
        # Aggregate to monthly
        df_daily_final['month'] = pd.to_datetime(df_daily_final['date']).dt.to_period('M')
        monthly_agg = df_daily_final.groupby(['month', 'group_id']).agg({
            'daily_consumption': 'sum',
            **{col: 'mean' for col in df_daily_final.columns 
              if col.startswith('ae_') or col.startswith('elec_')}
        }).reset_index()
        
        # Filter training data
        train_mask = monthly_agg['month'] <= pd.Period(Config.CUTOFF_DATE, 'M')
        X_train_monthly = monthly_agg[train_mask].drop(
            columns=['month', 'group_id', 'daily_consumption']
        )
        y_train_monthly = monthly_agg[train_mask]['daily_consumption']
        
        if len(X_train_monthly) > 12:
            results_12m = trainer.train_monthly_models(X_train_monthly, y_train_monthly)
    
    # 8. Final report
    print_section_header("PIPELINE COMPLETE")
    
    print("\n✅ FORTUM JUNCTION 2025 PIPELINE EXECUTED SUCCESSFULLY!")
    
    print("\n📊 FINAL DATASETS CREATED:")
    print(f"  • Hourly consumption dataset: {df_hourly_final.shape}")
    print(f"    - Features: {df_hourly_final.shape[1] - 3}")
    print(f"    - Records: {df_hourly_final.shape[0]:,}")
    
    print(f"\n  • Daily consumption dataset: {df_daily_final.shape}")
    print(f"    - Features: {df_daily_final.shape[1] - 3}")
    print(f"    - Records: {df_daily_final.shape[0]:,}")
    
    print("\n🎯 MODELS TRAINED:")
    print(f"  • 48-hour forecasting models: {len(trainer.models_48h)}")
    print(f"  • 12-month forecasting models: {len(trainer.models_12m)}")
    
    print("\n📁 OUTPUT FILES:")
    print(f"  • {Config.OUTPUT_DIR}/integrated_raw_data.csv")
    print(f"  • {Config.OUTPUT_DIR}/dataset_hourly_consumption.csv")
    print(f"  • {Config.OUTPUT_DIR}/dataset_daily_consumption.csv")
    
    print("\n🚀 NEXT STEPS:")
    print("  1. Load the final datasets for production model training")
    print("  2. Implement advanced models (LSTM, Temporal Fusion Transformer)")
    print("  3. Create ensemble predictions")
    print("  4. Generate submission file for the hackathon")
    
    print("\n" + "=" * 80)
    print("GOOD LUCK WITH THE HACKATHON! 🎉")
    print("=" * 80)

# ============================================================================
# SECTION 14: SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(Config.RANDOM_STATE)
    torch.manual_seed(Config.RANDOM_STATE)
    
    # Run main pipeline
    main()