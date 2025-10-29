import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')

from config.settings import DATA_PROCESSED, MODELS_DIR

def save_processed_data(df, filename):
    """Save processed data to file"""
    filepath = os.path.join(DATA_PROCESSED, filename)
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")

def load_processed_data(filename):
    """Load processed data from file"""
    filepath = os.path.join(DATA_PROCESSED, filename)
    df = pd.read_csv(filepath)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    print(f"Data loaded from {filepath}")
    return df

def generate_sample_data(n_days=1000):
    """Generate sample data for testing"""
    dates = pd.date_range(start='2018-01-01', periods=n_days, freq='D')
    
    data = {
        'date': dates,
        'pm25': np.random.normal(35, 15, n_days),
        'aai': np.random.normal(1.2, 0.5, n_days),
        'no2': np.random.normal(0.0002, 0.0001, n_days),
        'temperature': np.random.normal(20, 3, n_days),
        'wind_speed': np.random.normal(3, 1, n_days),
        'humidity': np.random.normal(60, 20, n_days),
        'precipitation': np.random.exponential(0.5, n_days),
        'road_density_km_km2': np.random.uniform(0, 20, n_days),
        'urban_cover_percent': np.random.uniform(0, 100, n_days),
        'latitude': np.random.normal(9.0, 0.1, n_days),
        'longitude': np.random.normal(38.76, 0.1, n_days)
    }
    
    df = pd.DataFrame(data)
    # Ensure PM2.5 is positive
    df['pm25'] = np.abs(df['pm25'])
    
    return df

def calculate_air_quality_index(pm25):
    """Calculate air quality index based on PM2.5"""
    if pm25 <= 12:
        return "Good", "green"
    elif pm25 <= 35.4:
        return "Moderate", "yellow"
    elif pm25 <= 55.4:
        return "Unhealthy for Sensitive Groups", "orange"
    elif pm25 <= 150.4:
        return "Unhealthy", "red"
    elif pm25 <= 250.4:
        return "Very Unhealthy", "purple"
    else:
        return "Hazardous", "maroon"

def create_prediction_intervals(predictions, confidence=0.95):
    """Create prediction intervals"""
    if len(predictions) == 0:
        return predictions, predictions
    
    # Simple prediction intervals based on empirical distribution
    std = np.std(predictions)
    z_score = 1.96 if confidence == 0.95 else 2.58  # 95% or 99% CI
    
    lower = predictions - z_score * std
    upper = predictions + z_score * std
    
    return lower, upper

def format_large_number(number):
    """Format large numbers for display"""
    if number >= 1e6:
        return f"{number/1e6:.1f}M"
    elif number >= 1e3:
        return f"{number/1e3:.1f}K"
    else:
        return f"{number:.0f}"

def get_feature_descriptions():
    """Get descriptions for features"""
    return {
        'pm25': 'Particulate Matter 2.5 micrometers (μg/m³) - primary air quality indicator',
        'aai': 'Absorbing Aerosol Index - satellite-based aerosol measurement',
        'no2': 'Nitrogen Dioxide tropospheric column (mol/m²) - pollution indicator',
        'temperature': 'Air temperature at 2m height (°C)',
        'wind_speed': 'Wind speed at 10m height (m/s)',
        'humidity': 'Relative humidity (%)',
        'precipitation': 'Daily precipitation (mm)',
        'road_density_km_km2': 'Road density from OpenStreetMap (km/km²)',
        'urban_cover_percent': 'Urban land cover percentage from OSM (%)',
        'month_sin': 'Cyclical encoding of month (sine component)',
        'month_cos': 'Cyclical encoding of month (cosine component)'
    }