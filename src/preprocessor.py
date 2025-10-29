import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.feature_columns = [
            'aai', 'no2', 'temperature', 'wind_speed', 'humidity', 
            'precipitation', 'road_density_km_km2', 'urban_cover_percent'
        ]
        self.preprocessor = None
        self.feature_names = None
        
    def create_temporal_features(self, df):
        """Create temporal features from date"""
        df = df.copy()
        
        # Basic temporal features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['day_of_week'] = df['date'].dt.dayofweek
        df['weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['quarter'] = df['date'].dt.quarter
        
        # Seasonal features
        df['season'] = df['month'] % 12 // 3 + 1
        
        # Cyclical encoding for temporal features
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year']/365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year']/365)
        
        return df
    
    def create_interaction_features(self, df):
        """Create interaction features"""
        df = df.copy()
        print("Available columns:", df.columns.tolist())
        
        # Weather interactions
        df['temp_humidity'] = df['temperature'] * df['humidity']
        df['wind_precip'] = df['wind_speed'] * df['precipitation']
        
        # Pollution interactions
        df['aai_no2'] = df['aai'] * df['no2']
        df['urban_pollution'] = df['urban_cover_percent'] * df['aai']
        
        # Seasonal pollution patterns
        df['winter_pollution'] = ((df['season'] == 1) | (df['season'] == 4)) * df['aai']
        df['summer_pollution'] = (df['season'] == 2) * df['aai']
        
        return df
    
    def create_lag_features(self, df, lags=[1, 2, 3, 7, 14, 30]):
        """Create lag features for time series"""
        df = df.copy().sort_values('date')
        
        # Lag features for PM2.5
        for lag in lags:
            df[f'pm25_lag_{lag}'] = df['pm25'].shift(lag)
        
        # Lag features for satellite data
        for col in ['aai', 'no2']:
            if col in df.columns:
                for lag in [1, 2, 3, 7]:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Rolling statistics
        windows = [3, 7, 14, 30]
        for window in windows:
            df[f'pm25_rolling_mean_{window}'] = df['pm25'].rolling(window=window, min_periods=1).mean()
            df[f'pm25_rolling_std_{window}'] = df['pm25'].rolling(window=window, min_periods=1).std()
            df[f'aai_rolling_mean_{window}'] = df['aai'].rolling(window=window, min_periods=1).mean()
        
        return df
    
    def prepare_features(self, df):
        """Prepare all features for modeling"""
        print("Creating temporal features...")
        df = self.create_temporal_features(df)
        
        print("Creating interaction features...")
        df = self.create_interaction_features(df)
        
        print("Creating lag features...")
        df = self.create_lag_features(df)
        
        # Select final feature set
        all_features = [col for col in df.columns if col not in ['date', 'pm25', 'year']]
        self.feature_names = all_features
        
        print(f"Total features created: {len(all_features)}")
        return df[all_features], df['pm25']
    
    def create_preprocessor(self, X):
        """Create preprocessing pipeline"""
        # Identify numeric and categorical features
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove any non-numeric features that might have been included
        numeric_features = [f for f in numeric_features if f in X.columns]
        
        print(f"Processing {len(numeric_features)} numeric features")
        
        # Create preprocessing pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numeric_features)
            ],
            remainder='drop'
        )
        
        return self.preprocessor
    
    def get_feature_names(self):
        """Get feature names after preprocessing"""
        if self.preprocessor is None:
            return self.feature_names
        
        # Get feature names from the preprocessor
        feature_names = []
        for name, trans, features in self.preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(features)
        
        return feature_names