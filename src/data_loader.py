import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import os
from config.settings import DATA_RAW, DATA_PROCESSED

class DataLoader:
    def __init__(self):
        self.data_paths = {
            'ground_measurements': os.path.join(DATA_RAW, 'ground_measurements.csv'),
            'satellite_data': os.path.join(DATA_RAW, 'satellite_data.csv'),
            'land_use_data': os.path.join(DATA_RAW, 'land_use_data.csv'),
            'weather_data': os.path.join(DATA_RAW, 'weather_data.csv')
        }
    
    def load_ground_measurements(self):
        """Load PM2.5 ground measurements"""
        df = pd.read_csv(self.data_paths['ground_measurements'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        print(f"Loaded {len(df)} ground measurements")
        return df
    
    def load_satellite_data(self):
        """Load satellite data (AAI, NO2)"""
        df = pd.read_csv(self.data_paths['satellite_data'])
        df['date'] = pd.to_datetime(df['date'])
        print(f"Loaded {len(df)} satellite records")
        return df
    
    def load_land_use_data(self):
        """Load land use data from OSM"""
        df = pd.read_csv(self.data_paths['land_use_data'])
        print(f"Loaded {len(df)} land use records")
        return df
    
    def load_weather_data(self):
        """Load meteorological data"""
        df = pd.read_csv(self.data_paths['weather_data'])
        df['date'] = pd.to_datetime(df['date'])
        print(f"Loaded {len(df)} weather records")
        return df
    
    def merge_all_data(self):
        """Merge all datasets into a single DataFrame"""
        # Load individual datasets
        ground_df = self.load_ground_measurements()
        satellite_df = self.load_satellite_data()
        land_use_df = self.load_land_use_data()
        weather_df = self.load_weather_data()
        
        # Merge satellite data with ground measurements
        merged_df = pd.merge(ground_df, satellite_df, on='date', how='left')
        
        # Merge with weather data
        merged_df = pd.merge(merged_df, weather_df, on='date', how='left')
        
        # Add land use data (spatial join based on coordinates)
        if 'latitude' in merged_df.columns and 'longitude' in merged_df.columns:
            # Clean coordinates before spatial join
            from sklearn.neighbors import NearestNeighbors
            import numpy as np
            
            # Remove rows with missing coordinates
            valid_coords = merged_df[['latitude', 'longitude']].dropna()
            print(f"Valid coordinates for spatial join: {len(valid_coords)}")
            
            if len(valid_coords) > 0 and len(land_use_df) > 0:
                # Prepare coordinates - ensure land use data has valid coordinates too
                land_use_coords = land_use_df[['centroid_lat', 'centroid_lon']].dropna()
                
                if len(land_use_coords) > 0:
                    # Prepare main coordinates (only valid ones)
                    main_coords = valid_coords.values
                    land_use_coords_values = land_use_coords.values
                    
                    # Find nearest land use data for each point
                    nbrs = NearestNeighbors(n_neighbors=1, metric='haversine').fit(land_use_coords_values)
                    distances, indices = nbrs.kneighbors(main_coords)
                    
                    # Add land use features - only for rows with valid coordinates
                    valid_indices = valid_coords.index
                    for col in ['road_density_km_km2', 'urban_cover_percent']:
                        if col in land_use_df.columns:
                            merged_df.loc[valid_indices, col] = land_use_df[col].iloc[indices.flatten()].values
                else:
                    print("Warning: No valid land use coordinates found")
            else:
                print("Warning: Not enough valid data for spatial join")
        
        print(f"Merged dataset shape: {merged_df.shape}")
        
        # Check for any remaining NaN values
        nan_count = merged_df.isna().sum().sum()
        if nan_count > 0:
            print(f"Warning: {nan_count} NaN values remaining in merged dataset")
            print("NaN values per column:")
            for col in merged_df.columns:
                missing = merged_df[col].isna().sum()
                if missing > 0:
                    print(f"  {col}: {missing}")
        
        return merged_df
    
    def merge_all_data_orig(self):
        """Merge all datasets into a single DataFrame"""
        # Load individual datasets
        ground_df = self.load_ground_measurements()
        satellite_df = self.load_satellite_data()
        land_use_df = self.load_land_use_data()
        weather_df = self.load_weather_data()
        
        # Merge satellite data with ground measurements
        merged_df = pd.merge(ground_df, satellite_df, on='date', how='left')
        
        # Merge with weather data
        merged_df = pd.merge(merged_df, weather_df, on='date', how='left')
        
        # Add land use data (spatial join based on coordinates)
        if 'latitude' in merged_df.columns and 'longitude' in merged_df.columns:
            # Simple nearest neighbor join for land use data
            from sklearn.neighbors import NearestNeighbors
            import numpy as np
            
            # Prepare coordinates
            main_coords = merged_df[['latitude', 'longitude']].values
            land_use_coords = land_use_df[['centroid_lat', 'centroid_lon']].values
            
            # Find nearest land use data for each point
            nbrs = NearestNeighbors(n_neighbors=1, metric='haversine').fit(land_use_coords)
            distances, indices = nbrs.kneighbors(main_coords)
            
            # Add land use features
            for col in ['road_density_km_km2', 'urban_cover_percent']:
                if col in land_use_df.columns:
                    merged_df[col] = land_use_df[col].iloc[indices.flatten()].values
        
        print(f"Merged dataset shape: {merged_df.shape}")
        return merged_df
    
    def validate_data(self, df):
        """Validate data quality and completeness"""
        print("\n=== DATA VALIDATION ===")
        print(f"Total records: {len(df)}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Missing values per column:")
        for col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                print(f"  {col}: {missing} ({missing/len(df)*100:.1f}%)")
        
        # Check PM2.5 statistics
        if 'pm25' in df.columns:
            pm25_stats = df['pm25'].describe()
            print(f"\nPM2.5 Statistics:")
            print(f"  Mean: {pm25_stats['mean']:.2f} μg/m³")
            print(f"  Std: {pm25_stats['std']:.2f} μg/m³")
            print(f"  Min: {pm25_stats['min']:.2f} μg/m³")
            print(f"  Max: {pm25_stats['max']:.2f} μg/m³")
            print(f"  WHO Guideline Exceedance: {(df['pm25'] > 15).sum()} days ({(df['pm25'] > 15).mean()*100:.1f}%)")
        
        return df