import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from plotly.colors import qualitative, sequential
import folium
from folium import plugins
import geopandas as gpd
from shapely.geometry import Point

class AdvancedVisualizer:
    def __init__(self):
        self.colors = qualitative.Set3
        self.sequential_colors = sequential.Plasma
    
    def create_timeseries_plot(self, df, title="PM2.5 Time Series Analysis"):
        """Create interactive time series plot with multiple features"""
        fig = sp.make_subplots(
            rows=3, cols=1,
            subplot_titles=['PM2.5 Concentration', 'Satellite Indicators', 'Meteorological Conditions'],
            vertical_spacing=0.08,
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # PM2.5 plot
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['pm25'], name='PM2.5',
                      line=dict(color='red', width=2),
                      hovertemplate='<b>%{x}</b><br>PM2.5: %{y:.1f} μg/m³<extra></extra>'),
            row=1, col=1
        )
        
        # Add WHO guideline
        fig.add_hline(y=15, line_dash="dash", line_color="red", 
                     annotation_text="WHO Guideline", row=1, col=1)
        
        # Satellite data
        if 'aai' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['date'], y=df['aai'], name='AAI',
                          line=dict(color='orange', width=1.5),
                          hovertemplate='AAI: %{y:.2f}<extra></extra>'),
                row=2, col=1
            )
        
        if 'no2' in df.columns:
            # Scale NO2 for better visualization
            no2_scaled = df['no2'] * 1000 if 'no2' in df.columns else None
            fig.add_trace(
                go.Scatter(x=df['date'], y=no2_scaled, name='NO₂ (scaled)',
                          line=dict(color='purple', width=1.5),
                          hovertemplate='NO₂: %{y:.4f}<extra></extra>'),
                row=2, col=1
            )
        
        # Meteorological data
        if 'temperature' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['date'], y=df['temperature'], name='Temperature',
                          line=dict(color='blue', width=1.5),
                          hovertemplate='Temp: %{y:.1f}°C<extra></extra>'),
                row=3, col=1
            )
        
        if 'wind_speed' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['date'], y=df['wind_speed'], name='Wind Speed',
                          line=dict(color='green', width=1.5),
                          hovertemplate='Wind: %{y:.1f} m/s<extra></extra>'),
                row=3, col=1
            )
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_yaxes(title_text="PM2.5 (μg/m³)", row=1, col=1)
        fig.update_yaxes(title_text="Satellite Index", row=2, col=1)
        fig.update_yaxes(title_text="Meteorological", row=3, col=1)
        
        return fig
    
    def create_correlation_heatmap(self, df, columns=None):
        """Create interactive correlation heatmap"""
        if columns is None:
            columns = ['pm25', 'aai', 'no2', 'temperature', 'wind_speed', 
                      'humidity', 'precipitation', 'road_density_km_km2', 'urban_cover_percent']
        
        # Select only available columns
        available_columns = [col for col in columns if col in df.columns]
        corr_matrix = df[available_columns].corr()
        
        fig = px.imshow(
            corr_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu_r",
            aspect="auto",
            zmin=-1, zmax=1
        )
        
        # Add annotations
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                fig.add_annotation(
                    x=i, y=j,
                    text=f"{corr_matrix.iloc[i, j]:.2f}",
                    showarrow=False,
                    font=dict(color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
                )
        
        return fig
    
    def create_seasonal_analysis(self, df):
        """Create seasonal analysis plots"""
        if 'month' not in df.columns:
            df['month'] = df['date'].dt.month
        
        monthly_stats = df.groupby('month').agg({
            'pm25': ['mean', 'std', 'min', 'max'],
            'aai': 'mean',
            'no2': 'mean',
            'temperature': 'mean'
        }).round(2)
        
        monthly_stats.columns = ['_'.join(col).strip() for col in monthly_stats.columns.values]
        monthly_stats = monthly_stats.reset_index()
        
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=['Monthly PM2.5 Patterns', 'Monthly AAI Patterns', 
                           'Monthly NO₂ Patterns', 'Temperature Correlation'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": True}]]
        )
        
        # PM2.5 by month
        fig.add_trace(
            go.Scatter(x=monthly_stats['month'], y=monthly_stats['pm25_mean'],
                      name='PM2.5 Mean', line=dict(color='red', width=3)),
            row=1, col=1
        )
        
        # Add confidence interval
        fig.add_trace(
            go.Scatter(x=monthly_stats['month'], 
                      y=monthly_stats['pm25_mean'] + monthly_stats['pm25_std'],
                      name='+1 Std', line=dict(width=0), showlegend=False),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=monthly_stats['month'], 
                      y=monthly_stats['pm25_mean'] - monthly_stats['pm25_std'],
                      name='-1 Std', line=dict(width=0), fill='tonexty',
                      fillcolor='rgba(255,0,0,0.2)', showlegend=False),
            row=1, col=1
        )
        
        # AAI by month
        if 'aai_mean' in monthly_stats.columns:
            fig.add_trace(
                go.Scatter(x=monthly_stats['month'], y=monthly_stats['aai_mean'],
                          name='AAI Mean', line=dict(color='orange', width=2)),
                row=1, col=2
            )
        
        # NO2 by month
        if 'no2_mean' in monthly_stats.columns:
            fig.add_trace(
                go.Scatter(x=monthly_stats['month'], y=monthly_stats['no2_mean'],
                          name='NO₂ Mean', line=dict(color='purple', width=2)),
                row=2, col=1
            )
        
        # Temperature vs PM2.5
        if 'temperature_mean' in monthly_stats.columns:
            fig.add_trace(
                go.Scatter(x=monthly_stats['month'], y=monthly_stats['temperature_mean'],
                          name='Temperature', line=dict(color='blue', width=2)),
                row=2, col=2, secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(x=monthly_stats['month'], y=monthly_stats['pm25_mean'],
                          name='PM2.5', line=dict(color='red', width=2)),
                row=2, col=2, secondary_y=True
            )
        
        fig.update_layout(height=600, title_text="Seasonal Analysis")
        fig.update_xaxes(title_text="Month", row=2, col=1)
        fig.update_xaxes(title_text="Month", row=2, col=2)
        fig.update_yaxes(title_text="PM2.5 (μg/m³)", row=1, col=1)
        fig.update_yaxes(title_text="AAI", row=1, col=2)
        fig.update_yaxes(title_text="NO₂", row=2, col=1)
        fig.update_yaxes(title_text="Temperature (°C)", row=2, col=2, secondary_y=False)
        fig.update_yaxes(title_text="PM2.5 (μg/m³)", row=2, col=2, secondary_y=True)
        
        return fig
    
    def create_feature_importance_plot(self, importance_df, top_n=15):
        """Create interactive feature importance plot"""
        fig = px.bar(
            importance_df.head(top_n),
            x='importance', y='feature',
            title=f'Top {top_n} Most Important Features',
            labels={'importance': 'Importance Score', 'feature': 'Feature'},
            color='importance',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False
        )
        
        return fig
    
    def create_prediction_analysis(self, y_true, y_pred, dates=None):
        """Create comprehensive prediction analysis"""
        residuals = y_true - y_pred
        
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=['Predictions vs Actual', 'Residuals Distribution', 
                           'Residuals vs Predicted', 'Prediction Error Over Time'],
            specs=[[{}, {}], [{}, {}]]
        )
        
        # Predictions vs Actual
        fig.add_trace(
            go.Scatter(x=y_true, y=y_pred, mode='markers',
                      name='Predictions', marker=dict(color='blue', opacity=0.6)),
            row=1, col=1
        )
        
        # Perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                      mode='lines', name='Perfect Prediction',
                      line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # Residuals distribution
        fig.add_trace(
            go.Histogram(x=residuals, name='Residuals',
                        marker_color='orange', nbinsx=50),
            row=1, col=2
        )
        
        # Residuals vs Predicted
        fig.add_trace(
            go.Scatter(x=y_pred, y=residuals, mode='markers',
                      name='Residuals', marker=dict(color='green', opacity=0.6)),
            row=2, col=1
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
        
        # Prediction error over time (if dates provided)
        if dates is not None:
            fig.add_trace(
                go.Scatter(x=dates, y=residuals, mode='lines',
                          name='Residuals Over Time', line=dict(color='purple')),
                row=2, col=2
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)
        
        fig.update_layout(height=600, title_text="Model Prediction Analysis")
        fig.update_xaxes(title_text="Actual PM2.5", row=1, col=1)
        fig.update_yaxes(title_text="Predicted PM2.5", row=1, col=1)
        fig.update_xaxes(title_text="Residual Value", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_xaxes(title_text="Predicted PM2.5", row=2, col=1)
        fig.update_yaxes(title_text="Residuals", row=2, col=1)
        
        if dates is not None:
            fig.update_xaxes(title_text="Date", row=2, col=2)
            fig.update_yaxes(title_text="Residuals", row=2, col=2)
        
        return fig
    
    def create_spatial_map(self, df, value_column='pm25', title="Spatial Distribution"):
        """Create interactive folium map"""
        # Center on Addis Ababa
        m = folium.Map(location=[9.0227, 38.7635], zoom_start=11)
        
        # Add points to map
        for idx, row in df.iterrows():
            if pd.notna(row[value_column]) and pd.notna(row['latitude']) and pd.notna(row['longitude']):
                # Color based on value
                value = row[value_column]
                if value_column == 'pm25':
                    color = 'red' if value > 15 else 'green'
                else:
                    color = 'blue'
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=8,
                    popup=f"{value_column}: {value:.1f}<br>Date: {row['date']}",
                    tooltip=f"{value_column}: {value:.1f}",
                    color=color,
                    fill=True,
                    fillColor=color
                ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        return m
    
    def create_model_comparison(self, results):
        """Create model comparison visualization"""
        models = list(results.keys())
        metrics = ['mae', 'rmse', 'r2']
        
        fig = sp.make_subplots(
            rows=1, cols=3,
            subplot_titles=['MAE Comparison', 'RMSE Comparison', 'R² Comparison']
        )
        
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in models]
            
            # For R², higher is better - reverse for consistent coloring
            if metric == 'r2':
                colors = ['green' if x == max(values) else 'lightgray' for x in values]
            else:
                colors = ['red' if x == min(values) else 'lightgray' for x in values]
            
            fig.add_trace(
                go.Bar(x=models, y=values, name=metric.upper(),
                      marker_color=colors,
                      hovertemplate=f'{metric.upper()}: %{{y:.3f}}<extra></extra>'),
                row=1, col=i+1
            )
        
        fig.update_layout(
            height=400,
            title_text="Model Performance Comparison",
            showlegend=False
        )
        
        return fig