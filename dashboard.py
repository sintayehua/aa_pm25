#!/usr/bin/env python3
"""
Interactive Streamlit Dashboard for PM2.5 Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_folium import folium_static
import os
import sys
import joblib

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import DataLoader
from preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from visualizer import AdvancedVisualizer
from dashboard_components import DashboardComponents
import utils
from config.settings import STREAMLIT_CONFIG, MODELS_DIR

# Page configuration
st.set_page_config(
    page_title=STREAMLIT_CONFIG['page_title'],
    page_icon=STREAMLIT_CONFIG['page_icon'],
    layout=STREAMLIT_CONFIG['layout'],
    initial_sidebar_state=STREAMLIT_CONFIG['initial_sidebar_state']
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .feature-importance-bar {
        background: linear-gradient(90deg, #4CAF50 0%, #FFC107 50%, #F44336 100%);
        height: 20px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

class PM25Dashboard:
    def __init__(self):
        self.loader = DataLoader()
        self.viz = AdvancedVisualizer()
        self.components = DashboardComponents()
        self.data = None
        self.model = None
        self.preprocessor = None
        
    def load_data(self):
        """Load and prepare data"""
        try:
            self.data = self.loader.merge_all_data()
            self.data = self.loader.validate_data(self.data)
            return True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.info("Generating sample data for demonstration...")
            self.data = utils.generate_sample_data(500)
            return False
    
    def load_model(self):
        """Load trained model"""
        try:
            trainer = ModelTrainer()
            self.preprocessor, feature_names = trainer.load_model()
            self.model = trainer
            return True
        except Exception as e:
            st.warning(f"Could not load trained model: {e}")
            return False
    
    def run(self):
        """Run the dashboard"""
        # Header
        st.markdown('<h1 class="main-header">🌍 Addis Ababa PM2.5 Analysis Dashboard</h1>', 
                   unsafe_allow_html=True)
        st.markdown("""
        **Real-time air pollution monitoring and prediction using satellite data and machine learning**
        """)
        
        # Load data
        if self.data is None:
            with st.spinner("Loading data..."):
                self.load_data()
        
        # Load model
        if self.model is None:
            with st.spinner("Loading model..."):
                self.load_model()
        
        # Sidebar
        st.sidebar.title("🔧 Navigation")
        app_mode = st.sidebar.selectbox(
            "Choose Analysis Mode",
            [
                "📊 Overview Dashboard",
                "📈 Time Series Analysis", 
                "🌤️ Seasonal Patterns",
                "🔍 Correlation Analysis",
                "🤖 Model Insights",
                "🗺️ Spatial Analysis",
                "🔮 Predictions",
                "📋 Data Explorer"
            ]
        )
        
        # Main content based on selection
        if app_mode == "📊 Overview Dashboard":
            self.show_overview()
        elif app_mode == "📈 Time Series Analysis":
            self.show_time_series()
        elif app_mode == "🌤️ Seasonal Patterns":
            self.show_seasonal_patterns()
        elif app_mode == "🔍 Correlation Analysis":
            self.show_correlation_analysis()
        elif app_mode == "🤖 Model Insights":
            self.show_model_insights()
        elif app_mode == "🗺️ Spatial Analysis":
            self.show_spatial_analysis()
        elif app_mode == "🔮 Predictions":
            self.show_predictions()
        elif app_mode == "📋 Data Explorer":
            self.show_data_explorer()
    
    def show_overview(self):
        """Show overview dashboard"""
        st.header("📊 Overview Dashboard")
        
        # Metric cards
        self.components.create_metric_cards(self.data)
        
        # Quick insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Recent Trends")
            recent_data = self.data.tail(30)
            fig = px.line(recent_data, x='date', y='pm25',
                         title="Last 30 Days PM2.5 Trend")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🌡️ Current Conditions")
            if len(self.data) > 0:
                latest = self.data.iloc[-1]
                aqi, color = utils.calculate_air_quality_index(latest['pm25'])
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Current Air Quality: <span style="color:{color}">{aqi}</span></h3>
                    <p>PM2.5: {latest['pm25']:.1f} μg/m³</p>
                    <p>Temperature: {latest.get('temperature', 'N/A'):.1f}°C</p>
                    <p>Wind Speed: {latest.get('wind_speed', 'N/A'):.1f} m/s</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Data summary
        self.components.create_data_summary(self.data)
    
    def show_time_series(self):
        """Show time series analysis"""
        st.header("📈 Time Series Analysis")
        
        # Date filter
        filtered_data = self.components.create_date_filter(self.data, "timeseries")
        
        # Feature selector
        available_features = [col for col in filtered_data.columns if col != 'date']
        selected_features = self.components.create_feature_selector(available_features)
        
        if len(selected_features) > 0:
            # Create time series plot
            fig = self.viz.create_timeseries_plot(
                filtered_data[['date'] + selected_features],
                title=f"Time Series Analysis: {', '.join(selected_features)}"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Additional analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Distribution Analysis")
            selected_var = st.selectbox("Select variable:", options=selected_features)
            fig = px.histogram(filtered_data, x=selected_var, 
                             title=f"Distribution of {selected_var}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📅 Monthly Aggregates")
            monthly_avg = filtered_data.groupby(filtered_data['date'].dt.to_period('M')).mean(numeric_only=True)
            monthly_avg = monthly_avg.reset_index()
            monthly_avg['date'] = monthly_avg['date'].astype(str)
            
            fig = px.bar(monthly_avg, x='date', y='pm25',
                        title="Monthly Average PM2.5")
            st.plotly_chart(fig, use_container_width=True)
    
    def show_seasonal_patterns(self):
        """Show seasonal patterns"""
        st.header("🌤️ Seasonal Patterns")
        
        # Seasonal analysis
        seasonal_fig = self.viz.create_seasonal_analysis(self.data)
        st.plotly_chart(seasonal_fig, use_container_width=True)
        
        # Interactive seasonal analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Seasonal Averages")
            season_data = self.data.copy()
            season_data['season'] = season_data['date'].dt.month % 12 // 3 + 1
            season_names = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Autumn'}
            season_data['season_name'] = season_data['season'].map(season_names)
            
            seasonal_avg = season_data.groupby('season_name').agg({
                'pm25': 'mean',
                'temperature': 'mean',
                'aai': 'mean'
            }).reset_index()
            
            fig = px.bar(seasonal_avg, x='season_name', y='pm25',
                        title="Seasonal PM2.5 Averages")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🌡️ Temperature vs PM2.5")
            fig = px.scatter(self.data, x='temperature', y='pm25',
                           trendline="ols",
                           title="Temperature vs PM2.5 Correlation")
            st.plotly_chart(fig, use_container_width=True)
    
    def show_correlation_analysis(self):
        """Show correlation analysis"""
        st.header("🔍 Correlation Analysis")
        
        # Correlation heatmap
        corr_fig = self.viz.create_correlation_heatmap(self.data)
        st.plotly_chart(corr_fig, use_container_width=True)
        
        # Feature relationships
        st.subheader("📊 Feature Relationships")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_feature = st.selectbox("X-axis feature:", 
                                   options=[col for col in self.data.columns if col != 'date'],
                                   index=1)
        
        with col2:
            y_feature = st.selectbox("Y-axis feature:",
                                   options=[col for col in self.data.columns if col != 'date'],
                                   index=0)
        
        if x_feature and y_feature:
            fig = px.scatter(self.data, x=x_feature, y=y_feature,
                           color='pm25', size='pm25',
                           hover_data=['date'],
                           title=f"{x_feature} vs {y_feature}",
                           trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
    
    def show_model_insights(self):
        """Show model insights"""
        st.header("🤖 Model Insights")
        
        if self.model is None:
            st.warning("No trained model available. Please train a model first.")
            return
        
        # Feature importance
        st.subheader("🎯 Feature Importance")
        if self.model.feature_importance is not None:
            importance_fig = self.viz.create_feature_importance_plot(
                self.model.feature_importance
            )
            st.plotly_chart(importance_fig, use_container_width=True)
            
            # Interactive feature importance table
            st.dataframe(self.model.feature_importance, use_container_width=True)
        
        # Model performance
        st.subheader("📊 Model Performance")
        if hasattr(self.model, 'cv_scores'):
            comparison_fig = self.viz.create_model_comparison(self.model.cv_scores)
            st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Model details
        st.subheader("🔧 Model Details")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Best Model", self.model.best_model_name.upper())
        
        with col2:
            if hasattr(self.model, 'cv_scores') and self.model.best_model_name in self.model.cv_scores:
                st.metric("RMSE", f"{self.model.cv_scores[self.model.best_model_name]['rmse']:.2f}")
        
        with col3:
            if hasattr(self.model, 'cv_scores') and self.model.best_model_name in self.model.cv_scores:
                st.metric("R² Score", f"{self.model.cv_scores[self.model.best_model_name]['r2']:.2f}")
    
    def show_spatial_analysis(self):
        """Show spatial analysis"""
        st.header("🗺️ Spatial Analysis")
        
        if 'latitude' not in self.data.columns or 'longitude' not in self.data.columns:
            st.warning("Spatial data not available. Latitude and longitude columns are required.")
            return
        
        # Create folium map
        st.subheader("📍 PM2.5 Spatial Distribution")
        
        # Value selector for map
        value_column = st.selectbox(
            "Select value to display on map:",
            options=['pm25', 'aai', 'no2', 'temperature'],
            index=0
        )
        
        # Create map
        map_fig = self.viz.create_spatial_map(self.data, value_column)
        
        # Display map
        folium_static(map_fig, width=800, height=600)
        
        # Spatial statistics
        st.subheader("📊 Spatial Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_value = self.data[value_column].mean()
            st.metric(f"Average {value_column}", f"{avg_value:.2f}")
        
        with col2:
            std_value = self.data[value_column].std()
            st.metric(f"Std Dev {value_column}", f"{std_value:.2f}")
        
        with col3:
            max_value = self.data[value_column].max()
            st.metric(f"Max {value_column}", f"{max_value:.2f}")
    
    def show_predictions(self):
        """Show prediction interface"""
        st.header("🔮 PM2.5 Predictions")
        
        if self.model is None:
            st.warning("No trained model available. Please train a model first.")
            return
        
        st.subheader("🎯 Make Predictions")
        
        # Input features for prediction
        col1, col2, col3 = st.columns(3)
        
        with col1:
            aai = st.slider("AAI (Absorbing Aerosol Index)", -2.0, 5.0, 1.2, 0.1)
            no2 = st.slider("NO₂ (mol/m²)", 0.0000, 0.0010, 0.0002, 0.0001)
            temperature = st.slider("Temperature (°C)", 0.0, 40.0, 20.0, 0.5)
        
        with col2:
            wind_speed = st.slider("Wind Speed (m/s)", 0.0, 20.0, 3.0, 0.1)
            humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0, 1.0)
            precipitation = st.slider("Precipitation (mm)", 0.0, 50.0, 0.5, 0.1)
        
        with col3:
            road_density = st.slider("Road Density (km/km²)", 0.0, 50.0, 10.0, 0.5)
            urban_cover = st.slider("Urban Cover (%)", 0.0, 100.0, 50.0, 1.0)
            month = st.slider("Month", 1, 12, 6)
        
        # Create input data
        input_data = pd.DataFrame({
            'aai': [aai],
            'no2': [no2],
            'temperature': [temperature],
            'wind_speed': [wind_speed],
            'humidity': [humidity],
            'precipitation': [precipitation],
            'road_density_km_km2': [road_density],
            'urban_cover_percent': [urban_cover],
            'month': [month]
        })
        
        # Add temporal features
        input_data['month_sin'] = np.sin(2 * np.pi * input_data['month']/12)
        input_data['month_cos'] = np.cos(2 * np.pi * input_data['month']/12)
        
        if st.button("Predict PM2.5"):
            try:
                # Make prediction
                prediction = self.model.predict(input_data, self.preprocessor)[0]
                
                # Display result
                aqi, color = utils.calculate_air_quality_index(prediction)
                
                st.success(f"### Predicted PM2.5: {prediction:.1f} μg/m³")
                st.markdown(f"### Air Quality: <span style='color:{color}'>{aqi}</span>", 
                           unsafe_allow_html=True)
                
                # Prediction intervals
                lower, upper = utils.create_prediction_intervals([prediction])
                st.info(f"95% Prediction Interval: {lower[0]:.1f} - {upper[0]:.1f} μg/m³")
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    
    def show_data_explorer(self):
        """Show data explorer"""
        st.header("📋 Data Explorer")
        
        # Data table
        st.subheader("📊 Raw Data")
        
        # Date filter for data explorer
        filtered_data = self.components.create_date_filter(self.data, "explorer")
        
        # Show data table
        st.dataframe(filtered_data, use_container_width=True)
        
        # Data download
        st.subheader("💾 Data Export")
        self.components.create_download_button(filtered_data)
        
        # Data statistics
        st.subheader("📈 Data Statistics")
        st.dataframe(filtered_data.describe(), use_container_width=True)
        
        # Missing values analysis
        st.subheader("🔍 Missing Values Analysis")
        missing_data = filtered_data.isnull().sum()
        missing_percent = (missing_data / len(filtered_data)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing Percentage': missing_percent
        })
        st.dataframe(missing_df[missing_df['Missing Count'] > 0])

# Run the dashboard
if __name__ == "__main__":
    dashboard = PM25Dashboard()
    dashboard.run()