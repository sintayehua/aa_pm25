import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import folium
from streamlit_folium import folium_static
import base64
from io import BytesIO

class DashboardComponents:
    def __init__(self):
        self.colors = px.colors.qualitative.Set3
    
    def create_metric_cards(self, df, predictions=None):
        """Create metric cards for dashboard"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_pm25 = df['pm25'].iloc[-1] if len(df) > 0 else 0
            st.metric(
                "Current PM2.5", 
                f"{current_pm25:.1f} μg/m³",
                delta=f"{(current_pm25 - 15):.1f} vs WHO" if current_pm25 > 15 else "Safe"
            )
        
        with col2:
            avg_pm25 = df['pm25'].mean()
            st.metric(
                "Average PM2.5", 
                f"{avg_pm25:.1f} μg/m³",
                delta=f"{(avg_pm25 - 15):.1f} vs WHO" if avg_pm25 > 15 else "Within guideline"
            )
        
        with col3:
            exceedance_days = (df['pm25'] > 15).sum()
            exceedance_percent = (df['pm25'] > 15).mean() * 100
            st.metric(
                "WHO Guideline Exceedance", 
                f"{exceedance_days} days",
                f"{exceedance_percent:.1f}%"
            )
        
        with col4:
            if predictions is not None and 'rmse' in predictions:
                st.metric(
                    "Model RMSE", 
                    f"{predictions['rmse']:.2f} μg/m³",
                    "Prediction Error"
                )
            else:
                max_pm25 = df['pm25'].max()
                st.metric(
                    "Maximum PM2.5", 
                    f"{max_pm25:.1f} μg/m³",
                    "Peak concentration"
                )
    
    def create_date_filter(self, df, key_suffix=""):
        """Create interactive date filter"""
        col1, col2 = st.columns(2)
        
        with col1:
            min_date = df['date'].min().date()
            max_date = df['date'].max().date()
            
            start_date = st.date_input(
                "Start Date",
                min_date,
                min_value=min_date,
                max_value=max_date,
                key=f"start_{key_suffix}"
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                max_date,
                min_value=min_date,
                max_value=max_date,
                key=f"end_{key_suffix}"
            )
        
        # Convert to datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        st.info(f"Showing {len(filtered_df)} records from {start_date.date()} to {end_date.date()}")
        
        return filtered_df
    
    def create_feature_selector(self, available_features, default_features=None):
        """Create interactive feature selector"""
        if default_features is None:
            default_features = ['pm25', 'aai', 'no2', 'temperature']
        
        selected_features = st.multiselect(
            "Select features to display:",
            options=available_features,
            default=default_features,
            help="Choose which variables to include in visualizations"
        )
        
        return selected_features
    
    def create_model_controls(self):
        """Create model training and prediction controls"""
        st.sidebar.header("Model Controls")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            retrain_model = st.button("🔄 Retrain Model")
        
        with col2:
            predict_new = st.button("🔮 Make Predictions")
        
        st.sidebar.subheader("Model Parameters")
        n_estimators = st.sidebar.slider("Number of estimators", 50, 500, 100, 50)
        max_depth = st.sidebar.slider("Max depth", 3, 20, 10)
        test_size = st.sidebar.slider("Test size (%)", 10, 40, 20) / 100
        
        return {
            'retrain': retrain_model,
            'predict': predict_new,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'test_size': test_size
        }
    
    def create_download_button(self, df, filename="pm25_analysis_data.csv"):
        """Create download button for data"""
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download Data as CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    def create_plot_controls(self):
        """Create controls for plot customization"""
        st.sidebar.header("Visualization Controls")
        
        plot_type = st.sidebar.selectbox(
            "Plot Type",
            ["Line Plot", "Scatter Plot", "Bar Chart", "Heatmap"]
        )
        
        color_scheme = st.sidebar.selectbox(
            "Color Scheme",
            ["Plotly", "Viridis", "Plasma", "Inferno", "Rainbow"]
        )
        
        show_trendline = st.sidebar.checkbox("Show Trendline", value=True)
        smooth_data = st.sidebar.checkbox("Smooth Data", value=False)
        
        return {
            'plot_type': plot_type,
            'color_scheme': color_scheme,
            'show_trendline': show_trendline,
            'smooth_data': smooth_data
        }
    
    def create_analysis_selector(self):
        """Create selector for different analysis types"""
        analysis_type = st.selectbox(
            "Select Analysis Type:",
            [
                "Time Series Analysis",
                "Seasonal Patterns", 
                "Correlation Analysis",
                "Feature Importance",
                "Model Performance",
                "Spatial Distribution",
                "Prediction Analysis"
            ]
        )
        
        return analysis_type
    
    def create_data_summary(self, df):
        """Create interactive data summary"""
        with st.expander("📊 Data Summary", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Temporal Range**")
                st.write(f"Start: {df['date'].min().date()}")
                st.write(f"End: {df['date'].max().date()}")
                st.write(f"Duration: {(df['date'].max() - df['date'].min()).days} days")
            
            with col2:
                st.write("**PM2.5 Statistics**")
                st.write(f"Mean: {df['pm25'].mean():.1f} μg/m³")
                st.write(f"Std: {df['pm25'].std():.1f} μg/m³")
                st.write(f"Min: {df['pm25'].min():.1f} μg/m³")
                st.write(f"Max: {df['pm25'].max():.1f} μg/m³")
            
            with col3:
                st.write("**Data Quality**")
                total_records = len(df)
                complete_records = df.notna().all(axis=1).sum()
                st.write(f"Total Records: {total_records}")
                st.write(f"Complete Records: {complete_records}")
                st.write(f"Completeness: {complete_records/total_records*100:.1f}%")
            
            # Show raw data preview
            if st.checkbox("Show raw data preview"):
                st.dataframe(df.head(10))