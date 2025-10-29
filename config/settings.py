import os
from datetime import datetime

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')
DATA_OUTPUTS = os.path.join(PROJECT_ROOT, 'data', 'outputs')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models', 'trained_models')

# Create directories
for directory in [DATA_RAW, DATA_PROCESSED, DATA_OUTPUTS, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Model parameters
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'n_iter': 50
}

# Visualization settings
PLOTLY_CONFIG = {
    'displayModeBar': True,
    'scrollZoom': True,
    'displaylogo': False
}

# Streamlit settings
STREAMLIT_CONFIG = {
    'page_title': 'Addis Ababa PM2.5 Analysis Dashboard',
    'page_icon': '🌍',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Color schemes
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'tertiary': '#2ca02c',
    'quaternary': '#d62728',
    'background': '#f0f2f6',
    'text': '#262730'
}