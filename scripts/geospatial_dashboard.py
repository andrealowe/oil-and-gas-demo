"""
Comprehensive Geospatial Dashboard for Oil & Gas Operations
Professional Aramco-inspired design with interactive facility monitoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/mnt/code')
from scripts.data_config import get_data_paths

def format_large_number(value):
    """Format large numbers with appropriate suffixes"""
    if value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value/1_000:.1f}K"
    else:
        return f"{value:.1f}"

# Page configuration
st.set_page_config(
    page_title="Oil & Gas Geospatial Operations Dashboard",
    page_icon="/mnt/code/docs/domino_logo.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force light theme by setting theme in session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Aramco-inspired custom CSS with Light Mode Override
st.markdown("""
<style>
    /* Force light theme override - prevents dark mode */
    .stApp {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    .stApp > .main {
        background-color: #ffffff !important;
    }
    
    .stSidebar {
        background-color: #f8f9fa !important;
    }
    
    .stSidebar .stMarkdown {
        color: #000000 !important;
    }
    
    /* Override any dark mode text and elements */
    .stMarkdown, .stText, p, span, div, h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }
    
    /* Force light mode for specific Streamlit components */
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    .stTextInput > div > div > input {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    .stDateInput > div > div > input {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Override media query that detects dark mode */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        
        .stApp > .main {
            background-color: #ffffff !important;
        }
    }
    
    /* Professional Light Mode color scheme */
    :root {
        --primary-blue: #2563eb;        /* Professional blue */
        --secondary-blue: #60a5fa;      /* Light blue */
        --accent-blue: #1d4ed8;         /* Darker blue for accents */
        --light-blue: #dbeafe;          /* Very light blue */
        --success-green: #059669;       /* Professional green */
        --warning-amber: #d97706;       /* Professional amber */
        --error-red: #dc2626;           /* Professional red */
        --gray-50: #ffffff;             /* Pure white backgrounds */
        --gray-100: #f8fafc;           /* Very light gray */
        --gray-200: #e5e7eb;           /* Light gray borders */
        --gray-600: #6b7280;           /* Medium gray text - improved contrast */
        --gray-700: #374151;           /* Better contrast for text */
        --gray-800: #1f2937;           /* Dark gray (for text) */
        --gray-900: #111827;           /* Very dark (for text) - improved contrast */
        --text-primary: #111827;        /* Primary text color - better contrast */
        --text-secondary: #374151;      /* Secondary text color - better contrast */
    }
    
    /* Main background and container styling */
    .main > div {
        padding: 2rem 1rem;
        background: linear-gradient(135deg, #f8fafb 0%, #e9ecef 100%);
        min-height: 100vh;
    }
    
    /* Professional Enterprise Header */
    .enterprise-header {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 50%, #60a5fa 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 16px -4px rgba(37, 99, 235, 0.25), 0 2px 6px -1px rgba(0, 0, 0, 0.08);
        position: relative;
        overflow: hidden;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .enterprise-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #fbbf24, #f59e0b, #d97706);
    }
    
    .header-content {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    }
    
    .header-left h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        color: white;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .header-subtitle {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 400;
        letter-spacing: 0.025em;
    }
    
    /* Blue box text styling - make text white */
    .enterprise-header .header-left h1,
    .enterprise-header .header-subtitle {
        color: white !important;
    }
    
    /* Professional forecast card styling */
    .metric-card, .forecast-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 2rem 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.08), 0 2px 6px -1px rgba(0, 0, 0, 0.04);
        border: 1px solid #e5e7eb;
        border-left: 4px solid var(--primary-blue);
        margin-bottom: 1.5rem;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover, .forecast-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px -4px rgba(0, 0, 0, 0.12), 0 4px 8px -2px rgba(0, 0, 0, 0.08);
        border-left-color: #1d4ed8;
    }
    
    .metric-value, .forecast-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--gray-900);
        word-break: break-word;
        line-height: 1.2;
        margin: 0.75rem 0;
    }
    
    .metric-label, .forecast-label {
        font-size: 0.95rem;
        color: var(--gray-600);
        margin-bottom: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        line-height: 1.3;
    }
    
    .metric-delta, .forecast-change {
        font-size: 0.85rem;
        margin-top: 0.75rem;
        color: var(--gray-600);
        font-weight: 500;
        line-height: 1.4;
    }
    
    /* Status indicators - professional MLOps colors */
    .status-excellent, .forecast-up { color: var(--success-green); }
    .status-good, .forecast-neutral { color: var(--warning-amber); }
    .status-warning, .forecast-down { color: var(--error-red); }
    .status-critical { color: var(--error-red); }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--aramco-light-gray);
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    /* Filter section */
    .filter-section {
        background: var(--aramco-light-gray);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    /* Loading spinner */
    .loading-spinner {
        text-align: center;
        color: var(--aramco-blue);
        font-size: 1.2rem;
        padding: 2rem;
    }
    
    /* Enhanced Buttons - More prominent styling with white text */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--accent-blue) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.025em !important;
        box-shadow: 0 4px 12px -2px rgba(37, 99, 235, 0.4) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        min-height: 48px !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--accent-blue) 0%, #1e40af 100%) !important;
        border: none !important;
        color: white !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px -4px rgba(37, 99, 235, 0.5) !important;
    }
    
    .stButton > button:active {
        color: white !important;
        transform: translateY(0px) !important;
        box-shadow: 0 2px 8px -2px rgba(37, 99, 235, 0.4) !important;
    }
    
    /* Primary button variant */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--accent-blue) 100%) !important;
        color: white !important;
        box-shadow: 0 4px 12px -2px rgba(37, 99, 235, 0.5) !important;
        font-weight: 700 !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, var(--accent-blue) 0%, #1e40af 100%) !important;
        color: white !important;
        box-shadow: 0 6px 16px -4px rgba(37, 99, 235, 0.6) !important;
    }
    
    .stButton > button[kind="primary"]:active {
        color: white !important;
    }
    
    /* Additional specificity for button text */
    .stButton button span,
    .stButton button div,
    .stButton button * {
        color: white !important;
    }
    
    /* Selectboxes and other inputs */
    .stSelectbox > div > div {
        background-color: white !important;
        border: 1px solid var(--gray-200) !important;
        border-radius: 6px !important;
    }
    
    .stSelectbox > div > div > div {
        color: var(--text-primary) !important;
    }
    
    /* Date inputs */
    .stDateInput > div > div > input {
        background-color: white !important;
        border: 1px solid var(--gray-200) !important;
        color: var(--text-primary) !important;
    }
    
    /* Number inputs */
    .stNumberInput > div > div > input {
        background-color: white !important;
        border: 1px solid var(--gray-200) !important;
        color: var(--text-primary) !important;
    }
    
    /* Multiselect */
    .stMultiSelect > div > div {
        background-color: white !important;
        border: 1px solid var(--gray-200) !important;
    }
    
    /* Multiselect selected items - make text readable */
    .stMultiSelect div[data-baseweb="tag"] {
        background-color: var(--primary-blue) !important;
        color: white !important;
    }
    
    .stMultiSelect div[data-baseweb="tag"] span {
        color: white !important;
    }
    
    /* Text inputs */
    .stTextInput > div > div > input {
        background-color: white !important;
        border: 1px solid var(--gray-200) !important;
        color: var(--text-primary) !important;
    }
    
    /* Metrics */
    .stMetric {
        background-color: white !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        border: 1px solid var(--gray-200) !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    }
    
    .stMetric > div {
        color: var(--text-primary) !important;
    }
    
    .stMetric [data-testid="metric-container"] > div {
        color: var(--text-primary) !important;
    }
    
    p, span, div, label {
        color: var(--text-primary) !important;
    }
    
    .stPlotlyChart {
        background-color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Data loading functions
@st.cache_data(ttl=600, show_spinner=True)
def load_geospatial_data():
    """Load geospatial facilities data with caching and optimization"""
    try:
        paths = get_data_paths('Oil-and-Gas-Demo')
        data_path = paths['base_data_path'] / 'geospatial_facilities.parquet'
        
        if data_path.exists():
            # Load only essential columns to reduce memory usage
            essential_columns = [
                'facility_id', 'facility_name', 'facility_type', 'region', 'status',
                'latitude', 'longitude', 'oil_production_bpd', 'gas_production_mcfd',
                'equipment_health_score', 'h2s_concentration_ppm', 'co2_concentration_ppm',
                'temperature_celsius', 'vibration_level_mm_s', 'pressure_psi', 
                'last_maintenance_date', 'next_maintenance_date', 'noise_level_db', 'utilization_rate', 'capacity_bpd'
            ]
            
            # Read all columns first to check what's available
            df_full = pd.read_parquet(data_path)
            available_columns = [col for col in essential_columns if col in df_full.columns]
            
            # Load only available essential columns
            df = df_full[available_columns].copy()
            
            # Clean up the data and handle any NaN values
            df = df.fillna({
                'oil_production_bpd': 0,
                'gas_production_mcfd': 0,
                'equipment_health_score': 0.7
            })
            
            # Rename columns to match expected names
            df = df.rename(columns={
                'h2s_concentration_ppm': 'h2s_concentration',
                'co2_concentration_ppm': 'co2_concentration'
            })
            
            # Add missing columns with default values if they don't exist
            if 'temperature_celsius' not in df.columns:
                df['temperature_celsius'] = np.random.uniform(40, 60, len(df))
            if 'vibration_level_mm_s' not in df.columns:
                df['vibration_level_mm_s'] = np.random.uniform(0.1, 1.0, len(df))
            if 'pressure_psi' not in df.columns:
                df['pressure_psi'] = np.random.uniform(1000, 2000, len(df))
            if 'last_maintenance_date' not in df.columns:
                # Create random dates in the last 6 months
                base_date = datetime.now() - timedelta(days=180)
                df['last_maintenance_date'] = [base_date + timedelta(days=np.random.randint(0, 180)) for _ in range(len(df))]
            if 'noise_level_db' not in df.columns:
                df['noise_level_db'] = np.random.uniform(60, 90, len(df))
            if 'utilization_rate' not in df.columns:
                df['utilization_rate'] = np.random.uniform(0.6, 0.95, len(df))
            if 'status' not in df.columns:
                df['status'] = np.random.choice(['Active', 'Maintenance', 'Standby'], len(df), p=[0.7, 0.2, 0.1])
            if 'capacity_bpd' not in df.columns:
                # Generate capacity as 110-150% of current production, with minimum of 50 bpd
                oil_prod = df['oil_production_bpd'].fillna(50)  # Fill NaN with default
                capacity_multiplier = np.random.uniform(1.1, 1.5, len(df))
                df['capacity_bpd'] = np.maximum(oil_prod * capacity_multiplier, 50)
            if 'next_maintenance_date' not in df.columns:
                # Generate next maintenance dates 30-180 days in the future
                base_date = datetime.now()
                df['next_maintenance_date'] = [base_date + timedelta(days=np.random.randint(30, 180)) for _ in range(len(df))]
            # Sample data if too large (keep first 1000 facilities)
            if len(df) > 1000:
                df = df.head(1000)
            return df
        else:
            # Create minimal sample data if file doesn't exist
            return create_sample_geospatial_data()
    except Exception as e:
        st.warning(f"Using sample data due to error: {e}")
        return create_sample_geospatial_data()

@st.cache_data(ttl=600)
def load_production_data():
    """Load production time series data with optimization"""
    try:
        paths = get_data_paths('Oil-and-Gas-Demo')
        data_path = paths['base_data_path'] / 'production_timeseries.parquet'
        
        if data_path.exists():
            # Load only recent data to reduce memory usage
            df = pd.read_parquet(data_path)
            df['date'] = pd.to_datetime(df['date'])
            # Keep only last 90 days of data
            recent_date = df['date'].max() - pd.Timedelta(days=90)
            df = df[df['date'] >= recent_date]
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.warning(f"Error loading production data: {e}")
        return pd.DataFrame()

def create_sample_geospatial_data():
    """Create sample geospatial data for demonstration"""
    np.random.seed(42)
    n_facilities = 50  # Small sample for fast loading
    
    regions = ['North America', 'Middle East', 'Europe']
    facility_types = ['Oil Well', 'Gas Well', 'Refinery', 'Storage Tank']
    
    data = {
        'facility_id': [f'FAC-{i:03d}' for i in range(1, n_facilities+1)],
        'facility_name': [f'Facility {i}' for i in range(1, n_facilities+1)],
        'facility_type': np.random.choice(facility_types, n_facilities),
        'region': np.random.choice(regions, n_facilities),
        'latitude': np.random.uniform(20, 60, n_facilities),
        'longitude': np.random.uniform(-120, 50, n_facilities),
        'oil_production_bpd': np.random.uniform(10, 200, n_facilities),
        'gas_production_mcfd': np.random.uniform(50, 500, n_facilities),
        'equipment_health_score': np.random.uniform(0.4, 0.95, n_facilities),
        'h2s_concentration': np.random.uniform(1, 15, n_facilities),
        'co2_concentration': np.random.uniform(50, 200, n_facilities),
        'temperature_celsius': np.random.uniform(40, 60, n_facilities),
        'vibration_level_mm_s': np.random.uniform(0.1, 1.0, n_facilities),
        'pressure_psi': np.random.uniform(1000, 2000, n_facilities),
        'noise_level_db': np.random.uniform(60, 90, n_facilities),
        'utilization_rate': np.random.uniform(0.6, 0.95, n_facilities),
        'status': np.random.choice(['Active', 'Maintenance', 'Standby'], n_facilities, p=[0.7, 0.2, 0.1])
    }
    
    # Add maintenance dates
    base_date = datetime.now() - timedelta(days=180)
    data['last_maintenance_date'] = [base_date + timedelta(days=np.random.randint(0, 180)) for _ in range(n_facilities)]
    
    # Add capacity as 110-150% of production
    data['capacity_bpd'] = np.array(data['oil_production_bpd']) * np.random.uniform(1.1, 1.5, n_facilities)
    
    # Add next maintenance dates (30-180 days in future)
    future_base = datetime.now()
    data['next_maintenance_date'] = [future_base + timedelta(days=np.random.randint(30, 180)) for _ in range(n_facilities)]
    
    return pd.DataFrame(data)

def call_geospatial_api(facility_data):
    """Call the geospatial prediction API"""
    try:
        # Simulate API call (replace with actual endpoint when available)
        api_url = "http://localhost:8000/predict"  # Update with actual API URL
        
        # For demonstration, create mock predictions based on facility data
        mock_prediction = {
            "facility_id": facility_data.get('facility_id', 'unknown'),
            "predictions": {
                "equipment_health": {
                    "score": np.random.uniform(0.4, 0.95),
                    "status": np.random.choice(["Excellent", "Good", "Fair", "Poor"], p=[0.3, 0.4, 0.2, 0.1]),
                    "confidence": np.random.uniform(0.7, 0.95)
                },
                "production_efficiency": {
                    "class": np.random.choice(["High", "Medium", "Low"], p=[0.4, 0.4, 0.2]),
                    "probability": np.random.uniform(0.6, 0.9),
                    "confidence": np.random.uniform(0.7, 0.9)
                },
                "environmental_risk": {
                    "level": np.random.choice(["Low", "Medium", "High"], p=[0.6, 0.3, 0.1]),
                    "probability": np.random.uniform(0.6, 0.9),
                    "confidence": np.random.uniform(0.7, 0.9)
                }
            }
        }
        return mock_prediction
    except Exception as e:
        st.warning(f"API call failed, using fallback prediction: {e}")
        return {"error": str(e)}

def create_facility_map(df):
    """Create interactive facility location map"""
    if df.empty:
        return go.Figure()
    
    # Create color mapping based on equipment health
    color_map = {
        'Excellent': '#84BD00',  # Aramco green
        'Good': '#FFA500',       # Orange
        'Fair': '#FF6B6B',       # Light red
        'Poor': '#DC143C'        # Dark red
    }
    
    # Determine health status based on equipment_health_score
    df['health_status'] = df['equipment_health_score'].apply(
        lambda x: 'Excellent' if x >= 0.8 else 'Good' if x >= 0.6 else 'Fair' if x >= 0.4 else 'Poor'
    )
    
    # Create map figure
    fig = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        color="health_status",
        color_discrete_map=color_map,
        size="oil_production_bpd",
        hover_name="facility_name",
        hover_data={
            "facility_type": True,
            "region": True,
            "oil_production_bpd": ":.0f",
            "gas_production_mcfd": ":.0f",
            "equipment_health_score": ":.2f",
            "utilization_rate": ":.1%"
        },
        title="Global Facility Locations and Health Status",
        zoom=2,
        height=500
    )
    
    fig.update_layout(
        mapbox_style="open-street-map",
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", size=12),
        title=dict(
            font=dict(size=16, color='#111827'),
            x=0.5
        ),
        coloraxis_colorbar=dict(
            title=dict(
                text="Equipment Health",
                font=dict(size=12)
            )
        )
    )
    
    return fig

def create_production_efficiency_chart(df):
    """Create production efficiency analysis chart"""
    if df.empty:
        return go.Figure()
    
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Calculate efficiency metrics with robust handling
    capacity = df['capacity_bpd'].fillna(df['oil_production_bpd'] * 1.2)
    capacity = capacity.fillna(100)  # Default capacity if still NaN
    df['production_efficiency'] = df['oil_production_bpd'] / capacity
    df['production_efficiency'] = df['production_efficiency'].fillna(0.7)  # Default efficiency
    
    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Production Efficiency by Region",
            "Equipment Health Distribution",
            "Utilization Rate Analysis",
            "Environmental Risk Factors"
        ),
        specs=[
            [{"secondary_y": True}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ]
    )
    
    # Production efficiency by region
    region_efficiency = df.groupby('region').agg({
        'production_efficiency': 'mean',
        'oil_production_bpd': 'sum',
        'facility_id': 'count'
    }).reset_index()
    
    fig.add_trace(
        go.Bar(
            x=region_efficiency['region'],
            y=region_efficiency['production_efficiency'],
            name="Avg Efficiency",
            marker_color='#00A3E0'
        ),
        row=1, col=1
    )
    
    # Equipment health distribution
    health_dist = df['equipment_health_score'].value_counts(bins=10)
    fig.add_trace(
        go.Histogram(
            x=df['equipment_health_score'],
            nbinsx=20,
            name="Health Distribution",
            marker_color='#84BD00'
        ),
        row=1, col=2
    )
    
    # Utilization rate vs production
    fig.add_trace(
        go.Scatter(
            x=df['utilization_rate'],
            y=df['oil_production_bpd'],
            mode='markers',
            name="Utilization vs Production",
            marker=dict(
                size=8,
                color=df['equipment_health_score'],
                colorscale='Viridis',
                showscale=True
            ),
            text=df['facility_name']
        ),
        row=2, col=1
    )
    
    # Environmental risk factors
    h2s_col = 'h2s_concentration' if 'h2s_concentration' in df.columns else 'h2s_concentration_ppm'
    co2_col = 'co2_concentration' if 'co2_concentration' in df.columns else 'co2_concentration_ppm'
    noise_col = 'noise_level_db' if 'noise_level_db' in df.columns else None
    
    # Use available environmental columns
    if h2s_col in df.columns and co2_col in df.columns:
        marker_size = df[noise_col]/5 if noise_col and noise_col in df.columns else 10
        fig.add_trace(
            go.Scatter(
                x=df[h2s_col],
                y=df[co2_col],
                mode='markers',
                name="Environmental Factors",
                marker=dict(
                    size=marker_size,
                    color='#FF6B6B',
                    opacity=0.6
                ),
                text=df['facility_name']
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Production Efficiency and Environmental Analysis",
        title_x=0.5,
        title=dict(font=dict(size=16, color='#111827'), x=0.5)
    )
    
    return fig

def create_equipment_health_dashboard(df):
    """Create equipment health monitoring dashboard"""
    if df.empty:
        return go.Figure()
    
    # Create equipment health metrics
    health_metrics = df.groupby('facility_type').agg({
        'equipment_health_score': ['mean', 'std', 'count'],
        'vibration_level_mm_s': 'mean',
        'temperature_celsius': 'mean',
        'pressure_psi': 'mean'
    }).round(2)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Health Score by Facility Type",
            "Temperature vs Vibration Analysis",
            "Pressure Distribution",
            "Maintenance Schedule Overview"
        )
    )
    
    # Health score by facility type
    facility_types = health_metrics.index
    health_scores = health_metrics[('equipment_health_score', 'mean')]
    
    fig.add_trace(
        go.Bar(
            x=facility_types,
            y=health_scores,
            name="Avg Health Score",
            marker_color=['#84BD00' if score > 0.7 else '#FFA500' if score > 0.5 else '#FF6B6B' for score in health_scores]
        ),
        row=1, col=1
    )
    
    # Temperature vs vibration
    fig.add_trace(
        go.Scatter(
            x=df['temperature_celsius'],
            y=df['vibration_level_mm_s'],
            mode='markers',
            name="Temp vs Vibration",
            marker=dict(
                size=df['equipment_health_score']*20,
                color=df['equipment_health_score'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Health Score")
            ),
            text=df['facility_name']
        ),
        row=1, col=2
    )
    
    # Pressure distribution
    fig.add_trace(
        go.Histogram(
            x=df['pressure_psi'],
            nbinsx=30,
            name="Pressure Distribution",
            marker_color='#00A3E0',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Maintenance schedule with robust date handling
    df = df.copy()  # Avoid SettingWithCopyWarning
    
    # Handle last maintenance date
    if 'last_maintenance_date' in df.columns:
        df['days_since_maintenance'] = (datetime.now() - pd.to_datetime(df['last_maintenance_date'])).dt.days
    else:
        df['days_since_maintenance'] = np.random.randint(30, 180, len(df))
    
    # Handle next maintenance date
    if 'next_maintenance_date' in df.columns:
        df['days_to_maintenance'] = (pd.to_datetime(df['next_maintenance_date']) - datetime.now()).dt.days
    else:
        df['days_to_maintenance'] = np.random.randint(30, 180, len(df))
    
    fig.add_trace(
        go.Scatter(
            x=df['days_since_maintenance'],
            y=df['days_to_maintenance'],
            mode='markers',
            name="Maintenance Schedule",
            marker=dict(
                size=10,
                color=['#FF6B6B' if days < 0 else '#84BD00' if days > 30 else '#FFA500' for days in df['days_to_maintenance']],
                opacity=0.7
            ),
            text=df['facility_name']
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Equipment Health Monitoring Dashboard",
        title_x=0.5,
        title=dict(font=dict(size=16, color='#111827'), x=0.5)
    )
    
    return fig

def display_kpi_metrics(df):
    """Display key performance indicators"""
    if df.empty:
        st.warning("No data available for metrics display")
        return
    
    # Calculate KPIs
    total_facilities = len(df)
    total_oil_production = df['oil_production_bpd'].sum()
    total_gas_production = df['gas_production_mcfd'].sum()
    avg_health_score = df['equipment_health_score'].mean()
    avg_utilization = df['utilization_rate'].mean()
    
    # Health status distribution
    health_excellent = len(df[df['equipment_health_score'] >= 0.8])
    health_good = len(df[(df['equipment_health_score'] >= 0.6) & (df['equipment_health_score'] < 0.8)])
    health_fair = len(df[(df['equipment_health_score'] >= 0.4) & (df['equipment_health_score'] < 0.6)])
    health_poor = len(df[df['equipment_health_score'] < 0.4])
    
    # Display metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total Facilities</div>
            <div class="metric-value">{:,}</div>
            <div class="metric-delta">Active Operations</div>
        </div>
        """.format(total_facilities), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Oil Production</div>
            <div class="metric-value">{}</div>
            <div class="metric-delta">BPD Total</div>
        </div>
        """.format(format_large_number(total_oil_production)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Gas Production</div>
            <div class="metric-value">{}</div>
            <div class="metric-delta">MCFD Total</div>
        </div>
        """.format(format_large_number(total_gas_production)), unsafe_allow_html=True)
    
    with col4:
        health_color = "status-excellent" if avg_health_score >= 0.8 else "status-good" if avg_health_score >= 0.6 else "status-warning"
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Avg Health Score</div>
            <div class="metric-value {}">{:.1%}</div>
            <div class="metric-delta">Equipment Health</div>
        </div>
        """.format(health_color, avg_health_score), unsafe_allow_html=True)
    
    with col5:
        util_color = "status-excellent" if avg_utilization >= 0.8 else "status-good" if avg_utilization >= 0.6 else "status-warning"
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Avg Utilization</div>
            <div class="metric-value {}">{:.1%}</div>
            <div class="metric-delta">Capacity Usage</div>
        </div>
        """.format(util_color, avg_utilization), unsafe_allow_html=True)
    
    # Health status breakdown
    st.markdown("### Equipment Health Status Distribution")
    health_col1, health_col2, health_col3, health_col4 = st.columns(4)
    
    with health_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Excellent Health</div>
            <div class="metric-value status-excellent">{health_excellent}</div>
            <div class="metric-delta">{health_excellent/total_facilities:.1%} of facilities</div>
        </div>
        """, unsafe_allow_html=True)
    
    with health_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Good Health</div>
            <div class="metric-value status-good">{health_good}</div>
            <div class="metric-delta">{health_good/total_facilities:.1%} of facilities</div>
        </div>
        """, unsafe_allow_html=True)
    
    with health_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Fair Health</div>
            <div class="metric-value status-warning">{health_fair}</div>
            <div class="metric-delta">{health_fair/total_facilities:.1%} of facilities</div>
        </div>
        """, unsafe_allow_html=True)
    
    with health_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Poor Health</div>
            <div class="metric-value status-critical">{health_poor}</div>
            <div class="metric-delta">{health_poor/total_facilities:.1%} of facilities</div>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main dashboard application"""
    
    # Professional Header
    st.markdown("""
    <div class="enterprise-header">
        <div class="header-content">
            <div class="header-left">
                <h1>Geospatial Operations Dashboard</h1>
                <div class="header-subtitle">Real-time facility monitoring, equipment health, and production optimization</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar filters
    st.sidebar.markdown("## Dashboard Filters")
    st.sidebar.markdown('<div class="filter-section">', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading geospatial data..."):
        facilities_df = load_geospatial_data()
        production_df = load_production_data()
    
    if facilities_df.empty:
        st.error("Unable to load facility data. Please check data sources.")
        return
    
    # Filters
    selected_regions = st.sidebar.multiselect(
        "Select Regions",
        options=facilities_df['region'].unique(),
        default=facilities_df['region'].unique()[:3]
    )
    
    selected_facility_types = st.sidebar.multiselect(
        "Select Facility Types",
        options=facilities_df['facility_type'].unique(),
        default=facilities_df['facility_type'].unique()
    )
    
    health_threshold = st.sidebar.slider(
        "Minimum Health Score",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Filter facilities by minimum equipment health score"
    )
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Data refresh
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Filter data
    filtered_df = facilities_df[
        (facilities_df['region'].isin(selected_regions)) &
        (facilities_df['facility_type'].isin(selected_facility_types)) &
        (facilities_df['equipment_health_score'] >= health_threshold)
    ]
    
    if filtered_df.empty:
        st.warning("No facilities match the selected filters.")
        return
    
    # Display KPI metrics
    display_kpi_metrics(filtered_df)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Facility Map", 
        "Production Analysis", 
        "Equipment Health", 
        "Facility Details",
        "AI Predictions"
    ])
    
    with tab1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        facility_map = create_facility_map(filtered_df)
        st.plotly_chart(facility_map, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Regional summary
        st.markdown("### Regional Performance Summary")
        regional_summary = filtered_df.groupby('region').agg({
            'facility_id': 'count',
            'oil_production_bpd': 'sum',
            'gas_production_mcfd': 'sum',
            'equipment_health_score': 'mean',
            'utilization_rate': 'mean'
        }).round(2)
        regional_summary.columns = ['Facilities', 'Oil Production (BPD)', 'Gas Production (MCFD)', 'Avg Health Score', 'Avg Utilization']
        st.dataframe(regional_summary, width='stretch')
    
    with tab2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        production_chart = create_production_efficiency_chart(filtered_df)
        st.plotly_chart(production_chart, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        equipment_dashboard = create_equipment_health_dashboard(filtered_df)
        st.plotly_chart(equipment_dashboard, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown("### Facility Details")
        
        # Facility selector
        facility_options = filtered_df['facility_name'].tolist()
        selected_facility = st.selectbox(
            "Select a facility for detailed view:",
            facility_options
        )
        
        if selected_facility:
            facility_data = filtered_df[filtered_df['facility_name'] == selected_facility].iloc[0]
            
            # Display facility details
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Basic Information**")
                st.write(f"**Facility ID:** {facility_data['facility_id']}")
                st.write(f"**Type:** {facility_data['facility_type']}")
                st.write(f"**Region:** {facility_data['region']}")
                # Add status if available, otherwise show health status
                if 'status' in facility_data:
                    st.write(f"**Status:** {facility_data['status']}")
                else:
                    st.write(f"**Health Status:** {facility_data.get('health_status', 'Unknown')}")
            
            with col2:
                st.markdown("**Production Metrics**")
                st.write(f"**Oil Production:** {format_large_number(facility_data['oil_production_bpd'])} BPD")
                st.write(f"**Gas Production:** {format_large_number(facility_data['gas_production_mcfd'])} MCFD")
                capacity = facility_data.get('capacity_bpd', 'N/A')
                if isinstance(capacity, (int, float)):
                    capacity_str = f"{format_large_number(capacity)} BPD"
                else:
                    capacity_str = str(capacity)
                st.write(f"**Capacity:** {capacity_str}")
                st.write(f"**Utilization Rate:** {facility_data['utilization_rate']:.1%}")
                throughput = facility_data.get('current_throughput_bpd', 'N/A')
                if isinstance(throughput, (int, float)):
                    throughput_str = f"{format_large_number(throughput)} BPD"
                else:
                    throughput_str = str(throughput)
                st.write(f"**Throughput:** {throughput_str}")
            
            with col3:
                st.markdown("**Equipment Health**")
                st.write(f"**Health Score:** {facility_data['equipment_health_score']:.1%}")
                st.write(f"**Temperature:** {facility_data['temperature_celsius']:.1f}°C")
                st.write(f"**Pressure:** {facility_data['pressure_psi']:,.0f} PSI")
                st.write(f"**Vibration:** {facility_data['vibration_level_mm_s']:.2f} mm/s")
                st.write(f"**Noise Level:** {facility_data['noise_level_db']:.1f} dB")
            
            # Environmental data
            st.markdown("**Environmental Monitoring**")
            env_col1, env_col2 = st.columns(2)
            
            with env_col1:
                st.write(f"**H2S Concentration:** {facility_data['h2s_concentration']:.1f} PPM")
                st.write(f"**CO2 Concentration:** {facility_data['co2_concentration']:.1f} PPM")
            
            with env_col2:
                st.write(f"**Energy Consumption:** {facility_data.get('energy_consumption_mwh', 'N/A')} MWh")
                st.write(f"**Water Usage:** {facility_data.get('water_usage_gallons_day', 'N/A')} Gal/Day")
    
    with tab5:
        st.markdown("### AI-Powered Facility Predictions")
        st.markdown("Get real-time predictions for equipment health, production efficiency, and environmental risk.")
        
        # Facility selector for predictions
        prediction_facility = st.selectbox(
            "Select facility for AI prediction:",
            facility_options,
            key="prediction_selector"
        )
        
        if st.button("Generate AI Predictions", type="primary"):
            if prediction_facility:
                facility_data = filtered_df[filtered_df['facility_name'] == prediction_facility].iloc[0].to_dict()
                
                with st.spinner("Generating AI predictions..."):
                    prediction_result = call_geospatial_api(facility_data)
                
                if "error" not in prediction_result:
                    predictions = prediction_result.get("predictions", {})
                    
                    # Display predictions
                    pred_col1, pred_col2, pred_col3 = st.columns(3)
                    
                    with pred_col1:
                        if "equipment_health" in predictions:
                            health = predictions["equipment_health"]
                            health_color = {
                                "Excellent": "status-excellent",
                                "Good": "status-good", 
                                "Fair": "status-warning",
                                "Poor": "status-critical"
                            }.get(health["status"], "")
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Equipment Health Prediction</div>
                                <div class="metric-value {health_color}">{health['status']}</div>
                                <div class="metric-delta">Score: {health['score']:.1%} | Confidence: {health['confidence']:.1%}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with pred_col2:
                        if "production_efficiency" in predictions:
                            efficiency = predictions["production_efficiency"]
                            eff_color = {
                                "High": "status-excellent",
                                "Medium": "status-good",
                                "Low": "status-warning"
                            }.get(efficiency["class"], "")
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Production Efficiency Prediction</div>
                                <div class="metric-value {eff_color}">{efficiency['class']}</div>
                                <div class="metric-delta">Probability: {efficiency['probability']:.1%} | Confidence: {efficiency['confidence']:.1%}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with pred_col3:
                        if "environmental_risk" in predictions:
                            risk = predictions["environmental_risk"]
                            risk_color = {
                                "Low": "status-excellent",
                                "Medium": "status-good",
                                "High": "status-critical"
                            }.get(risk["level"], "")
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Environmental Risk Prediction</div>
                                <div class="metric-value {risk_color}">{risk['level']}</div>
                                <div class="metric-delta">Probability: {risk['probability']:.1%} | Confidence: {risk['confidence']:.1%}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Show prediction details
                    st.markdown("### Prediction Details")
                    st.json(prediction_result)
                    
                else:
                    st.error(f"Prediction failed: {prediction_result.get('error', 'Unknown error')}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem; padding: 1rem;'>
        Oil & Gas Geospatial Dashboard | Last Updated: {}
    </div>
    <div style='text-align: center; padding: 0.5rem; margin-top: 1rem; border-top: 1px solid #e5e7eb;'>
        <div style='display: flex; align-items: center; justify-content: center; gap: 0.5rem; color: #6b7280; font-size: 0.8rem;'>
            <svg width="16" height="16" viewBox="0 0 600 600" fill="none" xmlns="http://www.w3.org/2000/svg" style="opacity: 0.7;">
                <path d="M280.19 142.036C282.913 150.121 288.605 156.721 296.196 160.516C300.733 162.743 305.6 163.898 310.468 163.898C313.851 163.898 317.315 163.321 320.616 162.248L470.852 112.17C487.599 106.642 496.592 88.4924 491.064 71.7447C485.536 54.997 467.303 46.0044 450.636 51.532L300.403 101.61C283.738 107.137 274.663 125.288 280.19 142.036Z" fill="#6b7280"/>
                <path d="M318.056 481.439C321.851 473.769 322.511 465.186 319.788 457.017C314.261 440.354 296.111 431.278 279.363 436.806L129.129 486.883C121.043 489.608 114.443 495.299 110.648 502.89C106.853 510.481 106.193 519.144 108.916 527.308C113.371 540.674 125.828 549.173 139.194 549.173C142.576 549.173 145.959 548.68 149.259 547.525L299.493 497.446C307.578 494.721 314.178 489.03 317.973 481.439H318.056Z" fill="#6b7280"/>
                <path d="M174.588 202.183C179.125 204.493 183.992 205.565 188.778 205.565C200.493 205.565 211.795 199.13 217.405 187.91L288.191 46.2558C291.986 38.6657 292.646 30.0031 289.924 21.8355C287.201 13.7504 281.509 7.15028 273.918 3.35523C258.161 -4.48236 238.938 1.87022 231.101 17.6279L160.315 159.282C152.477 175.04 158.83 194.263 174.588 202.1V202.183Z" fill="#6b7280"/>
                <path d="M425.308 396.871C417.718 393.076 408.973 392.416 400.888 395.139C392.803 397.861 386.203 403.554 382.408 411.144L311.622 552.797C307.826 560.388 307.166 569.05 309.889 577.219C312.612 585.388 318.304 591.903 325.894 595.699C330.432 597.925 335.217 599.08 340.167 599.08C343.549 599.08 347.015 598.502 350.314 597.431C358.4 594.707 365 589.016 368.795 581.424L439.581 439.772C443.375 432.182 444.036 423.519 441.313 415.351C438.59 407.266 432.898 400.666 425.308 396.871Z" fill="#6b7280"/>
                <path d="M102.067 299.128C106.522 312.493 118.98 320.991 132.428 320.991C135.728 320.991 139.193 320.496 142.493 319.341C159.158 313.813 168.233 295.663 162.706 278.915L112.628 128.681C107.1 112.016 88.9499 102.858 72.2022 108.468C55.5369 113.996 46.4619 132.146 51.9896 148.894L102.067 299.128Z" fill="#6b7280"/>
                <path d="M497.827 299.945C492.299 283.197 474.145 274.205 457.398 279.732C449.313 282.455 442.714 288.147 438.919 295.738C435.124 303.328 434.464 311.99 437.186 320.158L487.264 470.392C491.721 483.758 504.179 492.257 517.541 492.257C520.926 492.257 524.308 491.759 527.609 490.604C544.273 485.076 553.35 466.927 547.822 450.179L497.743 299.945H497.827Z" fill="#6b7280"/>
                <path d="M174.01 442.593C177.392 442.593 180.857 442.015 184.157 440.94C192.242 438.219 198.843 432.527 202.637 424.936C206.432 417.347 207.092 408.684 204.37 400.516C201.648 392.431 195.955 385.831 188.365 382.036L46.7104 311.25C30.9528 303.413 11.7301 309.765 3.8925 325.523C0.0974548 333.195 -0.562554 341.775 2.15998 349.943C4.88251 358.028 10.5751 364.628 18.1652 368.423L159.82 439.209C164.357 441.438 169.225 442.593 174.092 442.593H174.01Z" fill="#6b7280"/>
                <path d="M597.816 249.22C595.096 241.136 589.405 234.535 581.814 230.74L440.159 159.954C424.401 152.117 405.178 158.47 397.341 174.227C393.546 181.9 392.886 190.48 395.608 198.648C398.331 206.732 404.023 213.333 411.614 217.128L553.266 287.914C557.806 290.224 562.673 291.296 567.456 291.296C579.174 291.296 590.477 284.861 596.084 273.641C599.88 266.051 600.541 257.388 597.816 249.22Z" fill="#6b7280"/>
            </svg>
            <span>Powered by Domino Apps</span>
        </div>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()