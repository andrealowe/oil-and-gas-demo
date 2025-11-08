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
    page_icon="⚡",
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
    
    /* Primary color scheme */
    :root {
        --aramco-blue: #00A3E0;
        --aramco-green: #84BD00;
        --aramco-dark-blue: #0033A0;
        --aramco-light-gray: #F5F5F5;
        --aramco-dark-gray: #333333;
    }
    
    /* Main background and container styling */
    .main > div {
        padding: 2rem 1rem;
        background: linear-gradient(135deg, #f8fafb 0%, #e9ecef 100%);
        min-height: 100vh;
    }
    
    /* Header styling */
    .aramco-header {
        background: linear-gradient(135deg, var(--aramco-blue), var(--aramco-dark-blue));
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    
    /* Metric card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid var(--aramco-blue);
        margin-bottom: 1rem;
        font-family: 'Helvetica', sans-serif;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: var(--aramco-dark-blue);
        word-break: break-all;
        line-height: 1.2;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: var(--aramco-dark-gray);
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .metric-delta {
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Status indicators */
    .status-excellent { color: var(--aramco-green); }
    .status-good { color: #FFA500; }
    .status-warning { color: #FF6B6B; }
    .status-critical { color: #DC143C; }
    
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
        font=dict(family="Arial", size=12),
        title=dict(
            font=dict(size=16, color='#0033A0'),
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
        title=dict(font=dict(size=16, color='#0033A0'), x=0.5)
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
        title=dict(font=dict(size=16, color='#0033A0'), x=0.5)
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
    
    # Header
    st.markdown("""
    <div class="aramco-header">
        <h1>Oil & Gas Geospatial Operations Dashboard</h1>
        <p>Real-time facility monitoring, equipment health, and production optimization</p>
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
        st.experimental_rerun()
    
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
        st.plotly_chart(facility_map, use_container_width=True)
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
        st.dataframe(regional_summary, use_container_width=True)
    
    with tab2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        production_chart = create_production_efficiency_chart(filtered_df)
        st.plotly_chart(production_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        equipment_dashboard = create_equipment_health_dashboard(filtered_df)
        st.plotly_chart(equipment_dashboard, use_container_width=True)
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
        
        if st.button("🤖 Generate AI Predictions", type="primary"):
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
        Oil & Gas Geospatial Dashboard | Last Updated: {} | Data Source: Domino Data Lab
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()