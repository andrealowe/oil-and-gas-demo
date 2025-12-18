"""
Comprehensive Forecasting Dashboard for Oil & Gas Operations
Professional Aramco-inspired design with time series forecasting and predictive analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta, date
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
    page_title="Oil & Gas Forecasting Dashboard",
    page_icon="/mnt/code/docs/domino_logo.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force light theme by setting theme in session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Professional MLOps Platform CSS with Light Mode Override
st.markdown("""
<style>
    /* Hide Streamlit header and footer */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    .viewerBadge_container__1QSob {
        display: none !important;
    }
    
    footer {
        display: none !important;
    }
    
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
    
    /* Main background - clean enterprise styling */
    .main > div {
        padding: 1.5rem;
        background: var(--gray-50);
        min-height: 100vh;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
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
    
    .header-stats {
        display: flex;
        gap: 2rem;
        align-items: center;
    }
    
    .stat-item {
        text-align: center;
        padding: 0.5rem 1rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stat-label {
        display: block;
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.8);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    
    .stat-value {
        display: block;
        font-size: 1.2rem;
        font-weight: 700;
        color: white;
    }
    
    /* Forecast card styling - enterprise MLOps platform design */
    .forecast-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 2rem 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.08), 0 2px 6px -1px rgba(0, 0, 0, 0.04);
        border: 1px solid #e2e8f0;
        border-left: 4px solid var(--primary-blue);
        margin-bottom: 1.5rem;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.3s ease;
    }
    
    .forecast-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px -4px rgba(0, 0, 0, 0.12), 0 4px 8px -2px rgba(0, 0, 0, 0.08);
        border-left-color: #1d4ed8;
    }
    
    .forecast-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--gray-900);
        word-break: break-word;
        line-height: 1.2;
        margin: 0.75rem 0;
    }
    
    .forecast-label {
        font-size: 0.95rem;
        color: var(--gray-600);
        margin-bottom: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        line-height: 1.3;
    }
    
    .forecast-change {
        font-size: 0.85rem;
        margin-top: 0.75rem;
        color: var(--gray-600);
        font-weight: 500;
        line-height: 1.4;
    }
    
    /* Blue box text styling - make text white */
    .enterprise-header .header-left h1,
    .enterprise-header .header-subtitle,
    .enterprise-header .stat-label,
    .enterprise-header .stat-value {
        color: white !important;
    }
    
    .enterprise-header .stat-label {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Sidebar blue header text styling - make text white for header only */
    .stSidebar div[style*="background: linear-gradient"] h2,
    .stSidebar div[style*="background: linear-gradient"] p {
        color: white !important;
    }
    
    /* Regular sidebar text should be dark for readability */
    .stSidebar .stMarkdown p,
    .stSidebar .stMarkdown h3,
    .stSidebar .stSelectbox label,
    .stSidebar .stSlider label,
    .stSidebar .stMultiSelect label,
    .stSidebar .stCheckbox label,
    .stSidebar .stExpander label,
    .stSidebar label,
    .stSidebar .stMarkdown {
        color: var(--text-primary) !important;
    }
    
    /* Status indicators - professional MLOps colors */
    .forecast-up { color: var(--success-green); }
    .forecast-down { color: var(--error-red); }
    .forecast-neutral { color: var(--warning-amber); }
    
    /* Confidence indicators */
    .confidence-high { color: var(--success-green); }
    .confidence-medium { color: var(--warning-amber); }
    .confidence-low { color: var(--error-red); }
    
    /* Chart container styling */
    .chart-container {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid var(--gray-200);
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
    }
    
    /* Tab content styling */
    .stTabs > div[data-baseweb="tab-panel"] {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        margin-top: 1rem;
        border: 1px solid var(--gray-200);
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(135deg, var(--gray-100) 0%, var(--gray-50) 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border-left: 4px solid var(--accent-blue);
    }
    
    .section-header h3 {
        margin: 0;
        color: var(--gray-800);
        font-size: 1.25rem;
        font-weight: 600;
    }
    
    /* Streamlit specific overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        margin-bottom: 0;
        background: transparent;
        border-bottom: 1px solid var(--gray-200);
        padding-bottom: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
        border-radius: 10px 10px 0 0;
        color: #64748b !important;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 1rem 1.5rem;
        border: 1px solid #e2e8f0;
        border-bottom: none;
        transition: all 0.2s ease;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%) !important;
        color: var(--primary-blue) !important;
        border-color: var(--primary-blue);
        border-left: 3px solid var(--primary-blue);
        position: relative;
        z-index: 1;
        font-weight: 700;
        box-shadow: 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .stTabs [aria-selected="true"]:after {
        content: "";
        position: absolute;
        bottom: -1px;
        left: 0;
        right: 0;
        height: 2px;
        background: white;
    }
    
    /* Main container styling */
    .main .block-container {
        padding: 2rem 1rem;
        max-width: 1400px;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1lcbmhc, .css-1oe5cao {
        background-color: var(--gray-100);
    }
    
    /* Hide Streamlit branding */
    .css-1rs6os, .css-17ziqus {
        visibility: hidden;
    }
    
    /* Better spacing for content */
    .element-container {
        margin-bottom: 1rem;
    }
    
    /* Sidebar control panel styling */
    .control-panel {
        background: transparent;
        padding: 0;
        margin: 0;
    }
    
    /* Section headers within tabs - Enhanced professional styling */
    .tab-section-header {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 1.5rem 0 1rem 0;
        border-left: 4px solid var(--primary-blue);
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    
    .tab-section-header h3 {
        margin: 0;
        color: #1e293b;
        font-size: 1.25rem;
        font-weight: 700;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }
    
    /* Override Streamlit's default markdown styling in tabs */
    .stTabs [data-baseweb="tab-panel"] h3 {
        background: transparent !important;
        padding: 0.75rem 0 !important;
        margin-bottom: 1.5rem !important;
        border-bottom: 2px solid var(--gray-200) !important;
        border-radius: 0 !important;
        box-shadow: none !important;
    }
    
    /* Control panel styling */
    .control-panel {
        background: var(--aramco-light-gray);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #ddd;
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    /* Confidence indicator */
    .confidence-high { color: var(--aramco-green); }
    .confidence-medium { color: var(--aramco-orange); }
    .confidence-low { color: #FF6B6B; }
    
    /* Loading spinner */
    .loading-spinner {
        text-align: center;
        color: var(--aramco-blue);
        font-size: 1.2rem;
        padding: 2rem;
    }
    
    /* Maintenance alert */
    .maintenance-alert {
        background: #FFF3CD;
        border: 1px solid #FFEAA7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Parameter controls */
    .param-control {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid var(--accent-blue);
    }
    
    /* Comprehensive Light Mode Streamlit Component Overrides */
    
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
    
    /* Primary button variant for "Generate New Forecasts" - Blue theme */
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
    
    /* Progress bars */
    .stProgress > div > div {
        background-color: var(--gray-200) !important;
    }
    
    .stProgress > div > div > div {
        background-color: var(--primary-blue) !important;
    }
    
    /* Columns */
    .stColumn {
        background-color: transparent !important;
    }
    
    /* Expander */
    .stExpander {
        background-color: white !important;
        border: 1px solid var(--gray-200) !important;
        border-radius: 8px !important;
    }
    
    .stExpander > div > div {
        color: var(--text-primary) !important;
    }
    
    /* Code blocks */
    .stCode {
        background-color: var(--gray-100) !important;
        border: 1px solid var(--gray-200) !important;
        color: var(--text-primary) !important;
    }
    
    /* Dataframe */
    .stDataFrame {
        background-color: white !important;
    }
    
    /* Alert messages */
    .stAlert {
        background-color: white !important;
        border-radius: 8px !important;
        border: 1px solid var(--gray-200) !important;
    }
    
    .stAlert > div {
        color: var(--text-primary) !important;
    }
    
    /* Info messages */
    .stInfo {
        background-color: #e0f2fe !important;
        border-left: 4px solid var(--primary-blue) !important;
        color: var(--text-primary) !important;
    }
    
    /* Success messages */
    .stSuccess {
        background-color: #e8f5e8 !important;
        border-left: 4px solid var(--success-green) !important;
        color: var(--text-primary) !important;
    }
    
    /* Warning messages */
    .stWarning {
        background-color: #fff4e6 !important;
        border-left: 4px solid var(--warning-amber) !important;
        color: var(--text-primary) !important;
    }
    
    /* Error messages */
    .stError {
        background-color: #fef2f2 !important;
        border-left: 4px solid var(--error-red) !important;
        color: var(--text-primary) !important;
    }
    
    /* Text elements */
    p, span, div, label {
        color: var(--text-primary) !important;
    }
    
    /* Ensure plotly charts have white backgrounds */
    .stPlotlyChart {
        background-color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Light mode chart configuration
def apply_light_mode_layout(fig):
    """Apply consistent light mode styling to Plotly charts"""
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", color='#111827'),
        title_font=dict(color='#111827', size=16),
        xaxis=dict(
            gridcolor='#e5e7eb',
            linecolor='#e5e7eb',
            tickcolor='#6b7280',
            title_font=dict(color='#374151', size=14),
            tickfont=dict(color='#374151', size=12)
        ),
        yaxis=dict(
            gridcolor='#e5e7eb',
            linecolor='#e5e7eb',
            tickcolor='#6b7280',
            title_font=dict(color='#374151', size=14),
            tickfont=dict(color='#374151', size=12)
        ),
        legend=dict(
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='#e5e7eb',
            borderwidth=1,
            font=dict(color='#374151', size=12)
        )
    )
    return fig

# Data loading functions
@st.cache_data(ttl=300)
def load_production_timeseries():
    """Load production time series data with caching - prioritizes real data"""
    try:
        paths = get_data_paths('Oil-and-Gas-Demo')
        
        # Try to load real data first
        real_data_path = paths['base_data_path'] / 'production_timeseries_real_based.parquet'
        if real_data_path.exists():
            df = pd.read_parquet(real_data_path)
            df['date'] = pd.to_datetime(df['date'])
            # Add data source indicator
            df['data_source'] = 'Real (US Production Trends)'
            st.sidebar.success("✅ Using Real Production Data")
            return df
        
        # Fallback to synthetic data
        synthetic_data_path = paths['base_data_path'] / 'production_timeseries.parquet'
        df = pd.read_parquet(synthetic_data_path)
        df['date'] = pd.to_datetime(df['date'])
        df['data_source'] = 'Synthetic'
        st.sidebar.info("📊 Using Synthetic Production Data")
        return df
    except Exception as e:
        st.error(f"Error loading production data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_price_data():
    """Load price time series data with caching - prioritizes real data"""
    try:
        paths = get_data_paths('Oil-and-Gas-Demo')
        
        # Try to load real FRED price data first
        real_price_path = paths['base_data_path'] / 'price_data.csv'
        if real_price_path.exists():
            df = pd.read_csv(real_price_path)
            df['date'] = pd.to_datetime(df['date'])
            # Rename columns to match app expectations
            column_mapping = {
                'brent_crude_price': 'crude_oil_price',
                'wti_crude_price': 'wti_crude_price', 
                'natural_gas_price': 'natural_gas_price'
            }
            df = df.rename(columns=column_mapping)
            df['data_source'] = 'Real (FRED)'
            st.sidebar.success("✅ Using Real Price Data (FRED)")
            return df
        
        # Fallback to synthetic data
        synthetic_path = paths['base_data_path'] / 'prices_timeseries.parquet'
        df = pd.read_parquet(synthetic_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        df['data_source'] = 'Synthetic'
        st.sidebar.info("📊 Using Synthetic Price Data")
        return df
    except Exception as e:
        st.warning(f"Error loading price data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300) 
def load_demand_data():
    """Load demand time series data with caching"""
    try:
        paths = get_data_paths('Oil-and-Gas-Demo')
        data_path = paths['base_data_path'] / 'demand_timeseries.parquet'
        df = pd.read_parquet(data_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.warning(f"Error loading demand data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_maintenance_data():
    """Load maintenance time series data with caching"""
    try:
        paths = get_data_paths('Oil-and-Gas-Demo')
        data_path = paths['base_data_path'] / 'maintenance_timeseries.parquet'
        df = pd.read_parquet(data_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.warning(f"Error loading maintenance data: {e}")
        return pd.DataFrame()

def call_forecasting_api(forecast_params):
    """Call the forecasting prediction API"""
    try:
        # Simulate API call (replace with actual endpoint when available)
        api_url = "http://localhost:8000/forecast"  # Update with actual API URL
        
        # For demonstration, create mock forecast based on parameters
        horizon_days = forecast_params.get('horizon_days', 30)
        forecast_type = forecast_params.get('forecast_type', 'oil_production_bpd')
        
        # Generate mock forecast data - start from end of available historical data
        # Use a fixed recent date that would follow historical data
        start_date = datetime(2025, 11, 2)  # Start forecasts from recent date
        
        dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(horizon_days)]
        
        # Base values and trends for different forecast types
        forecast_configs = {
            'oil_production_bpd': {'base': 50000, 'trend': -0.1, 'volatility': 0.05},
            'gas_production_mcfd': {'base': 150000, 'trend': 0.05, 'volatility': 0.08},
            'crude_oil_price_usd_bbl': {'base': 75.0, 'trend': 0.02, 'volatility': 0.12},
            'natural_gas_price_usd_mmbtu': {'base': 3.5, 'trend': 0.03, 'volatility': 0.15},
            'gasoline_demand_thousand_bpd': {'base': 8000, 'trend': 0.01, 'volatility': 0.06},
            'diesel_demand_thousand_bpd': {'base': 4000, 'trend': 0.015, 'volatility': 0.07}
        }
        
        config = forecast_configs.get(forecast_type, forecast_configs['oil_production_bpd'])
        
        # Generate forecast values with trend and seasonality
        # Try to get the last meaningful historical value for a smoother connection
        try:
            production_df = load_production_timeseries()
            if not production_df.empty:
                if forecast_type == 'oil_production_bpd' and 'oil_production_bpd' in production_df.columns:
                    # Group by date and sum to get daily totals, then get the last non-zero value
                    daily_totals = production_df.groupby('date')['oil_production_bpd'].sum()
                    non_zero_values = daily_totals[daily_totals > 0]
                    if len(non_zero_values) > 0:
                        last_historical_value = non_zero_values.iloc[-1]
                        config['base'] = last_historical_value
                elif forecast_type == 'gas_production_mcfd' and 'gas_production_mcfd' in production_df.columns:
                    daily_totals = production_df.groupby('date')['gas_production_mcfd'].sum()
                    non_zero_values = daily_totals[daily_totals > 0]
                    if len(non_zero_values) > 0:
                        last_historical_value = non_zero_values.iloc[-1]
                        config['base'] = last_historical_value
        except:
            pass  # Use default base values if there's any error
        
        values = []
        for i in range(horizon_days):
            trend_value = config['base'] * (1 + config['trend'] * i / 365)
            seasonal_value = config['base'] * 0.1 * np.sin(2 * np.pi * i / 365)
            random_value = np.random.normal(0, config['base'] * config['volatility'])
            forecast_value = max(0, trend_value + seasonal_value + random_value)
            values.append(round(forecast_value, 2))
        
        lower_bound = [v * 0.9 for v in values]
        upper_bound = [v * 1.1 for v in values]
        
        mock_forecast = {
            "status": "success",
            "forecast": {
                "dates": dates,
                "values": values,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "source": "synthetic"
            },
            "metadata": {
                "forecast_type": forecast_type,
                "horizon_days": horizon_days,
                "confidence": np.random.uniform(0.75, 0.95),
                "model_version": "1.0"
            }
        }
        
        return mock_forecast
        
    except Exception as e:
        st.warning(f"API call failed, using fallback forecast: {e}")
        return {"error": str(e)}

def create_production_forecast_chart(production_df, forecast_data=None):
    """Create production forecasting chart with historical and predicted data"""
    fig = go.Figure()
    
    if not production_df.empty:
        # Filter to last 3 years but end before forecast period
        three_years_ago = datetime(2021, 12, 1)  # Start of historical period
        historical_end = datetime(2025, 11, 1)    # End before forecast starts
        production_df['date'] = pd.to_datetime(production_df['date'])
        recent_production = production_df[
            (production_df['date'] >= three_years_ago) & 
            (production_df['date'] <= historical_end)
        ]
        
        # Historical production data (aggregated daily)
        daily_production = recent_production.groupby('date').agg({
            'oil_production_bpd': 'sum',
            'gas_production_mcfd': 'sum'
        }).reset_index()
        
        # Oil production historical
        fig.add_trace(go.Scatter(
            x=daily_production['date'],
            y=daily_production['oil_production_bpd'],
            mode='lines',
            name='Historical Oil Production',
            line=dict(color='#1f2937', width=2)
        ))
        
        # Gas production historical (secondary y-axis)
        fig.add_trace(go.Scatter(
            x=daily_production['date'],
            y=daily_production['gas_production_mcfd'],
            mode='lines',
            name='Historical Gas Production',
            line=dict(color='#059669', width=2),
            yaxis='y2'
        ))
    
    # Add forecast data if available
    if forecast_data and 'forecast' in forecast_data:
        forecast = forecast_data['forecast']
        forecast_dates = pd.to_datetime(forecast['dates'])
        
        # Note: Bridge connection disabled to prevent timeline issues
        # Forecast will appear as a separate section on the chart
        
        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast['values'],
            mode='lines',
            name='Forecast',
            line=dict(color='#3b82f6', width=3, dash='dash')
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast['upper_bound'],
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast['lower_bound'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name='Confidence Interval',
            fillcolor='rgba(59,130,246,0.2)'
        ))
    
    # Update layout with zoom and interactive features
    fig.update_layout(
        title='Production Forecasting Analysis',
        xaxis_title='Date',
        yaxis_title='Oil Production (BPD)',
        yaxis2=dict(
            title='Gas Production (MCFD)',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        hovermode='x unified',
        height=500,
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", size=12),
        title_font=dict(size=16, color='#111827'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        # Enable zoom and pan
        dragmode='zoom',
        selectdirection='d'
    )
    
    # Update axes for better zoom experience
    fig.update_xaxes(
        showspikes=True,
        spikecolor="gray",
        spikesnap="cursor",
        spikemode="across"
    )
    fig.update_yaxes(
        showspikes=True,
        spikecolor="gray",
        spikesnap="cursor",
        spikemode="across"
    )
    
    return apply_light_mode_layout(fig)

def create_price_forecast_chart(price_df, price_forecast_data=None):
    """Create price forecasting chart"""
    fig = go.Figure()
    
    if not price_df.empty:
        # Filter to last 3 years but end before forecast period
        three_years_ago = datetime(2021, 12, 1)
        historical_end = datetime(2025, 11, 1)
        price_df['date'] = pd.to_datetime(price_df['date'])
        recent_price = price_df[
            (price_df['date'] >= three_years_ago) & 
            (price_df['date'] <= historical_end)
        ]
        
        # Historical price data
        price_cols = [col for col in recent_price.columns if 'price' in col.lower()]
        
        colors = ['#1f2937', '#059669', '#3b82f6', '#dc2626']
        for i, col in enumerate(price_cols[:4]):  # Limit to 4 price series
            fig.add_trace(go.Scatter(
                x=recent_price['date'] if 'date' in recent_price.columns else recent_price.index,
                y=recent_price[col],
                mode='lines',
                name=f'Historical {col.replace("_", " ").title()}',
                line=dict(color=colors[i % len(colors)], width=2)
            ))
    
    # Add forecast data if available
    if price_forecast_data and 'forecast' in price_forecast_data:
        forecast = price_forecast_data['forecast']
        forecast_dates = pd.to_datetime(forecast['dates'])
        
        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast['values'],
            mode='lines',
            name='Price Forecast',
            line=dict(color='#3b82f6', width=3, dash='dash')
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast['upper_bound'],
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast['lower_bound'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name='Price Confidence Interval',
            fillcolor='rgba(59,130,246,0.2)'
        ))
    
    fig.update_layout(
        title='Oil & Gas Price Forecasting',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        height=500,
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", size=12),
        title_font=dict(size=16, color='#111827'),
        # Enable zoom and pan
        dragmode='zoom',
        selectdirection='d'
    )
    
    # Update axes for better zoom experience
    fig.update_xaxes(
        showspikes=True,
        spikecolor="gray",
        spikesnap="cursor",
        spikemode="across"
    )
    fig.update_yaxes(
        showspikes=True,
        spikecolor="gray",
        spikesnap="cursor",
        spikemode="across"
    )
    
    return apply_light_mode_layout(fig)

def create_demand_forecast_chart(demand_df, demand_forecast_data=None):
    """Create demand forecasting chart by region"""
    if demand_df.empty:
        return go.Figure()
    
    # Filter to last 3 years but end before forecast period
    three_years_ago = datetime(2021, 12, 1)
    historical_end = datetime(2025, 11, 1)
    demand_df['date'] = pd.to_datetime(demand_df['date'])
    recent_demand = demand_df[
        (demand_df['date'] >= three_years_ago) & 
        (demand_df['date'] <= historical_end)
    ]
    
    # Group by date and region if available
    if 'region' in recent_demand.columns and 'date' in recent_demand.columns:
        regional_demand = recent_demand.groupby(['date', 'region']).sum().reset_index()
        
        fig = px.line(
            regional_demand,
            x='date',
            y=regional_demand.columns[-1],  # Assume last column is demand
            color='region',
            title='Regional Demand Forecasting',
            color_discrete_sequence=['#3b82f6', '#059669', '#0f172a', '#dc2626', '#7c3aed']
        )
    else:
        # Simple time series
        demand_col = [col for col in recent_demand.columns if 'demand' in col.lower()]
        if demand_col:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=recent_demand['date'] if 'date' in recent_demand.columns else recent_demand.index,
                y=recent_demand[demand_col[0]],
                mode='lines',
                name='Historical Demand',
                line=dict(color='#0f172a', width=2)
            ))
        else:
            return go.Figure()
    
    # Add forecast if available
    if demand_forecast_data and 'forecast' in demand_forecast_data:
        forecast = demand_forecast_data['forecast']
        forecast_dates = pd.to_datetime(forecast['dates'])
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast['values'],
            mode='lines',
            name='Demand Forecast',
            line=dict(color='#3b82f6', width=3, dash='dash')
        ))
    
    fig.update_layout(
        title='Regional Demand Forecasting',
        xaxis_title='Date',
        yaxis_title='Demand (Thousand BPD)',
        height=500,
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", size=12),
        title_font=dict(size=16, color='#111827'),
        hovermode='x unified',
        # Enable zoom and pan
        dragmode='zoom',
        selectdirection='d'
    )
    
    # Update axes for better zoom experience
    fig.update_xaxes(
        showspikes=True,
        spikecolor="gray",
        spikesnap="cursor",
        spikemode="across"
    )
    fig.update_yaxes(
        showspikes=True,
        spikecolor="gray",
        spikesnap="cursor",
        spikemode="across"
    )
    
    return apply_light_mode_layout(fig)

def create_maintenance_optimization_chart(maintenance_df):
    """Create maintenance scheduling optimization chart"""
    if maintenance_df.empty:
        return go.Figure()
    
    # Filter to last 3 years but end before forecast period
    three_years_ago = datetime(2021, 12, 1)
    historical_end = datetime(2025, 11, 1)
    if 'date' in maintenance_df.columns:
        maintenance_df['date'] = pd.to_datetime(maintenance_df['date'])
        maintenance_df = maintenance_df[
            (maintenance_df['date'] >= three_years_ago) & 
            (maintenance_df['date'] <= historical_end)
        ]
    
    # Create maintenance schedule visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Maintenance Events Timeline",
            "Cost Analysis", 
            "Equipment Downtime",
            "Efficiency Impact"
        )
    )
    
    # Mock maintenance schedule data if not available
    if 'maintenance_type' not in maintenance_df.columns:
        # Create mock data
        maintenance_types = ['Preventive', 'Corrective', 'Predictive']
        maintenance_df['maintenance_type'] = np.random.choice(maintenance_types, len(maintenance_df))
        maintenance_df['cost'] = np.random.uniform(10000, 100000, len(maintenance_df))
        maintenance_df['downtime_hours'] = np.random.uniform(2, 24, len(maintenance_df))
    
    # Maintenance events timeline
    if 'date' in maintenance_df.columns:
        maintenance_counts = maintenance_df.groupby(['date', 'maintenance_type']).size().reset_index(name='count')
        
        for mtype in maintenance_df['maintenance_type'].unique():
            type_data = maintenance_counts[maintenance_counts['maintenance_type'] == mtype]
            fig.add_trace(
                go.Scatter(
                    x=type_data['date'],
                    y=type_data['count'],
                    mode='markers+lines',
                    name=f'{mtype} Maintenance',
                    marker=dict(size=8)
                ),
                row=1, col=1
            )
    
    # Cost analysis
    if 'cost' in maintenance_df.columns:
        cost_by_type = maintenance_df.groupby('maintenance_type')['cost'].sum().reset_index()
    else:
        cost_by_type = pd.DataFrame({
            'maintenance_type': ['Preventive', 'Corrective', 'Predictive'],
            'cost': [50000, 80000, 30000]
        })
    fig.add_trace(
        go.Bar(
            x=cost_by_type['maintenance_type'],
            y=cost_by_type['cost'],
            name='Maintenance Costs',
            marker_color=['#1f2937', '#059669', '#3b82f6']
        ),
        row=1, col=2
    )
    
    # Equipment downtime
    if 'downtime_hours' in maintenance_df.columns:
        downtime_by_type = maintenance_df.groupby('maintenance_type')['downtime_hours'].mean().reset_index()
    else:
        downtime_by_type = pd.DataFrame({
            'maintenance_type': ['Preventive', 'Corrective', 'Predictive'],
            'downtime_hours': [8, 16, 4]
        })
    fig.add_trace(
        go.Bar(
            x=downtime_by_type['maintenance_type'],
            y=downtime_by_type['downtime_hours'],
            name='Avg Downtime',
            marker_color=['#dc2626', '#d97706', '#059669']
        ),
        row=2, col=1
    )
    
    # Efficiency impact (mock correlation)
    efficiency_impact = np.random.uniform(0.8, 1.0, len(maintenance_df))
    downtime_values = maintenance_df['downtime_hours'] if 'downtime_hours' in maintenance_df.columns else np.random.uniform(2, 24, len(maintenance_df))
    cost_values = maintenance_df['cost'] if 'cost' in maintenance_df.columns else np.random.uniform(10000, 100000, len(maintenance_df))
    
    fig.add_trace(
        go.Scatter(
            x=downtime_values,
            y=efficiency_impact,
            mode='markers',
            name='Efficiency vs Downtime',
            marker=dict(
                size=10,
                color=cost_values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Cost ($)")
            )
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Maintenance Scheduling Optimization",
        title_x=0.5,
        title_font=dict(size=16, color='#111827'),
        # Enable zoom and pan
        dragmode='zoom',
        selectdirection='d'
    )
    
    # Update all subplots for better zoom experience
    for i in range(1, 5):  # 4 subplots
        fig.update_xaxes(
            showspikes=True,
            spikecolor="gray",
            spikesnap="cursor",
            spikemode="across",
            row=(i-1)//2+1,
            col=(i-1)%2+1
        )
        fig.update_yaxes(
            showspikes=True,
            spikecolor="gray",
            spikesnap="cursor",
            spikemode="across",
            row=(i-1)//2+1,
            col=(i-1)%2+1
        )
    
    return apply_light_mode_layout(fig)

def display_forecast_metrics(forecast_data_list):
    """Display forecasting KPI metrics"""
    st.markdown('<div class="tab-section-header"><h3>Forecast Summary Metrics</h3></div>', unsafe_allow_html=True)
    
    # Calculate summary metrics from available forecasts
    total_forecasts = len(forecast_data_list)
    avg_confidence = np.mean([f.get('metadata', {}).get('confidence', 0.8) for f in forecast_data_list if f])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="forecast-card">
            <div class="forecast-label">Active Forecasts</div>
            <div class="forecast-value">{total_forecasts}</div>
            <div class="forecast-change">Models Running</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        confidence_class = "confidence-high" if avg_confidence >= 0.8 else "confidence-medium" if avg_confidence >= 0.6 else "confidence-low"
        st.markdown(f"""
        <div class="forecast-card">
            <div class="forecast-label">Avg Confidence</div>
            <div class="forecast-value {confidence_class}">{avg_confidence:.1%}</div>
            <div class="forecast-change">Model Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Mock forecast horizon
        horizon_days = 30
        st.markdown(f"""
        <div class="forecast-card">
            <div class="forecast-label">Forecast Horizon</div>
            <div class="forecast-value">{horizon_days}</div>
            <div class="forecast-change">Days Ahead</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Mock update frequency
        st.markdown(f"""
        <div class="forecast-card">
            <div class="forecast-label">Update Frequency</div>
            <div class="forecast-value">Daily</div>
            <div class="forecast-change">Refresh Rate</div>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main forecasting dashboard application"""
    
    # Professional Header
    st.markdown("""
    <div class="enterprise-header">
        <div class="header-content">
            <div class="header-left">
                <h1>Energy Analytics Platform</h1>
                <div class="header-subtitle">Strategic Forecasting & Risk Management Dashboard</div>
            </div>
            <div class="header-right">
                <div class="header-stats">
                    <div class="stat-item">
                        <span class="stat-label">Active Models</span>
                        <span class="stat-value">4</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Coverage</span>
                        <span class="stat-value">Global</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Updated</span>
                        <span class="stat-value">Live</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Professional Sidebar
    st.sidebar.markdown("""
    <div style='background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%); padding: 1.5rem; margin: -1rem -1rem 1.5rem -1rem; border-radius: 0 0 12px 12px;'>
        <h2 style='color: white; margin: 0; font-size: 1.5rem; font-weight: 700; text-align: center;'>Control Panel</h2>
        <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; text-align: center; font-size: 0.9rem;'>Forecast Management</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Forecast generation button at the top
    st.sidebar.markdown("### Quick Actions")
    
    # Generate forecasts button
    generate_forecasts = st.sidebar.button("Generate New Forecasts", type="primary")
    
    st.sidebar.markdown("---")  # Separator
    
    # Load data
    with st.spinner("Loading time series data..."):
        production_df = load_production_timeseries()
        price_df = load_price_data()
        demand_df = load_demand_data()
        maintenance_df = load_maintenance_data()
    
    # Data Quality Banner - show data sources being used
    if not production_df.empty or not price_df.empty:
        data_sources = []
        if not production_df.empty and 'data_source' in production_df.columns:
            prod_source = production_df['data_source'].iloc[0]
            if 'Real' in prod_source:
                data_sources.append("✅ Production: Real US Trends")
            else:
                data_sources.append("📊 Production: Synthetic")
        
        if not price_df.empty and 'data_source' in price_df.columns:
            price_source = price_df['data_source'].iloc[0]
            if 'Real' in price_source:
                data_sources.append("✅ Prices: Real Market Data")
            else:
                data_sources.append("📊 Prices: Synthetic")
        
        if data_sources:
            data_source_text = " | ".join(data_sources)
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #e8f5e8 0%, #f0f9ff 100%); 
                        padding: 12px 20px; margin: 15px 0; border-radius: 10px; 
                        border-left: 5px solid #10b981; font-size: 14px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <strong>📈 Data Sources:</strong> {data_source_text}
            </div>
            """, unsafe_allow_html=True)
    
    # Forecast parameters
    st.sidebar.markdown("### Forecast Parameters")
    
    forecast_horizon = st.sidebar.slider(
        "Forecast Horizon (Days)",
        min_value=7,
        max_value=365,
        value=30,
        step=7,
        help="Number of days to forecast ahead"
    )
    
    forecast_types = [
        'oil_production_bpd',
        'gas_production_mcfd', 
        'crude_oil_price_usd_bbl',
        'natural_gas_price_usd_mmbtu',
        'gasoline_demand_thousand_bpd',
        'diesel_demand_thousand_bpd'
    ]
    
    selected_forecast_types = st.sidebar.multiselect(
        "Select Forecast Types",
        options=forecast_types,
        default=['oil_production_bpd', 'crude_oil_price_usd_bbl', 'gasoline_demand_thousand_bpd'],
        help="Choose which metrics to forecast"
    )
    
    confidence_level = st.sidebar.selectbox(
        "Confidence Level",
        options=[0.80, 0.85, 0.90, 0.95],
        index=2,
        format_func=lambda x: f"{x:.0%}",
        help="Statistical confidence level for predictions"
    )
    
    # Advanced parameters
    with st.sidebar.expander("Advanced Parameters"):
        seasonality_adjustment = st.checkbox(
            "Seasonality Adjustment", 
            value=True,
            help="Include seasonal patterns in forecasts"
        )
        
        market_volatility = st.slider(
            "Market Volatility Factor",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Adjust for current market volatility"
        )
        
        external_factors = st.multiselect(
            "External Factors",
            options=["Geopolitical Events", "Weather Patterns", "Economic Indicators", "Supply Chain"],
            help="Include external factors in forecasting models"
        )
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize session state for forecasts
    if 'default_forecasts_generated' not in st.session_state:
        st.session_state.default_forecasts_generated = False
        st.session_state.forecast_results = {}
    
    # Store forecast results
    forecast_results = st.session_state.forecast_results
    
    # Generate default forecasts on first load
    if not st.session_state.default_forecasts_generated:
        with st.spinner("Loading default forecasts..."):
            for forecast_type in selected_forecast_types:
                forecast_params = {
                    'forecast_type': forecast_type,
                    'horizon_days': forecast_horizon,
                    'confidence_level': confidence_level,
                    'seasonality': seasonality_adjustment,
                    'volatility_factor': market_volatility,
                    'external_factors': external_factors
                }
                
                forecast_result = call_forecasting_api(forecast_params)
                forecast_results[forecast_type] = forecast_result
            
            st.session_state.forecast_results = forecast_results
            st.session_state.default_forecasts_generated = True
    
    # Handle manual forecast generation
    if generate_forecasts:
        st.sidebar.success("Generating forecasts...")
        
        forecast_results = {}
        for forecast_type in selected_forecast_types:
            forecast_params = {
                'forecast_type': forecast_type,
                'horizon_days': forecast_horizon,
                'confidence_level': confidence_level,
                'seasonality': seasonality_adjustment,
                'volatility_factor': market_volatility,
                'external_factors': external_factors
            }
            
            with st.spinner(f"Generating {forecast_type} forecast..."):
                forecast_result = call_forecasting_api(forecast_params)
                forecast_results[forecast_type] = forecast_result
        
        # Update session state with new forecasts
        st.session_state.forecast_results = forecast_results
        st.sidebar.success("Forecasts generated!")
    
    # Display forecast metrics
    if forecast_results:
        display_forecast_metrics(list(forecast_results.values()))
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Production Forecasts",
        "Price Forecasts", 
        "Demand Forecasts",
        "Maintenance Planning",
        "Forecast Controls",
        "Forecast Summary"
    ])
    
    with tab1:
        st.markdown('<div class="tab-section-header"><h3>Production Forecasting Analysis</h3></div>', unsafe_allow_html=True)
        
        # Production forecast chart
        production_forecast = forecast_results.get('oil_production_bpd')
        production_chart = create_production_forecast_chart(production_df, production_forecast)
        
        st.plotly_chart(production_chart, width='stretch', config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': ['drawrect', 'eraseshape'],
            'modeBarButtonsToRemove': ['select2d', 'lasso2d']
        })
        
        # Production insights
        if not production_df.empty:
            st.markdown('<div class="tab-section-header"><h3>Production Insights</h3></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Oil production statistics
                recent_oil = production_df.groupby('date')['oil_production_bpd'].sum().tail(30).mean()
                oil_trend = "increasing" if recent_oil > production_df.groupby('date')['oil_production_bpd'].sum().head(30).mean() else "decreasing"
                
                st.markdown(f"""
                **Oil Production Analysis:**
                - Recent daily average: {recent_oil:,.0f} BPD
                - 30-day trend: {oil_trend}
                - Peak production: {production_df.groupby('date')['oil_production_bpd'].sum().max():,.0f} BPD
                """)
            
            with col2:
                # Gas production statistics
                recent_gas = production_df.groupby('date')['gas_production_mcfd'].sum().tail(30).mean()
                gas_trend = "increasing" if recent_gas > production_df.groupby('date')['gas_production_mcfd'].sum().head(30).mean() else "decreasing"
                
                st.markdown(f"""
                **Gas Production Analysis:**
                - Recent daily average: {recent_gas:,.0f} MCFD
                - 30-day trend: {gas_trend}
                - Peak production: {production_df.groupby('date')['gas_production_mcfd'].sum().max():,.0f} MCFD
                """)
            
            # Regional production breakdown
            if 'region' in production_df.columns:
                st.markdown('<div class="tab-section-header"><h3>Production by Region</h3></div>', unsafe_allow_html=True)
                regional_production = production_df.groupby('region').agg({
                    'oil_production_bpd': 'sum',
                    'gas_production_mcfd': 'sum'
                }).round(0)
                st.dataframe(regional_production, width='stretch')
    
    with tab2:
        st.markdown('<div class="tab-section-header"><h3>Oil & Gas Price Forecasting</h3></div>', unsafe_allow_html=True)
        
        # Price forecast chart
        price_forecast = forecast_results.get('crude_oil_price_usd_bbl')
        price_chart = create_price_forecast_chart(price_df, price_forecast)
        
        st.plotly_chart(price_chart, width='stretch', config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': ['drawrect', 'eraseshape'],
            'modeBarButtonsToRemove': ['select2d', 'lasso2d']
        })
        
        # Price analysis
        if price_forecast:
            st.markdown('<div class="tab-section-header"><h3>Price Forecast Analysis</h3></div>', unsafe_allow_html=True)
            
            forecast_data = price_forecast.get('forecast', {})
            if forecast_data:
                forecast_values = forecast_data.get('values', [])
                
                if forecast_values:
                    avg_forecast = np.mean(forecast_values)
                    price_volatility = np.std(forecast_values) / avg_forecast
                    confidence = price_forecast.get('metadata', {}).get('confidence', 0.8)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        price_color = "forecast-up" if len(forecast_values) > 1 and forecast_values[-1] > forecast_values[0] else "forecast-down"
                        st.markdown(f"""
                        <div class="forecast-card">
                            <div class="forecast-label">Average Forecast Price</div>
                            <div class="forecast-value {price_color}">${avg_forecast:.2f}</div>
                            <div class="forecast-change">per barrel</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        volatility_color = "forecast-neutral" if price_volatility < 0.15 else "forecast-down"
                        st.markdown(f"""
                        <div class="forecast-card">
                            <div class="forecast-label">Price Volatility</div>
                            <div class="forecast-value {volatility_color}">{price_volatility:.1%}</div>
                            <div class="forecast-change">Standard Deviation</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        confidence_color = "forecast-up" if confidence > 0.8 else "forecast-neutral"
                        st.markdown(f"""
                        <div class="forecast-card">
                            <div class="forecast-label">Forecast Confidence</div>
                            <div class="forecast-value {confidence_color}">{confidence:.1%}</div>
                            <div class="forecast-change">Model Accuracy</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Market factors
        st.markdown('<div class="tab-section-header"><h3>Market Factors</h3></div>', unsafe_allow_html=True)
        factor_col1, factor_col2 = st.columns(2)
        
        with factor_col1:
            st.markdown("""
            **Bullish Factors:**
            - Global economic recovery
            - Supply chain constraints
            - Geopolitical tensions
            - Seasonal demand increase
            """)
        
        with factor_col2:
            st.markdown("""
            **Bearish Factors:**
            - Increased production capacity
            - Alternative energy adoption
            - Economic slowdown concerns
            - Strategic reserve releases
            """)
    
    with tab3:
        st.markdown('<div class="tab-section-header"><h3>Regional Demand Forecasting</h3></div>', unsafe_allow_html=True)
        
        # Demand forecast chart
        demand_forecast = forecast_results.get('gasoline_demand_thousand_bpd')
        demand_chart = create_demand_forecast_chart(demand_df, demand_forecast)
        
        st.plotly_chart(demand_chart, width='stretch', config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': ['drawrect', 'eraseshape'],
            'modeBarButtonsToRemove': ['select2d', 'lasso2d']
        })
        
        # Demand analysis
        st.markdown('<div class="tab-section-header"><h3>Demand Analysis by Product</h3></div>', unsafe_allow_html=True)
        
        demand_products = ['gasoline_demand_thousand_bpd', 'diesel_demand_thousand_bpd']
        for product in demand_products:
            if product in forecast_results:
                forecast = forecast_results[product]
                forecast_data = forecast.get('forecast', {})
                
                if forecast_data and forecast_data.get('values'):
                    values = forecast_data['values']
                    product_name = product.replace('_', ' ').replace('thousand bpd', '(K BPD)').title()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        current_demand = values[0] if values else 0
                        future_demand = values[-1] if len(values) > 1 else 0
                        demand_change = ((future_demand - current_demand) / current_demand * 100) if current_demand > 0 else 0
                        change_color = "forecast-up" if demand_change > 0 else "forecast-down"
                        
                        st.markdown(f"""
                        **{product_name}:**
                        - Current forecast: {format_large_number(current_demand)} KBPD
                        - Future forecast: {format_large_number(future_demand)} KBPD
                        - <span class="{change_color}">Change: {demand_change:+.1f}%</span>
                        """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="tab-section-header"><h3>Maintenance Scheduling Optimization</h3></div>', unsafe_allow_html=True)
        
        # Maintenance optimization chart
        maintenance_chart = create_maintenance_optimization_chart(maintenance_df)
        
        st.plotly_chart(maintenance_chart, width='stretch', config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': ['drawrect', 'eraseshape'],
            'modeBarButtonsToRemove': ['select2d', 'lasso2d']
        })
        
        # Maintenance recommendations
        st.markdown('<div class="tab-section-header"><h3>Maintenance Recommendations</h3></div>', unsafe_allow_html=True)
        
        # Mock maintenance alerts
        alert_col1, alert_col2 = st.columns(2)
        
        with alert_col1:
            st.markdown("""
            <div class="maintenance-alert">
                <strong>ALERT: Upcoming Maintenance</strong><br>
                • Facility ABC-001: Scheduled in 5 days<br>
                • Facility XYZ-045: Overdue by 2 days<br>
                • Facility DEF-123: Preventive maintenance due
            </div>
            """, unsafe_allow_html=True)
        
        with alert_col2:
            st.markdown("""
            <div class="maintenance-alert">
                <strong>INSIGHT: Optimization Opportunities</strong><br>
                • Consolidate maintenance for Region A<br>
                • Predictive maintenance for high-value assets<br>
                • Cost savings: $2.3M annually
            </div>
            """, unsafe_allow_html=True)
        
        # Maintenance schedule table
        if not maintenance_df.empty:
            st.markdown('<div class="tab-section-header"><h3>Maintenance Schedule</h3></div>', unsafe_allow_html=True)
            
            # Create a sample maintenance schedule
            future_dates = pd.date_range(start=datetime.now(), periods=20, freq='D')
            schedule_data = {
                'Date': future_dates,
                'Facility': [f'FAC-{i:03d}' for i in range(1, 21)],
                'Type': np.random.choice(['Preventive', 'Corrective', 'Predictive'], 20),
                'Priority': np.random.choice(['High', 'Medium', 'Low'], 20),
                'Est. Duration (hrs)': np.random.randint(4, 24, 20),
                'Est. Cost ($)': np.random.randint(10000, 100000, 20)
            }
            
            schedule_df = pd.DataFrame(schedule_data)
            st.dataframe(schedule_df, width='stretch')
    
    with tab5:
        st.markdown('<div class="tab-section-header"><h3>Interactive Forecast Controls</h3></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="param-control">', unsafe_allow_html=True)
        st.markdown('<h4 style="color: var(--gray-700); margin: 1rem 0 0.5rem 0; font-weight: 500;">Model Parameters</h4>', unsafe_allow_html=True)
        
        # Interactive parameter controls
        param_col1, param_col2 = st.columns(2)
        
        with param_col1:
            model_sensitivity = st.slider(
                "Model Sensitivity",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Adjust model sensitivity to market changes"
            )
            
            trend_weight = st.slider(
                "Trend Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Weight given to trend vs seasonality"
            )
        
        with param_col2:
            noise_reduction = st.slider(
                "Noise Reduction",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Filter out market noise"
            )
            
            forecast_smoothing = st.slider(
                "Forecast Smoothing",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.1,
                help="Smooth forecast predictions"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Model performance metrics
        st.markdown('<div class="tab-section-header"><h3>Model Performance</h3></div>', unsafe_allow_html=True)
        
        performance_col1, performance_col2, performance_col3 = st.columns(3)
        
        with performance_col1:
            st.markdown("""
            <div class="forecast-card">
                <div class="forecast-label">MAPE (7-day)</div>
                <div class="forecast-value confidence-high">4.2%</div>
                <div class="forecast-change">Mean Absolute Percentage Error</div>
            </div>
            """, unsafe_allow_html=True)
        
        with performance_col2:
            st.markdown("""
            <div class="forecast-card">
                <div class="forecast-label">RMSE</div>
                <div class="forecast-value confidence-medium">12.4</div>
                <div class="forecast-change">Root Mean Square Error</div>
            </div>
            """, unsafe_allow_html=True)
        
        with performance_col3:
            st.markdown("""
            <div class="forecast-card">
                <div class="forecast-label">R²</div>
                <div class="forecast-value confidence-high">0.89</div>
                <div class="forecast-change">Coefficient of Determination</div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab6:
        st.markdown('<div class="tab-section-header"><h3>Comprehensive Forecast Summary</h3></div>', unsafe_allow_html=True)
        
        if forecast_results:
            # Summary table
            summary_data = []
            
            for forecast_type, forecast_result in forecast_results.items():
                if forecast_result and 'forecast' in forecast_result:
                    forecast_data = forecast_result['forecast']
                    metadata = forecast_result.get('metadata', {})
                    
                    if forecast_data.get('values'):
                        current_value = forecast_data['values'][0]
                        future_value = forecast_data['values'][-1] if len(forecast_data['values']) > 1 else current_value
                        change_pct = ((future_value - current_value) / current_value * 100) if current_value > 0 else 0
                        
                        # Format values appropriately based on type
                        if 'production' in forecast_type or 'demand' in forecast_type:
                            current_formatted = format_large_number(current_value)
                            future_formatted = format_large_number(future_value)
                        elif 'price' in forecast_type:
                            current_formatted = f"${current_value:.2f}"
                            future_formatted = f"${future_value:.2f}"
                        else:
                            current_formatted = f"{current_value:,.1f}"
                            future_formatted = f"{future_value:,.1f}"
                        
                        summary_data.append({
                            'Forecast Type': forecast_type.replace('_', ' ').title(),
                            'Current Value': current_formatted,
                            'Forecast Value': future_formatted,
                            'Change (%)': f"{change_pct:+.1f}%",
                            'Confidence': f"{metadata.get('confidence', 0.8):.1%}",
                            'Horizon (Days)': metadata.get('horizon_days', forecast_horizon)
                        })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, width='stretch')
            
            # Export forecast results
            if st.button("Export Forecast Results"):
                # Create export data
                export_data = {
                    'forecast_summary': summary_data,
                    'generation_time': datetime.now().isoformat(),
                    'parameters': {
                        'horizon_days': forecast_horizon,
                        'confidence_level': confidence_level,
                        'forecast_types': selected_forecast_types
                    }
                }
                
                st.download_button(
                    label="Download Forecast Summary (JSON)",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"forecast_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        else:
            st.info("Generate forecasts using the sidebar controls to see summary results here.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem; padding: 1rem;'>
        Oil & Gas Forecasting Dashboard | Last Updated: {}
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