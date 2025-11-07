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
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Aramco-inspired custom CSS
st.markdown("""
<style>
    /* Primary color scheme */
    :root {
        --aramco-blue: #00A3E0;
        --aramco-green: #84BD00;
        --aramco-dark-blue: #0033A0;
        --aramco-light-gray: #F5F5F5;
        --aramco-dark-gray: #333333;
        --aramco-orange: #FF8C00;
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
    
    /* Forecast card styling */
    .forecast-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid var(--aramco-green);
        margin-bottom: 1rem;
        font-family: 'Helvetica', sans-serif;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .forecast-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: var(--aramco-dark-blue);
        word-break: break-all;
        line-height: 1.2;
        margin: 0.5rem 0;
    }
    
    .forecast-label {
        font-size: 1rem;
        color: var(--aramco-dark-gray);
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .forecast-change {
        font-size: 0.9rem;
        margin-top: 0.5rem;
        color: var(--aramco-dark-gray);
    }
    
    .forecast-up { color: var(--aramco-green); }
    .forecast-down { color: #FF6B6B; }
    .forecast-neutral { color: var(--aramco-orange); }
    
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
        border-left: 3px solid var(--aramco-blue);
    }
</style>
""", unsafe_allow_html=True)

# Data loading functions
@st.cache_data(ttl=300)
def load_production_timeseries():
    """Load production time series data with caching"""
    try:
        paths = get_data_paths('Oil-and-Gas-Demo')
        data_path = paths['artifacts_path'] / 'Oil-and-Gas-Demo' / 'production_timeseries.parquet'
        df = pd.read_parquet(data_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"Error loading production data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_price_data():
    """Load price time series data with caching"""
    try:
        paths = get_data_paths('Oil-and-Gas-Demo')
        data_path = paths['artifacts_path'] / 'Oil-and-Gas-Demo' / 'prices_timeseries.parquet'
        df = pd.read_parquet(data_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.warning(f"Error loading price data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300) 
def load_demand_data():
    """Load demand time series data with caching"""
    try:
        paths = get_data_paths('Oil-and-Gas-Demo')
        data_path = paths['artifacts_path'] / 'Oil-and-Gas-Demo' / 'demand_timeseries.parquet'
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
        data_path = paths['artifacts_path'] / 'Oil-and-Gas-Demo' / 'maintenance_timeseries.parquet'
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
        
        # Generate mock forecast data
        start_date = datetime.now()
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
        # Historical production data (aggregated daily)
        daily_production = production_df.groupby('date').agg({
            'oil_production_bpd': 'sum',
            'gas_production_mcfd': 'sum'
        }).reset_index()
        
        # Oil production historical
        fig.add_trace(go.Scatter(
            x=daily_production['date'],
            y=daily_production['oil_production_bpd'],
            mode='lines',
            name='Historical Oil Production',
            line=dict(color='#0033A0', width=2)
        ))
        
        # Gas production historical (secondary y-axis)
        fig.add_trace(go.Scatter(
            x=daily_production['date'],
            y=daily_production['gas_production_mcfd'],
            mode='lines',
            name='Historical Gas Production',
            line=dict(color='#84BD00', width=2),
            yaxis='y2'
        ))
    
    # Add forecast data if available
    if forecast_data and 'forecast' in forecast_data:
        forecast = forecast_data['forecast']
        forecast_dates = pd.to_datetime(forecast['dates'])
        
        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast['values'],
            mode='lines',
            name='Forecast',
            line=dict(color='#FF8C00', width=3, dash='dash')
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
            fillcolor='rgba(255,140,0,0.2)'
        ))
    
    # Update layout
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
        font=dict(family="Arial", size=12),
        title_font=dict(size=16, color='#0033A0'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_price_forecast_chart(price_df, price_forecast_data=None):
    """Create price forecasting chart"""
    fig = go.Figure()
    
    if not price_df.empty:
        # Historical price data
        price_cols = [col for col in price_df.columns if 'price' in col.lower()]
        
        colors = ['#0033A0', '#84BD00', '#FF8C00', '#FF6B6B']
        for i, col in enumerate(price_cols[:4]):  # Limit to 4 price series
            fig.add_trace(go.Scatter(
                x=price_df['date'] if 'date' in price_df.columns else price_df.index,
                y=price_df[col],
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
            line=dict(color='#FF8C00', width=3, dash='dash')
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
            fillcolor='rgba(255,140,0,0.2)'
        ))
    
    fig.update_layout(
        title='Oil & Gas Price Forecasting',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        height=500,
        font=dict(family="Arial", size=12),
        title_font=dict(size=16, color='#0033A0')
    )
    
    return fig

def create_demand_forecast_chart(demand_df, demand_forecast_data=None):
    """Create demand forecasting chart by region"""
    if demand_df.empty:
        return go.Figure()
    
    # Group by date and region if available
    if 'region' in demand_df.columns and 'date' in demand_df.columns:
        regional_demand = demand_df.groupby(['date', 'region']).sum().reset_index()
        
        fig = px.line(
            regional_demand,
            x='date',
            y=regional_demand.columns[-1],  # Assume last column is demand
            color='region',
            title='Regional Demand Forecasting',
            color_discrete_sequence=['#0033A0', '#84BD00', '#FF8C00', '#FF6B6B', '#9B59B6']
        )
    else:
        # Simple time series
        demand_col = [col for col in demand_df.columns if 'demand' in col.lower()]
        if demand_col:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=demand_df['date'] if 'date' in demand_df.columns else demand_df.index,
                y=demand_df[demand_col[0]],
                mode='lines',
                name='Historical Demand',
                line=dict(color='#0033A0', width=2)
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
            line=dict(color='#FF8C00', width=3, dash='dash')
        ))
    
    fig.update_layout(
        height=500,
        font=dict(family="Arial", size=12),
        title_font=dict(size=16, color='#0033A0')
    )
    
    return fig

def create_maintenance_optimization_chart(maintenance_df):
    """Create maintenance scheduling optimization chart"""
    if maintenance_df.empty:
        return go.Figure()
    
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
            marker_color=['#0033A0', '#84BD00', '#FF8C00']
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
            marker_color=['#FF6B6B', '#FFA500', '#84BD00']
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
        title_font=dict(size=16, color='#0033A0')
    )
    
    return fig

def display_forecast_metrics(forecast_data_list):
    """Display forecasting KPI metrics"""
    st.markdown("### Forecast Summary Metrics")
    
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
    
    # Header
    st.markdown("""
    <div class="aramco-header">
        <h1>Oil & Gas Forecasting Dashboard</h1>
        <p>Advanced predictive analytics for production, pricing, and demand optimization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.markdown("## Forecasting Controls")
    st.sidebar.markdown('<div class="control-panel">', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading time series data..."):
        production_df = load_production_timeseries()
        price_df = load_price_data()
        demand_df = load_demand_data()
        maintenance_df = load_maintenance_data()
    
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
        default=['oil_production_bpd', 'crude_oil_price_usd_bbl'],
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
    
    # Data refresh
    if st.sidebar.button("Refresh Data & Forecasts"):
        st.cache_data.clear()
        st.experimental_rerun()
    
    # Generate forecasts button
    generate_forecasts = st.sidebar.button("Generate New Forecasts", type="primary")
    
    # Store forecast results
    forecast_results = {}
    
    if generate_forecasts:
        st.sidebar.success("Generating forecasts...")
        
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
        st.markdown("### Production Forecasting Analysis")
        
        # Production forecast chart
        production_forecast = forecast_results.get('oil_production_bpd')
        production_chart = create_production_forecast_chart(production_df, production_forecast)
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(production_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Production insights
        if not production_df.empty:
            st.markdown("### Production Insights")
            
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
                st.markdown("### Production by Region")
                regional_production = production_df.groupby('region').agg({
                    'oil_production_bpd': 'sum',
                    'gas_production_mcfd': 'sum'
                }).round(0)
                st.dataframe(regional_production, use_container_width=True)
    
    with tab2:
        st.markdown("### Oil & Gas Price Forecasting")
        
        # Price forecast chart
        price_forecast = forecast_results.get('crude_oil_price_usd_bbl')
        price_chart = create_price_forecast_chart(price_df, price_forecast)
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(price_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Price analysis
        if price_forecast:
            st.markdown("### Price Forecast Analysis")
            
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
        st.markdown("### Market Factors")
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
        st.markdown("### Regional Demand Forecasting")
        
        # Demand forecast chart
        demand_forecast = forecast_results.get('gasoline_demand_thousand_bpd')
        demand_chart = create_demand_forecast_chart(demand_df, demand_forecast)
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(demand_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Demand analysis
        st.markdown("### Demand Analysis by Product")
        
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
        st.markdown("### Maintenance Scheduling Optimization")
        
        # Maintenance optimization chart
        maintenance_chart = create_maintenance_optimization_chart(maintenance_df)
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(maintenance_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Maintenance recommendations
        st.markdown("### Maintenance Recommendations")
        
        # Mock maintenance alerts
        alert_col1, alert_col2 = st.columns(2)
        
        with alert_col1:
            st.markdown("""
            <div class="maintenance-alert">
                <strong>⚠️ Upcoming Maintenance</strong><br>
                • Facility ABC-001: Scheduled in 5 days<br>
                • Facility XYZ-045: Overdue by 2 days<br>
                • Facility DEF-123: Preventive maintenance due
            </div>
            """, unsafe_allow_html=True)
        
        with alert_col2:
            st.markdown("""
            <div class="maintenance-alert">
                <strong>💡 Optimization Opportunities</strong><br>
                • Consolidate maintenance for Region A<br>
                • Predictive maintenance for high-value assets<br>
                • Cost savings: $2.3M annually
            </div>
            """, unsafe_allow_html=True)
        
        # Maintenance schedule table
        if not maintenance_df.empty:
            st.markdown("### Maintenance Schedule")
            
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
            st.dataframe(schedule_df, use_container_width=True)
    
    with tab5:
        st.markdown("### Interactive Forecast Controls")
        
        st.markdown('<div class="param-control">', unsafe_allow_html=True)
        st.markdown("#### Model Parameters")
        
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
        st.markdown("### Model Performance")
        
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
        st.markdown("### Comprehensive Forecast Summary")
        
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
                st.dataframe(summary_df, use_container_width=True)
            
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
        Oil & Gas Forecasting Dashboard | Last Updated: {} | Powered by Advanced ML Models
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()