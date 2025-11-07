"""
Domino Model API Endpoint for Oil & Gas Forecasting Dashboard
Provides production, price, and demand forecasting capabilities
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/mnt/code')

class OilGasForecastingModel:
    def __init__(self):
        """Initialize the forecasting models"""
        self.models = {}
        self.forecasts_cache = {}
        self.load_models()
        self.load_precomputed_forecasts()
    
    def load_models(self):
        """Load trained forecasting models from artifacts"""
        models_path = Path('/mnt/artifacts/models/forecasting')
        
        try:
            # Load available model files
            if models_path.exists():
                model_files = list(models_path.glob('*.pkl'))
                for model_file in model_files:
                    model_name = model_file.stem
                    try:
                        self.models[model_name] = joblib.load(model_file)
                        print(f"Loaded {model_name}")
                    except Exception as e:
                        print(f"Error loading {model_name}: {e}")
            
            print(f"Loaded {len(self.models)} forecasting models")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            self.create_fallback_models()
    
    def load_precomputed_forecasts(self):
        """Load precomputed forecast data"""
        forecasts_path = Path('/mnt/artifacts/models/forecasting')
        
        try:
            if forecasts_path.exists():
                forecast_files = list(forecasts_path.glob('*_forecast.csv'))
                for forecast_file in forecast_files:
                    forecast_name = forecast_file.stem.replace('_forecast', '')
                    try:
                        df = pd.read_csv(forecast_file)
                        self.forecasts_cache[forecast_name] = df
                        print(f"Loaded precomputed forecasts for {forecast_name}")
                    except Exception as e:
                        print(f"Error loading forecast {forecast_name}: {e}")
            
        except Exception as e:
            print(f"Error loading precomputed forecasts: {e}")
    
    def create_fallback_models(self):
        """Create simple fallback models for demonstration"""
        print("Creating fallback forecasting models")
        
        # Create synthetic time series data for fallback
        dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D')
        
        # Oil production trend with seasonality
        oil_trend = 100 + np.random.normal(0, 5, len(dates))
        oil_seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
        oil_data = oil_trend + oil_seasonal + np.random.normal(0, 2, len(dates))
        
        # Price data with volatility
        price_trend = 70 + np.random.normal(0, 8, len(dates))
        price_data = np.maximum(30, price_trend + np.random.normal(0, 5, len(dates)))
        
        # Create fallback forecasts
        future_dates = pd.date_range(start='2025-01-01', periods=180, freq='D')
        
        self.forecasts_cache = {
            'oil_production': pd.DataFrame({
                'ds': future_dates,
                'yhat': 100 + np.random.normal(0, 3, 180),
                'yhat_lower': 95 + np.random.normal(0, 2, 180),
                'yhat_upper': 105 + np.random.normal(0, 2, 180)
            }),
            'crude_oil_price': pd.DataFrame({
                'ds': future_dates,
                'yhat': 70 + np.random.normal(0, 5, 180),
                'yhat_lower': 65 + np.random.normal(0, 3, 180),
                'yhat_upper': 75 + np.random.normal(0, 3, 180)
            })
        }

# Global model instance
model = OilGasForecastingModel()

def predict(data):
    """
    Main prediction function for Domino Model API
    
    Parameters:
    data (dict): Input data with forecast parameters
    
    Returns:
    dict: Forecast predictions with confidence intervals
    """
    
    try:
        # Extract forecast parameters
        forecast_type = data.get('forecast_type', 'oil_production')
        horizon_days = data.get('horizon_days', 30)
        start_date = data.get('start_date', datetime.now().strftime('%Y-%m-%d'))
        
        # Generate forecast
        forecast = generate_forecast(forecast_type, horizon_days, start_date)
        
        return {
            "status": "success",
            "forecast": forecast,
            "metadata": {
                "forecast_type": forecast_type,
                "horizon_days": horizon_days,
                "start_date": start_date,
                "generated_at": datetime.now().isoformat(),
                "model_version": "1.0"
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "forecast": None
        }

def generate_forecast(forecast_type, horizon_days, start_date):
    """
    Generate forecast for specified type and horizon
    
    Parameters:
    forecast_type (str): Type of forecast (oil_production, crude_oil_price, etc.)
    horizon_days (int): Number of days to forecast
    start_date (str): Start date for forecast
    
    Returns:
    dict: Forecast data with dates, values, and confidence intervals
    """
    
    # Parse start date
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    forecast_dates = [start_dt + timedelta(days=i) for i in range(horizon_days)]
    
    # Check if we have precomputed forecasts
    if forecast_type in model.forecasts_cache:
        df = model.forecasts_cache[forecast_type]
        
        # Filter by date range if available
        if 'ds' in df.columns:
            df['ds'] = pd.to_datetime(df['ds'])
            mask = (df['ds'] >= start_dt) & (df['ds'] < start_dt + timedelta(days=horizon_days))
            filtered_df = df[mask].head(horizon_days)
            
            if len(filtered_df) > 0:
                return {
                    "dates": filtered_df['ds'].dt.strftime('%Y-%m-%d').tolist(),
                    "values": filtered_df['yhat'].round(2).tolist(),
                    "lower_bound": filtered_df.get('yhat_lower', filtered_df['yhat'] * 0.9).round(2).tolist(),
                    "upper_bound": filtered_df.get('yhat_upper', filtered_df['yhat'] * 1.1).round(2).tolist(),
                    "source": "precomputed"
                }
    
    # Generate synthetic forecast if no precomputed data available
    return generate_synthetic_forecast(forecast_type, forecast_dates)

def generate_synthetic_forecast(forecast_type, forecast_dates):
    """Generate synthetic forecast data for demonstration"""
    
    horizon_days = len(forecast_dates)
    
    # Base parameters by forecast type
    forecast_configs = {
        'oil_production_bpd': {
            'base_value': 50000,
            'trend': -0.1,  # Slight decline
            'volatility': 0.05,
            'seasonal_amplitude': 0.1
        },
        'gas_production_mcfd': {
            'base_value': 150000,
            'trend': 0.05,  # Slight growth
            'volatility': 0.08,
            'seasonal_amplitude': 0.15
        },
        'crude_oil_price_usd_bbl': {
            'base_value': 75.0,
            'trend': 0.02,
            'volatility': 0.12,
            'seasonal_amplitude': 0.05
        },
        'natural_gas_price_usd_mmbtu': {
            'base_value': 3.5,
            'trend': 0.03,
            'volatility': 0.15,
            'seasonal_amplitude': 0.2
        },
        'brent_crude_usd_bbl': {
            'base_value': 77.0,
            'trend': 0.02,
            'volatility': 0.11,
            'seasonal_amplitude': 0.04
        }
    }
    
    # Default configuration
    config = forecast_configs.get(forecast_type, {
        'base_value': 100.0,
        'trend': 0.0,
        'volatility': 0.05,
        'seasonal_amplitude': 0.1
    })
    
    # Generate forecast values
    values = []
    for i, date in enumerate(forecast_dates):
        # Trend component
        trend_value = config['base_value'] * (1 + config['trend'] * i / 365)
        
        # Seasonal component
        seasonal_value = config['base_value'] * config['seasonal_amplitude'] * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
        
        # Random component
        random_value = np.random.normal(0, config['base_value'] * config['volatility'])
        
        # Combine components
        forecast_value = max(0, trend_value + seasonal_value + random_value)
        values.append(forecast_value)
    
    # Calculate confidence intervals
    values = np.array(values)
    lower_bound = values * 0.9  # 10% lower
    upper_bound = values * 1.1  # 10% higher
    
    return {
        "dates": [date.strftime('%Y-%m-%d') for date in forecast_dates],
        "values": values.round(2).tolist(),
        "lower_bound": lower_bound.round(2).tolist(),
        "upper_bound": upper_bound.round(2).tolist(),
        "source": "synthetic"
    }

def get_available_forecasts():
    """Get list of available forecast types"""
    
    available_types = [
        'oil_production_bpd',
        'gas_production_mcfd', 
        'crude_oil_price_usd_bbl',
        'natural_gas_price_usd_mmbtu',
        'brent_crude_usd_bbl',
        'wti_crude_usd_bbl',
        'gasoline_demand_thousand_bpd',
        'diesel_demand_thousand_bpd'
    ]
    
    # Add types from precomputed forecasts
    available_types.extend(list(model.forecasts_cache.keys()))
    
    return list(set(available_types))

def multi_forecast(data):
    """
    Generate multiple forecasts in a single request
    
    Parameters:
    data (dict): Contains list of forecast requests
    
    Returns:
    dict: Multiple forecast results
    """
    
    try:
        requests = data.get('requests', [])
        forecasts = {}
        
        for request in requests:
            forecast_id = request.get('id', f"forecast_{len(forecasts)}")
            forecast_type = request.get('forecast_type', 'oil_production_bpd')
            horizon_days = request.get('horizon_days', 30)
            start_date = request.get('start_date', datetime.now().strftime('%Y-%m-%d'))
            
            forecast = generate_forecast(forecast_type, horizon_days, start_date)
            forecasts[forecast_id] = {
                "forecast_type": forecast_type,
                "forecast": forecast
            }
        
        return {
            "status": "success",
            "forecasts": forecasts,
            "total_forecasts": len(forecasts)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "forecasts": None
        }

# Health check function for Domino
def health_check():
    """Health check endpoint for Domino monitoring"""
    return {
        "status": "healthy",
        "models_loaded": len(model.models),
        "cached_forecasts": len(model.forecasts_cache),
        "available_forecast_types": get_available_forecasts(),
        "version": "1.0"
    }

# Example usage for testing
if __name__ == "__main__":
    # Test single forecast
    test_data = {
        "forecast_type": "crude_oil_price_usd_bbl",
        "horizon_days": 30,
        "start_date": "2025-01-01"
    }
    
    result = predict(test_data)
    print("Single forecast test:")
    print(json.dumps(result, indent=2, default=str))
    
    # Test multi-forecast
    multi_test_data = {
        "requests": [
            {
                "id": "oil_forecast",
                "forecast_type": "oil_production_bpd",
                "horizon_days": 30,
                "start_date": "2025-01-01"
            },
            {
                "id": "price_forecast", 
                "forecast_type": "crude_oil_price_usd_bbl",
                "horizon_days": 60,
                "start_date": "2025-01-01"
            }
        ]
    }
    
    multi_result = multi_forecast(multi_test_data)
    print("\nMulti-forecast test:")
    print(json.dumps(multi_result, indent=2, default=str))
    
    # Test health check
    health = health_check()
    print("\nHealth check:")
    print(json.dumps(health, indent=2))