#!/usr/bin/env python3
"""
Simple Oil & Gas Price Prediction API for Domino Model Endpoints
Simplified version with mock price forecasts
"""

import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict_prices(commodity="crude_oil", forecast_days=30, start_date=None):
    """
    Price prediction function for Domino Model API
    
    Args:
        commodity (str): "crude_oil" or "natural_gas"
        forecast_days (int): Days to forecast (1-365)
        start_date (str): Start date YYYY-MM-DD (optional)
    
    Returns:
        dict: Price forecast results
    """
    try:
        # Validate inputs
        forecast_days = int(forecast_days)
        if forecast_days < 1 or forecast_days > 365:
            raise ValueError("forecast_days must be between 1 and 365")
        
        if commodity not in ['crude_oil', 'natural_gas']:
            raise ValueError("commodity must be 'crude_oil' or 'natural_gas'")
        
        # Parse start date or use tomorrow
        if start_date is None:
            start_dt = datetime.now() + timedelta(days=1)
        else:
            start_dt = datetime.strptime(str(start_date), '%Y-%m-%d')
        
        # Generate forecast dates
        forecast_dates = [
            (start_dt + timedelta(days=i)).strftime('%Y-%m-%d') 
            for i in range(forecast_days)
        ]
        
        # Set base price and volatility by commodity
        if commodity == 'crude_oil':
            base_price = 75.0  # USD per barrel
            volatility = 0.025  # 2.5% daily volatility
            unit = 'USD/barrel'
            target_price = 75.0
        else:  # natural_gas
            base_price = 3.8   # USD per MMBtu
            volatility = 0.04   # 4% daily volatility
            unit = 'USD/MMBtu'
            target_price = 3.8
        
        # Generate price predictions
        predictions = []
        current_price = base_price
        
        for i, date in enumerate(forecast_dates):
            # Price movement factors
            # 1. Mean reversion to target price
            mean_reversion = 0.05 * (target_price - current_price) / target_price
            
            # 2. Seasonal effects (for natural gas)
            if commodity == 'natural_gas':
                month = (start_dt + timedelta(days=i)).month
                if month in [12, 1, 2]:  # Winter
                    seasonal_factor = 0.15
                elif month in [6, 7, 8]:  # Summer
                    seasonal_factor = -0.1
                else:
                    seasonal_factor = 0.0
            else:
                seasonal_factor = 0.02 * np.sin(2 * np.pi * i / 365)  # Mild seasonal variation
            
            # 3. Random volatility
            volatility_factor = np.random.normal(0, volatility)
            
            # Calculate price change
            price_change = mean_reversion + seasonal_factor + volatility_factor
            current_price = max(0.1, current_price * (1 + price_change))
            
            # Calculate confidence intervals
            confidence_width = volatility * 1.96  # 95% confidence
            lower_bound = max(0.1, current_price * (1 - confidence_width))
            upper_bound = current_price * (1 + confidence_width)
            
            predictions.append({
                'date': date,
                'commodity': commodity,
                'predicted_price': round(current_price, 2),
                'unit': unit,
                'confidence_lower': round(lower_bound, 2),
                'confidence_upper': round(upper_bound, 2),
                'volatility': round(volatility * 100, 1)  # As percentage
            })
        
        # Calculate summary statistics
        prices = [p['predicted_price'] for p in predictions]
        price_change_pct = ((prices[-1] - prices[0]) / prices[0]) * 100
        
        # Prepare response
        response = {
            'model_info': {
                'name': 'simple_price_predictor',
                'version': '1.0',
                'type': 'mock_price_model'
            },
            'input_parameters': {
                'commodity': commodity,
                'forecast_days': forecast_days,
                'start_date': start_date or start_dt.strftime('%Y-%m-%d')
            },
            'predictions': predictions,
            'summary': {
                'commodity': commodity,
                'unit': unit,
                'forecast_period': f"{forecast_dates[0]} to {forecast_dates[-1]}",
                'starting_price': round(prices[0], 2),
                'ending_price': round(prices[-1], 2),
                'price_change_percent': round(price_change_pct, 2),
                'average_price': round(np.mean(prices), 2),
                'max_price': round(max(prices), 2),
                'min_price': round(min(prices), 2),
                'price_volatility': round(np.std(prices), 2)
            },
            'metadata': {
                'prediction_timestamp': datetime.now().isoformat(),
                'api_version': '1.0'
            }
        }
        
        logger.info(f"Generated price forecast for {commodity} over {forecast_days} days")
        return response
        
    except Exception as e:
        logger.error(f"Error in price prediction: {e}")
        return {
            'error': str(e),
            'input_parameters': {
                'commodity': commodity,
                'forecast_days': forecast_days,
                'start_date': start_date
            },
            'timestamp': datetime.now().isoformat()
        }

# Example usage for testing
if __name__ == "__main__":
    print("=== Simple Price Prediction API Test ===")
    
    # Test 1: Crude oil forecast
    print("\n1. 30-day crude oil forecast:")
    result1 = predict_prices()
    print(f"Summary: {result1['summary']}")
    
    # Test 2: Natural gas forecast
    print("\n2. 14-day natural gas forecast:")
    result2 = predict_prices(commodity="natural_gas", forecast_days=14)
    print(f"Summary: {result2['summary']}")
    
    # Test 3: With start date
    print("\n3. Oil forecast from December:")
    result3 = predict_prices(commodity="crude_oil", forecast_days=21, start_date="2024-12-01")
    print(f"Summary: {result3['summary']}")