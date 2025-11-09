#!/usr/bin/env python3
"""
Simple Oil & Gas Demand Forecast API for Domino Model Endpoints
Simplified version with mock demand forecasts
"""

import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict_demand(region="global", commodity="crude_oil", forecast_days=30, start_date=None):
    """
    Demand prediction function for Domino Model API
    
    Args:
        region (str): Geographic region or "global"
        commodity (str): "crude_oil", "natural_gas", or "refined_products"
        forecast_days (int): Days to forecast (1-365)
        start_date (str): Start date YYYY-MM-DD (optional)
    
    Returns:
        dict: Demand forecast results
    """
    try:
        # Validate inputs
        forecast_days = int(forecast_days)
        if forecast_days < 1 or forecast_days > 365:
            raise ValueError("forecast_days must be between 1 and 365")
        
        valid_regions = ['north_america', 'europe', 'asia_pacific', 'middle_east', 'africa', 'south_america', 'global']
        if region not in valid_regions:
            raise ValueError(f"region must be one of: {', '.join(valid_regions)}")
        
        valid_commodities = ['crude_oil', 'natural_gas', 'refined_products', 'gasoline', 'diesel']
        if commodity not in valid_commodities:
            raise ValueError(f"commodity must be one of: {', '.join(valid_commodities)}")
        
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
        
        # Set base demand by region and commodity (in thousands of units per day)
        base_demands = {
            ('global', 'crude_oil'): 100000,  # 100M bpd
            ('global', 'natural_gas'): 400000,  # 400M mcfd
            ('global', 'refined_products'): 80000,  # 80M bpd
            ('north_america', 'crude_oil'): 20000,
            ('north_america', 'natural_gas'): 80000,
            ('europe', 'crude_oil'): 15000,
            ('asia_pacific', 'crude_oil'): 35000,
            ('middle_east', 'crude_oil'): 8000,
        }
        
        # Get base demand or calculate from global if not specified
        base_demand = base_demands.get((region, commodity))
        if base_demand is None:
            global_demand = base_demands.get(('global', commodity), 50000)
            regional_factors = {
                'north_america': 0.22, 'europe': 0.16, 'asia_pacific': 0.38,
                'middle_east': 0.08, 'africa': 0.06, 'south_america': 0.10
            }
            base_demand = global_demand * regional_factors.get(region, 0.15)
        
        # Set units
        if commodity == 'natural_gas':
            unit = 'thousand_mcfd'
        else:
            unit = 'thousand_bpd'
        
        # Growth rates by region (annual)
        growth_rates = {
            'north_america': 0.015,  # 1.5%
            'europe': 0.01,          # 1.0%
            'asia_pacific': 0.03,    # 3.0%
            'middle_east': 0.025,    # 2.5%
            'africa': 0.035,         # 3.5%
            'south_america': 0.02,   # 2.0%
            'global': 0.02           # 2.0%
        }
        
        annual_growth = growth_rates.get(region, 0.02)
        daily_growth = annual_growth / 365
        
        # Generate demand predictions
        predictions = []
        current_demand = base_demand
        
        for i, date in enumerate(forecast_dates):
            # Demand factors
            # 1. Economic growth trend
            growth_factor = 1 + daily_growth
            
            # 2. Seasonal patterns
            month = (start_dt + timedelta(days=i)).month
            if commodity == 'natural_gas':
                # Winter heating demand
                if month in [12, 1, 2]:
                    seasonal_factor = 1.15
                elif month in [6, 7, 8]:
                    seasonal_factor = 0.90
                else:
                    seasonal_factor = 1.0
            elif commodity in ['gasoline', 'refined_products']:
                # Summer driving season
                if month in [6, 7, 8]:
                    seasonal_factor = 1.08
                else:
                    seasonal_factor = 1.0
            else:
                seasonal_factor = 1.0
            
            # 3. Economic cycle effects
            if start_dt.year >= 2025:
                economic_factor = 1.005  # Economic recovery
            else:
                economic_factor = 1.0
            
            # 4. Daily variation
            daily_variation = np.random.normal(1.0, 0.02)  # ±2% daily variation
            
            # Calculate demand
            current_demand *= growth_factor
            predicted_demand = current_demand * seasonal_factor * economic_factor * daily_variation
            predicted_demand = max(0, predicted_demand)
            
            # Calculate confidence intervals
            confidence_width = 0.1  # ±10% confidence range
            lower_bound = predicted_demand * (1 - confidence_width)
            upper_bound = predicted_demand * (1 + confidence_width)
            
            predictions.append({
                'date': date,
                'region': region,
                'commodity': commodity,
                'predicted_demand': round(predicted_demand, 1),
                'unit': unit,
                'confidence_lower': round(lower_bound, 1),
                'confidence_upper': round(upper_bound, 1)
            })
        
        # Calculate summary statistics
        demands = [p['predicted_demand'] for p in predictions]
        demand_change_pct = ((demands[-1] - demands[0]) / demands[0]) * 100
        
        # Prepare response
        response = {
            'model_info': {
                'name': 'simple_demand_predictor',
                'version': '1.0',
                'type': 'mock_demand_model'
            },
            'input_parameters': {
                'region': region,
                'commodity': commodity,
                'forecast_days': forecast_days,
                'start_date': start_date or start_dt.strftime('%Y-%m-%d')
            },
            'predictions': predictions,
            'summary': {
                'region': region,
                'commodity': commodity,
                'unit': unit,
                'forecast_period': f"{forecast_dates[0]} to {forecast_dates[-1]}",
                'starting_demand': round(demands[0], 1),
                'ending_demand': round(demands[-1], 1),
                'demand_change_percent': round(demand_change_pct, 2),
                'average_daily_demand': round(np.mean(demands), 1),
                'peak_demand': round(max(demands), 1),
                'minimum_demand': round(min(demands), 1),
                'total_forecast_demand': round(sum(demands), 1),
                'annual_growth_rate': round(annual_growth * 100, 1)
            },
            'metadata': {
                'prediction_timestamp': datetime.now().isoformat(),
                'api_version': '1.0'
            }
        }
        
        logger.info(f"Generated demand forecast for {commodity} in {region} over {forecast_days} days")
        return response
        
    except Exception as e:
        logger.error(f"Error in demand prediction: {e}")
        return {
            'error': str(e),
            'input_parameters': {
                'region': region,
                'commodity': commodity,
                'forecast_days': forecast_days,
                'start_date': start_date
            },
            'timestamp': datetime.now().isoformat()
        }

# Example usage for testing
if __name__ == "__main__":
    print("=== Simple Demand Prediction API Test ===")
    
    # Test 1: Global crude oil demand
    print("\n1. 30-day global crude oil demand:")
    result1 = predict_demand()
    print(f"Summary: {result1['summary']}")
    
    # Test 2: North America natural gas demand
    print("\n2. 14-day North America natural gas demand:")
    result2 = predict_demand(region="north_america", commodity="natural_gas", forecast_days=14)
    print(f"Summary: {result2['summary']}")
    
    # Test 3: Asia Pacific refined products
    print("\n3. Asia Pacific refined products demand:")
    result3 = predict_demand(region="asia_pacific", commodity="refined_products", forecast_days=21)
    print(f"Summary: {result3['summary']}")