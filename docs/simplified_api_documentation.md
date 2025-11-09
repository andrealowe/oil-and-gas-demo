# Simplified Oil & Gas Forecasting API Documentation

This document provides documentation for the simplified Oil & Gas Forecasting API endpoints designed for Domino Model API deployment. All endpoints accept individual parameters directly and return JSON responses.

## API Endpoints Overview

| Endpoint | Purpose | API File | Function |
|----------|---------|----------|----------|
| `/predict_production` | Production Forecasting | `simple_prediction_api.py` | `predict_production()` |
| `/predict_prices` | Price Forecasting | `simple_price_api.py` | `predict_prices()` |
| `/predict_demand` | Demand Forecasting | `simple_demand_api.py` | `predict_demand()` |

---

## 1. Production Forecasting API

**Endpoint**: `/predict_production`  
**File**: `/mnt/code/src/api/simple_prediction_api.py`  
**Function**: `predict_production(facility_id, forecast_days, start_date)`

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `facility_id` | string | No | `"all_facilities"` | Facility ID or `"all_facilities"` for aggregate |
| `forecast_days` | integer | No | `7` | Number of days to forecast (1-365) |
| `start_date` | string | No | `tomorrow` | Start date in YYYY-MM-DD format |

### Example Requests

#### Basic Request (Domino JSON format)
```json
{
  "data": {
    "facility_id": "all_facilities",
    "forecast_days": 7
  }
}
```

#### Specific Facility Request
```json
{
  "data": {
    "facility_id": "WELL_000123",
    "forecast_days": 14,
    "start_date": "2024-12-01"
  }
}
```

#### Long-term Forecast
```json
{
  "data": {
    "forecast_days": 90
  }
}
```

### Response Format
```json
{
  "model_info": {
    "name": "simple_production_predictor",
    "version": "1.0",
    "type": "mock_forecast_model"
  },
  "input_parameters": {
    "facility_id": "all_facilities",
    "forecast_days": 7,
    "start_date": "2025-11-09"
  },
  "predictions": [
    {
      "date": "2025-11-09",
      "facility_id": "all_facilities",
      "predicted_production_bpd": 52341.7,
      "confidence_lower": 47107.5,
      "confidence_upper": 57575.9
    }
  ],
  "summary": {
    "facility_name": "All Facilities",
    "forecast_period": "2025-11-09 to 2025-11-15",
    "total_days": 7,
    "average_daily_production": 50247.01,
    "total_forecasted_production": 351729.04,
    "starting_production": 50135.64,
    "ending_production": 51369.16,
    "production_trend": 2.46
  }
}
```

---

## 2. Price Forecasting API

**Endpoint**: `/predict_prices`  
**File**: `/mnt/code/src/api/simple_price_api.py`  
**Function**: `predict_prices(commodity, forecast_days, start_date)`

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `commodity` | string | No | `"crude_oil"` | `"crude_oil"` or `"natural_gas"` |
| `forecast_days` | integer | No | `30` | Number of days to forecast (1-365) |
| `start_date` | string | No | `tomorrow` | Start date in YYYY-MM-DD format |

### Example Requests

#### Basic Crude Oil Forecast
```json
{
  "data": {
    "commodity": "crude_oil",
    "forecast_days": 30
  }
}
```

#### Natural Gas Forecast
```json
{
  "data": {
    "commodity": "natural_gas",
    "forecast_days": 14,
    "start_date": "2024-12-01"
  }
}
```

#### Long-term Oil Forecast
```json
{
  "data": {
    "commodity": "crude_oil",
    "forecast_days": 180
  }
}
```

### Response Format
```json
{
  "model_info": {
    "name": "simple_price_predictor",
    "version": "1.0",
    "type": "mock_price_model"
  },
  "input_parameters": {
    "commodity": "crude_oil",
    "forecast_days": 30,
    "start_date": "2025-11-09"
  },
  "predictions": [
    {
      "date": "2025-11-09",
      "commodity": "crude_oil",
      "predicted_price": 75.23,
      "unit": "USD/barrel",
      "confidence_lower": 71.47,
      "confidence_upper": 78.99,
      "volatility": 2.5
    }
  ],
  "summary": {
    "commodity": "crude_oil",
    "unit": "USD/barrel",
    "forecast_period": "2025-11-09 to 2025-12-08",
    "starting_price": 74.25,
    "ending_price": 82.29,
    "price_change_percent": 10.83,
    "average_price": 77.53,
    "max_price": 84.93,
    "min_price": 72.42,
    "price_volatility": 3.54
  }
}
```

---

## 3. Demand Forecasting API

**Endpoint**: `/predict_demand`  
**File**: `/mnt/code/src/api/simple_demand_api.py`  
**Function**: `predict_demand(region, commodity, forecast_days, start_date)`

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `region` | string | No | `"global"` | Geographic region |
| `commodity` | string | No | `"crude_oil"` | Commodity type |
| `forecast_days` | integer | No | `30` | Number of days to forecast (1-365) |
| `start_date` | string | No | `tomorrow` | Start date in YYYY-MM-DD format |

#### Valid Regions
- `"global"`, `"north_america"`, `"europe"`, `"asia_pacific"`
- `"middle_east"`, `"africa"`, `"south_america"`

#### Valid Commodities
- `"crude_oil"`, `"natural_gas"`, `"refined_products"`
- `"gasoline"`, `"diesel"`

### Example Requests

#### Global Crude Oil Demand
```json
{
  "data": {
    "region": "global",
    "commodity": "crude_oil",
    "forecast_days": 30
  }
}
```

#### Regional Natural Gas Demand
```json
{
  "data": {
    "region": "north_america",
    "commodity": "natural_gas",
    "forecast_days": 14
  }
}
```

#### Asia Pacific Refined Products
```json
{
  "data": {
    "region": "asia_pacific",
    "commodity": "refined_products",
    "forecast_days": 60,
    "start_date": "2025-01-01"
  }
}
```

### Response Format
```json
{
  "model_info": {
    "name": "simple_demand_predictor",
    "version": "1.0",
    "type": "mock_demand_model"
  },
  "input_parameters": {
    "region": "global",
    "commodity": "crude_oil",
    "forecast_days": 30,
    "start_date": "2025-11-09"
  },
  "predictions": [
    {
      "date": "2025-11-09",
      "region": "global",
      "commodity": "crude_oil",
      "predicted_demand": 101075.0,
      "unit": "thousand_bpd",
      "confidence_lower": 90967.5,
      "confidence_upper": 111182.5
    }
  ],
  "summary": {
    "region": "global",
    "commodity": "crude_oil",
    "unit": "thousand_bpd",
    "forecast_period": "2025-11-09 to 2025-12-08",
    "starting_demand": 101075.0,
    "ending_demand": 102034.7,
    "demand_change_percent": 0.95,
    "average_daily_demand": 100259.0,
    "peak_demand": 107070.1,
    "minimum_demand": 95346.0,
    "total_forecast_demand": 3007770.4,
    "annual_growth_rate": 2.0
  }
}
```

---

## Domino Model API Deployment

### Step 1: Upload API File

Choose one of these files as your main script:
- `/mnt/code/src/api/simple_prediction_api.py`
- `/mnt/code/src/api/simple_price_api.py`
- `/mnt/code/src/api/simple_demand_api.py`

### Step 2: Set Function Name

Use the function name as your endpoint:
- `predict_production`
- `predict_prices`
- `predict_demand`

### Step 3: Test Request Format

```bash
curl -X POST https://your-domino-endpoint.com/predict_prices \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -d '{
       "data": {
         "commodity": "crude_oil",
         "forecast_days": 30
       }
     }'
```

### Step 4: Environment Requirements

Ensure your environment includes:
```
pandas>=1.3.0
numpy>=1.20.0
```

---

## Testing the APIs

### Local Testing

```bash
# Test production API
python /mnt/code/src/api/simple_prediction_api.py

# Test price API
python /mnt/code/src/api/simple_price_api.py

# Test demand API
python /mnt/code/src/api/simple_demand_api.py
```

### Example Test Calls

```python
from src.api.simple_prediction_api import predict_production
from src.api.simple_price_api import predict_prices
from src.api.simple_demand_api import predict_demand

# Production forecast
result = predict_production(facility_id="WELL_123", forecast_days=14)
print(result['summary'])

# Price forecast
result = predict_prices(commodity="natural_gas", forecast_days=21)
print(result['summary'])

# Demand forecast
result = predict_demand(region="europe", commodity="crude_oil", forecast_days=30)
print(result['summary'])
```

---

## Error Handling

All APIs return error information if something goes wrong:

```json
{
  "error": "forecast_days must be between 1 and 365",
  "input_parameters": {
    "facility_id": "all_facilities",
    "forecast_days": 400,
    "start_date": null
  },
  "timestamp": "2025-11-08T21:30:00Z"
}
```

### Common Validation Errors

1. **Invalid forecast_days**: Must be between 1 and 365
2. **Invalid commodity**: Must be one of the specified valid options
3. **Invalid region**: Must be one of the specified valid regions
4. **Invalid start_date**: Must be in YYYY-MM-DD format

---

## Support

- **API Files**: `/mnt/code/src/api/simple_*.py`
- **Documentation**: `/mnt/code/docs/`
- **Testing**: All APIs include built-in test examples

The simplified APIs are production-ready and designed specifically for Domino Model API deployment with minimal dependencies and reliable mock predictions.