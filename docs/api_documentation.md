# Oil & Gas Forecasting API Documentation

This document provides comprehensive documentation for the simplified Oil & Gas Forecasting API endpoints deployed on Domino Data Lab. All endpoints accept individual parameters directly as specified in the Domino Model API documentation.

## API Endpoints Overview

| Endpoint | Purpose | Primary Use Case | API File |
|----------|---------|------------------|----------|
| `/predict_production` | Production Forecasting | Forecast oil/gas production volumes | `simple_prediction_api.py` |
| `/predict_prices` | Price Forecasting | Forecast crude oil and natural gas prices | `simple_price_api.py` |
| `/predict_demand` | Demand Forecasting | Forecast regional energy demand | `simple_demand_api.py` |

---

## 1. Production Forecasting API

**Endpoint**: `/predict_production`  
**File**: `/mnt/code/src/api/prediction_api.py`  
**Function**: `predict_production(input_data)`

### Input Format

#### Required Parameters
```json
{
  "data": {
    "start": 1,      // Starting day index (1-based)
    "stop": 30       // Ending day index (1-based, inclusive)
  }
}
```

#### Optional Parameters
```json
{
  "data": {
    "start": 1,
    "stop": 30
  },
  "facility_id": "WELL_000001",     // Specific facility or "all_facilities"
  "aggregation": "total"            // "total", "average", or "facility"
}
```

### Example Requests

#### Basic 7-day Production Forecast
```json
{
  "data": {
    "start": 1,
    "stop": 7
  }
}
```

#### Specific Facility 30-day Forecast
```json
{
  "data": {
    "start": 5,
    "stop": 35
  },
  "facility_id": "WELL_000001",
  "aggregation": "facility"
}
```

#### Long-term Total Production Forecast
```json
{
  "data": {
    "start": 1,
    "stop": 180
  },
  "facility_id": "all_facilities",
  "aggregation": "total"
}
```

### Parameter Details
- **`start/stop`**: 1-based day indices for forecast range (inclusive)
- **`facility_id`**: 
  - `"WELL_000001"` (specific facility ID)
  - `"all_facilities"` (aggregate all facilities)
- **`aggregation`**:
  - `"total"`: Sum all production values
  - `"average"`: Average production per facility
  - `"facility"`: Individual facility breakdowns

### Response Format
```json
{
  "model_info": {
    "name": "oil_gas_production_champion",
    "version": "1.0",
    "framework": "ensemble_production_model"
  },
  "input_parameters": {
    "data": {"start": 1, "stop": 30},
    "facility_id": "all_facilities",
    "aggregation": "total"
  },
  "predictions": [
    {
      "date": "2025-11-09",
      "predicted_production_bpd": 52341.7,
      "facility_id": "all_facilities",
      "confidence_interval": [48200, 56500]
    }
  ],
  "summary": {
    "total_forecast_days": 30,
    "average_daily_production": 52000.5,
    "total_forecasted_production": 1560015.0,
    "forecast_period": "2025-11-09 to 2025-12-08"
  }
}
```

---

## 2. Price Forecasting API

**Endpoint**: `/predict_prices`  
**File**: `/mnt/code/src/api/price_prediction_api.py`  
**Function**: `predict_prices(input_data)`

### Input Format

#### Required Parameters
```json
{
  "data": {
    "start": 1,      // Starting day index (1-based)
    "stop": 100      // Ending day index (1-based, inclusive)
  }
}
```

#### Optional Parameters
```json
{
  "data": {
    "start": 1,
    "stop": 100
  },
  "commodity": "crude_oil",         // "crude_oil" or "natural_gas"
  "market_indicators": {            // Market adjustment factors
    "supply_disruption": false,
    "demand_surge": false,
    "economic_downturn": false
  }
}
```

### Example Requests

#### Basic 30-day Crude Oil Forecast
```json
{
  "data": {
    "start": 1,
    "stop": 30
  }
}
```

#### Natural Gas Winter Forecast with Market Factors
```json
{
  "data": {
    "start": 10,
    "stop": 40
  },
  "commodity": "natural_gas",
  "market_indicators": {
    "supply_disruption": false,
    "demand_surge": true,
    "economic_downturn": false
  }
}
```

#### Long-term Oil Price Forecast
```json
{
  "data": {
    "start": 1,
    "stop": 365
  },
  "commodity": "crude_oil",
  "market_indicators": {
    "supply_disruption": true,
    "demand_surge": false,
    "economic_downturn": false
  }
}
```

### Parameter Details
- **`start/stop`**: 1-based day indices for forecast range (inclusive)
- **`commodity`**: 
  - `"crude_oil"` (WTI crude oil in USD/barrel)
  - `"natural_gas"` (natural gas in USD/MMBtu)
- **`market_indicators`**:
  - `supply_disruption`: +10% price adjustment if true
  - `demand_surge`: +5% price adjustment if true
  - `economic_downturn`: -8% price adjustment if true

### Response Format
```json
{
  "model_info": {
    "name": "oil_gas_price_prediction_champion",
    "version": "1.0",
    "framework": "ensemble_price_model"
  },
  "input_parameters": {
    "data": {"start": 1, "stop": 30},
    "commodity": "crude_oil",
    "market_indicators": {}
  },
  "predictions": [
    {
      "date": "2025-11-09",
      "predicted_price": 75.23,
      "commodity": "crude_oil",
      "unit": "USD/barrel",
      "confidence_interval_lower": 69.85,
      "confidence_interval_upper": 80.61,
      "volatility": 0.025
    }
  ],
  "summary": {
    "commodity": "crude_oil",
    "forecast_period": "2025-11-09 to 2025-12-08",
    "starting_price": 75.23,
    "ending_price": 77.15,
    "price_change_percent": 2.55,
    "average_price": 76.12,
    "max_price": 82.45,
    "min_price": 71.89,
    "price_volatility": 2.84
  }
}
```

---

## 3. Demand Forecasting API

**Endpoint**: `/predict_demand`  
**File**: `/mnt/code/src/api/demand_forecast_api.py`  
**Function**: `predict_demand(input_data)`

### Input Format

#### Required Parameters
```json
{
  "data": {
    "start": 1,      // Starting day index (1-based)
    "stop": 100      // Ending day index (1-based, inclusive)
  }
}
```

#### Optional Parameters
```json
{
  "data": {
    "start": 1,
    "stop": 100
  },
  "region": "north_america",        // Geographic region
  "commodity": "crude_oil",         // Commodity type
  "economic_factors": {             // Economic context
    "gdp_growth": 2.5,
    "industrial_activity": "high",
    "weather_impact": "normal"
  }
}
```

### Example Requests

#### Basic Global Demand Forecast
```json
{
  "data": {
    "start": 1,
    "stop": 90
  }
}
```

#### Regional Natural Gas Demand (Winter)
```json
{
  "data": {
    "start": 15,
    "stop": 75
  },
  "region": "europe",
  "commodity": "natural_gas",
  "economic_factors": {
    "gdp_growth": 1.8,
    "industrial_activity": "medium",
    "weather_impact": "cold_winter"
  }
}
```

#### Asia Pacific Refined Products Forecast
```json
{
  "data": {
    "start": 1,
    "stop": 180
  },
  "region": "asia_pacific",
  "commodity": "refined_products",
  "economic_factors": {
    "gdp_growth": 4.2,
    "industrial_activity": "high",
    "weather_impact": "normal"
  }
}
```

### Parameter Details
- **`start/stop`**: 1-based day indices for forecast range (inclusive)
- **`region`**: 
  - `"north_america"`, `"europe"`, `"asia_pacific"`
  - `"middle_east"`, `"south_america"`, `"africa"`
  - `"global"` (all regions combined)
- **`commodity`**: 
  - `"crude_oil"`, `"natural_gas"`, `"refined_products"`
  - `"gasoline"`, `"diesel"`, `"jet_fuel"`
- **`economic_factors`**: Economic indicators affecting demand

### Response Format
```json
{
  "model_info": {
    "name": "oil_gas_demand_forecast_champion",
    "version": "1.0",
    "framework": "ensemble_demand_model"
  },
  "input_parameters": {
    "data": {"start": 1, "stop": 90},
    "region": "north_america",
    "commodity": "crude_oil",
    "economic_factors": {}
  },
  "predictions": [
    {
      "date": "2025-11-09",
      "predicted_demand": 18500000,
      "region": "north_america",
      "commodity": "crude_oil",
      "unit": "barrels_per_day",
      "confidence_interval": [17800000, 19200000]
    }
  ],
  "summary": {
    "commodity": "crude_oil",
    "region": "north_america",
    "forecast_period": "2025-11-09 to 2026-02-06",
    "average_daily_demand": 18650000,
    "total_demand_period": 1678500000,
    "demand_growth_rate": 2.3,
    "seasonal_peak": "2025-12-15"
  }
}
```

---

## Common Validation Rules

All endpoints share these validation rules:

1. **Range Validation**:
   - `start` and `stop` must be >= 1
   - `start` must be <= `stop`
   - Maximum forecast range: 365 days

2. **Date Calculation**:
   - `start=1` means "starting tomorrow"
   - `stop=30` with `start=1` means "30-day forecast"
   - Dates are calculated from current timestamp + (start_day - 1)

3. **Error Handling**:
   - Invalid parameters return descriptive error messages
   - All responses include input parameter echoing
   - Timestamps and metadata for debugging

---

## Deployment Instructions

### For Domino Model API Deployment

1. **Create Model API Endpoint**:
   ```bash
   # Upload one of these files as the main script:
   # - src/api/prediction_api.py
   # - src/api/price_prediction_api.py  
   # - src/api/demand_forecast_api.py
   ```

2. **Function Name**: Use the function name for the endpoint:
   - `predict_production`
   - `predict_prices` 
   - `predict_demand`

3. **Request Format**: All requests use JSON with the Domino format:
   ```bash
   curl -X POST https://your-domino-endpoint.com/predict_prices \
        -H "Content-Type: application/json" \
        -d '{"data": {"start": 1, "stop": 30}}'
   ```

4. **Environment**: Ensure Python environment includes:
   - pandas, numpy, scikit-learn
   - mlflow, datetime, logging
   - All dependencies in requirements.txt

### Testing Commands

```bash
# Test production API
python src/api/prediction_api.py

# Test price API  
python src/api/price_prediction_api.py

# Test demand API
python src/api/demand_forecast_api.py
```

---

## Support and Troubleshooting

### Common Issues

1. **"start index must be <= stop index"**: Ensure start <= stop
2. **"forecast range cannot exceed 365 days"**: Reduce the date range
3. **"commodity must be ..."**: Use exact string values from documentation
4. **"region must be ..."**: Use exact region names (lowercase with underscores)

### Contact

- **Documentation**: `/mnt/code/docs/`
- **Source Code**: `/mnt/code/src/api/`
- **Model Artifacts**: `/mnt/artifacts/models/`
- **MLflow Tracking**: http://localhost:8768

All APIs are production-ready and follow Domino Data Lab best practices for model deployment and monitoring.