# Oil & Gas Synthetic Datasets Documentation

## Overview

This project contains comprehensive synthetic datasets designed for oil and gas company dashboards and analytics applications. The datasets simulate realistic operational data from a global oil and gas company with facilities across six major regions.

## Project Information

- **Project Name**: oil_gas_dashboards
- **Generation Date**: 2024-11-07
- **Data Quality Score**: 0.892/1.0
- **Total Records**: 1,681,543
- **MLflow Experiment**: oil_gas_data_generation_oil_gas_dashboards

## Dataset Structure

### 1. Geospatial Facilities Dataset
**File**: `/mnt/data/Oil-and-Gas-Demo/geospatial_facilities.parquet`

Contains location and operational data for oil and gas facilities worldwide.

**Key Fields**:
- `facility_id`: Unique facility identifier
- `facility_type`: Type of facility (oil_well, refinery, storage_tank, processing_plant, terminal, compressor_station, pipeline_station)
- `latitude`, `longitude`: Geographic coordinates
- `region`: Global region (North America, Middle East, Europe, South America, Africa, Asia Pacific)
- `country`: Country location
- `status`: Operational status (active, maintenance, inactive)
- `oil_production_bpd`: Oil production in barrels per day
- `gas_production_mcfd`: Gas production in thousand cubic feet per day
- `equipment_health_score`: Equipment condition score (0-1)
- Environmental monitoring data (H2S, CO2, noise levels)
- Maintenance scheduling information

**Statistics**:
- Total facilities: 1,725
- Oil wells: 1,500 (87.0%)
- Refineries: 25 (1.4%)
- Other facilities: 200 (11.6%)
- Regions covered: 6
- Countries: 36

### 2. Production Time Series Dataset
**File**: `/mnt/data/Oil-and-Gas-Demo/production_timeseries.parquet`

Daily production data for all oil wells from 2022-2024.

**Key Fields**:
- `date`: Production date
- `facility_id`: Well identifier
- `oil_production_bpd`: Daily oil production
- `gas_production_mcfd`: Daily gas production
- `water_cut_percent`: Water content percentage
- `gor`: Gas-oil ratio
- `wellhead_pressure_psi`: Wellhead pressure
- `flowing_tubing_pressure_psi`: Tubing pressure

**Statistics**:
- Records: 1,644,000
- Date range: 2022-01-01 to 2024-12-31
- Wells tracked: 1,500
- Production variability includes seasonal factors, equipment degradation, and maintenance impacts

### 3. Price Time Series Dataset
**File**: `/mnt/data/Oil-and-Gas-Demo/prices_timeseries.parquet`

Daily commodity prices for crude oil, natural gas, and refined products.

**Key Fields**:
- `date`: Price date
- `crude_oil_price_usd_bbl`: Crude oil price (USD/barrel)
- `natural_gas_price_usd_mcf`: Natural gas price (USD/MCF)
- `gasoline_price_usd_gal`: Gasoline price (USD/gallon)
- `diesel_price_usd_gal`: Diesel price (USD/gallon)
- `jet_fuel_price_usd_gal`: Jet fuel price (USD/gallon)
- `brent_crude_usd_bbl`: Brent crude benchmark
- `wti_crude_usd_bbl`: WTI crude benchmark

**Statistics**:
- Records: 1,096 (daily prices for 3 years)
- Price volatility and trends included
- Correlated product pricing relationships

### 4. Demand Forecasting Dataset
**File**: `/mnt/data/Oil-and-Gas-Demo/demand_timeseries.parquet`

Regional energy demand patterns and forecasting data.

**Key Fields**:
- `date`: Demand date
- `region`: Geographic region
- `oil_demand_thousand_bpd`: Oil demand (thousand barrels/day)
- `gas_demand_thousand_mcfd`: Gas demand (thousand MCF/day)
- `gasoline_demand_thousand_bpd`: Gasoline demand
- `diesel_demand_thousand_bpd`: Diesel demand
- `jet_fuel_demand_thousand_bpd`: Jet fuel demand

**Statistics**:
- Records: 6,576
- Seasonal demand patterns included
- Economic growth trends incorporated
- Regional demand variations

### 5. Maintenance Schedule Dataset
**File**: `/mnt/data/Oil-and-Gas-Demo/maintenance_timeseries.parquet`

Equipment maintenance events and scheduling data.

**Key Fields**:
- `date`: Maintenance date
- `facility_id`: Facility identifier
- `maintenance_type`: Type (preventive, corrective, predictive, shutdown)
- `duration_hours`: Maintenance duration
- `cost_usd`: Maintenance cost
- `priority`: Priority level (low, medium, high, critical)
- `planned_start`, `planned_end`: Scheduled timeframe

**Statistics**:
- Records: 23,371 maintenance events
- Cost distribution by maintenance type
- Realistic scheduling patterns
- Priority-based planning

### 6. Weather Impact Dataset
**File**: `/mnt/data/Oil-and-Gas-Demo/weather_timeseries.parquet`

Regional weather conditions and operational impact assessment.

**Key Fields**:
- `date`: Weather date
- `region`: Geographic region
- `temperature_celsius`: Daily temperature
- `humidity_percent`: Humidity level
- `wind_speed_kmh`: Wind speed
- `precipitation_mm`: Precipitation amount
- `weather_condition`: Condition type (normal, hurricane, heat_wave, cold_snap, heavy_rain)
- `operation_impact_score`: Impact on operations (0-1)

**Statistics**:
- Records: 6,576
- Extreme weather events included (2% frequency)
- Regional climate variations
- Operational impact correlation

## Data Quality Assessment

### Quality Scores by Dataset
- **Geospatial**: 0.750 - Good coverage with realistic facility distribution
- **Production**: 1.000 - Excellent temporal consistency and data integrity
- **Prices**: 1.000 - Complete price series with realistic volatility
- **Demand**: 1.000 - Comprehensive regional demand patterns
- **Maintenance**: 0.600 - Good scheduling patterns, some data sparsity expected
- **Weather**: 1.000 - Complete weather coverage with impact modeling

### Data Validation Results
- **Missing Values**: <0.1% across all datasets
- **Temporal Consistency**: 100% for time series data
- **Geographic Coverage**: Global representation across 6 regions
- **Realistic Distributions**: Log-normal production, seasonal patterns, equipment degradation

## Technical Implementation

### File Storage
- **Data Location**: `/mnt/data/Oil-and-Gas-Demo/`
- **Format**: Apache Parquet for efficient columnar storage
- **Size**: Optimized for dashboard query performance
- **Compression**: Snappy compression enabled

### MLflow Integration
- **Experiment**: oil_gas_data_generation_oil_gas_dashboards
- **Tracking URI**: http://localhost:8768
- **Artifacts Logged**: Sample data, generation summary, quality metrics
- **Reproducibility**: Seeded random generation for consistent results

### Data Configuration
- **Project Type**: Git-based (DOMINO_WORKING_DIR=/mnt/code)
- **Data Path Resolution**: Automatic based on project type
- **Path Utility**: `/mnt/code/scripts/data_config.py`

## Dashboard Recommendations

### 1. Geospatial Operations Dashboard
- Interactive world map with facility locations
- Production heat maps by region
- Equipment health monitoring
- Environmental compliance tracking

### 2. Production Analytics Dashboard
- Time series production trends
- Well performance comparisons
- Regional production analysis
- Decline curve analysis

### 3. Financial Performance Dashboard
- Price trend analysis
- Revenue optimization
- Cost per barrel analysis
- Market volatility assessment

### 4. Demand Forecasting Dashboard
- Regional demand patterns
- Seasonal adjustment models
- Supply-demand balancing
- Market opportunity identification

### 5. Maintenance Operations Dashboard
- Maintenance scheduling calendar
- Cost optimization analysis
- Equipment reliability tracking
- Preventive maintenance planning

### 6. Risk Management Dashboard
- Weather impact assessment
- Operational risk scoring
- Extreme weather preparedness
- Business continuity planning

### 7. Executive Summary Dashboard
- Key performance indicators
- Regional performance comparison
- Financial summary metrics
- Operational efficiency trends

## Usage Instructions

### Loading Data
```python
import pandas as pd
import sys
sys.path.insert(0, '/mnt/code')
from scripts.data_config import get_data_paths

# Get correct data paths
paths = get_data_paths('oil_gas_dashboards')
data_dir = paths['base_data_path']

# Load datasets
geospatial_df = pd.read_parquet(data_dir / 'geospatial_facilities.parquet')
production_df = pd.read_parquet(data_dir / 'production_timeseries.parquet')
prices_df = pd.read_parquet(data_dir / 'prices_timeseries.parquet')
```

### Data Exploration
- **Jupyter Notebook**: `/mnt/code/notebooks/oil_gas_data_exploration.ipynb`
- **Interactive Analysis**: Comprehensive exploration with visualizations
- **Quality Assessment**: Built-in data quality checks and validation

### Dashboard Development
1. Use Plotly/Dash for interactive Python dashboards
2. Deploy as Domino Apps for enterprise access
3. Implement real-time refresh using scheduled jobs
4. Create API endpoints for data serving
5. Use MLflow for model deployment and monitoring

## Support and Maintenance

### Data Updates
- **Regeneration**: Run `python scripts/oil_gas_data_generator.py`
- **Customization**: Modify parameters in generator for different scenarios
- **Extension**: Add new facility types or regions as needed

### Quality Monitoring
- **Automated Checks**: Built-in validation during generation
- **MLflow Tracking**: All quality metrics logged automatically
- **Alert Thresholds**: Quality scores below 0.7 require review

### Documentation
- **Code Documentation**: Comprehensive inline documentation
- **API Reference**: Generated from docstrings
- **Usage Examples**: Included in exploration notebook

## Contact and Support
For questions about the synthetic data generation or dashboard development recommendations, please refer to the project documentation or contact the data engineering team.

---

*Generated by Oil & Gas Synthetic Data Generator*  
*Project: oil_gas_dashboards*  
*Quality Score: 0.892*  
*MLflow Experiment: http://localhost:8768/#/experiments/1902*