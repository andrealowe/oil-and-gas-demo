# Oil & Gas Dashboard Suite Documentation

## Overview

This documentation covers two comprehensive Streamlit dashboard applications designed for oil & gas operations monitoring and forecasting. Both dashboards feature professional Aramco-inspired design with enterprise-grade functionality.

## Dashboard Applications

### 1. Geospatial Operations Dashboard (`/mnt/code/scripts/geospatial_dashboard.py`)

**Purpose:** Real-time facility monitoring, equipment health assessment, and production optimization.

**Key Features:**
- Interactive global facility map with health status visualization
- Equipment health monitoring and predictive maintenance alerts
- Production efficiency analysis and regional performance comparisons
- Environmental risk assessment and compliance monitoring
- AI-powered facility predictions through integrated API endpoints

**Tabs & Functionality:**
1. **Facility Map:** Interactive world map showing all facilities with color-coded health status
2. **Production Analysis:** Multi-dimensional production efficiency charts and regional comparisons
3. **Equipment Health:** Real-time equipment monitoring dashboard with maintenance scheduling
4. **Facility Details:** Detailed individual facility information and metrics
5. **AI Predictions:** Real-time AI predictions for equipment health, production efficiency, and environmental risk

### 2. Forecasting Dashboard (`/mnt/code/scripts/forecasting_dashboard.py`)

**Purpose:** Advanced predictive analytics for production, pricing, and demand optimization.

**Key Features:**
- Time series forecasting for oil/gas production, prices, and demand
- Interactive forecast parameter controls and model customization
- Maintenance scheduling optimization with cost analysis
- Regional demand forecasting and market analysis
- Comprehensive forecast export and reporting capabilities

**Tabs & Functionality:**
1. **Production Forecasts:** Historical trends and future production predictions
2. **Price Forecasts:** Oil & gas price predictions with market factor analysis
3. **Demand Forecasts:** Regional demand analysis and product-specific forecasting
4. **Maintenance Planning:** Predictive maintenance scheduling and optimization
5. **Forecast Controls:** Interactive model parameters and performance metrics
6. **Forecast Summary:** Comprehensive results export and analysis

## Technical Architecture

### Design System
- **Color Palette:** Aramco-inspired professional colors
  - Primary Blue: #00A3E0
  - Secondary Green: #84BD00
  - Dark Blue: #0033A0
  - Light Gray: #F5F5F5
  - Orange Accent: #FF8C00

- **Typography:** Clean, professional fonts (Arial, Helvetica)
- **Layout:** Card-based design with responsive grid system
- **Charts:** Plotly visualizations with consistent styling

### Data Sources
- **Facilities Data:** `/mnt/data/Oil-and-Gas-Demo/geospatial_facilities.parquet`
- **Production Data:** `/mnt/data/Oil-and-Gas-Demo/production_timeseries.parquet`
- **Price Data:** `/mnt/data/Oil-and-Gas-Demo/prices_timeseries.parquet`
- **Demand Data:** `/mnt/data/Oil-and-Gas-Demo/demand_timeseries.parquet`
- **Maintenance Data:** `/mnt/data/Oil-and-Gas-Demo/maintenance_timeseries.parquet`

### API Integration
- **Geospatial API:** `/mnt/code/src/api/domino_geospatial_endpoint.py`
- **Forecasting API:** `/mnt/code/src/api/domino_forecasting_endpoint.py`

## Deployment Instructions

### Domino App Deployment

#### Geospatial Dashboard
```bash
# Use the provided launcher script
./app_geospatial.sh

# Or manually deploy with:
streamlit run scripts/geospatial_dashboard.py --server.port=8501
```

#### Forecasting Dashboard
```bash
# Use the provided launcher script
./app_forecasting.sh

# Or manually deploy with:
streamlit run scripts/forecasting_dashboard.py --server.port=8502
```

### Hardware Requirements
- **CPU:** Minimum 2 cores, recommended 4+ cores
- **Memory:** Minimum 8GB RAM, recommended 16GB+ for large datasets
- **Storage:** 5GB minimum for data caching
- **Network:** Stable internet connection for API calls

### Environment Setup
1. Ensure all requirements are installed: `pip install -r requirements.txt`
2. Verify data access to `/mnt/data/Oil-and-Gas-Demo/`
3. Check API endpoints are available (optional, fallback data provided)
4. Run health checks using the launcher scripts

## User Guide

### Getting Started

#### Geospatial Dashboard
1. **Access:** Navigate to the Domino App URL (typically port 8501)
2. **Filters:** Use sidebar to filter by region, facility type, and health threshold
3. **Navigation:** Switch between tabs for different analyses
4. **Interaction:** Click map points for facility details, use dropdowns for specific analysis

#### Forecasting Dashboard
1. **Access:** Navigate to the Domino App URL (typically port 8502)
2. **Parameters:** Configure forecast horizon, types, and confidence levels in sidebar
3. **Generation:** Click "Generate New Forecasts" to create predictions
4. **Analysis:** Review results across different tabs
5. **Export:** Use the summary tab to export forecast results

### Key Features Guide

#### Interactive Maps (Geospatial)
- **Color Coding:** Green (Excellent), Orange (Good), Red (Fair/Poor)
- **Size Mapping:** Bubble size represents production volume
- **Hover Information:** Detailed facility metrics on hover
- **Filtering:** Real-time filtering by multiple criteria

#### Forecast Controls (Forecasting)
- **Horizon Selection:** 7 days to 1 year forecasting periods
- **Model Parameters:** Sensitivity, trend weighting, noise reduction
- **Confidence Levels:** 80%, 85%, 90%, 95% statistical confidence
- **External Factors:** Include geopolitical, weather, economic factors

#### Performance Monitoring
- **Real-time Updates:** Data refreshes every 5 minutes
- **Health Checks:** Automated system health monitoring
- **Error Handling:** Graceful degradation with fallback data
- **Performance Metrics:** Response times and accuracy measures

### Advanced Features

#### AI Predictions (Geospatial)
- Select any facility for AI-powered predictions
- Equipment health scoring with confidence intervals
- Production efficiency classification (High/Medium/Low)
- Environmental risk assessment (Low/Medium/High)

#### Forecast Export (Forecasting)
- JSON format with comprehensive metadata
- Timestamped results for audit trails
- Parameter documentation for reproducibility
- Confidence intervals and model performance metrics

## Troubleshooting

### Common Issues

#### Data Loading Problems
**Issue:** "Unable to load facility data"
**Solution:**
1. Check data directory permissions: `/mnt/data/Oil-and-Gas-Demo/`
2. Verify parquet files are accessible
3. Clear cache and refresh: Sidebar > "Refresh Data"

#### API Connection Issues
**Issue:** "API call failed, using fallback prediction"
**Solution:**
1. Check network connectivity
2. Verify API endpoints are running
3. Dashboard will automatically use synthetic data as fallback

#### Performance Issues
**Issue:** Slow loading or timeouts
**Solution:**
1. Reduce data filters to smaller subsets
2. Check available memory and CPU resources
3. Clear browser cache and refresh
4. Use data sampling in sidebar controls

#### Visualization Problems
**Issue:** Charts not displaying correctly
**Solution:**
1. Refresh the browser page
2. Check browser JavaScript console for errors
3. Try different browser (Chrome/Firefox recommended)
4. Clear Streamlit cache: Sidebar > "Refresh Data"

### Performance Optimization

#### Data Caching
- Implement 5-minute TTL on data loading functions
- Use Streamlit's `@st.cache_data` decorator
- Clear cache periodically to refresh data

#### Memory Management
- Filter large datasets before processing
- Use data sampling for exploratory analysis
- Monitor memory usage in Domino workspace

#### Network Optimization
- Minimize API calls through intelligent caching
- Use fallback data when API is unavailable
- Implement request timeouts for reliability

## Maintenance and Updates

### Regular Maintenance
- **Data Refresh:** Automated daily data updates
- **Model Retraining:** Weekly model performance evaluation
- **Security Updates:** Monthly dependency updates
- **Performance Monitoring:** Continuous resource monitoring

### Version Control
- All dashboards are version controlled in Git
- Feature branches for new functionality
- Tagged releases for production deployment

### Support
- **Technical Issues:** Contact Domino Data Lab support
- **Feature Requests:** Submit through internal ticketing system
- **Bug Reports:** Use GitHub issues or internal tracking

## Security and Compliance

### Data Security
- All data access through secure Domino File System
- No external data transmission without encryption
- User access controlled by Domino RBAC

### Compliance Features
- Audit trail for all user interactions
- Data lineage tracking for regulatory requirements
- Secure API endpoints with authentication

### Privacy
- No personal data collection or storage
- Session-based state management only
- Compliance with corporate data governance policies

## Future Enhancements

### Planned Features
- Real-time data streaming integration
- Advanced ML model ensemble forecasting
- Mobile-responsive design improvements
- Additional visualization types and customization

### Integration Roadmap
- SAP integration for maintenance scheduling
- GIS system integration for enhanced mapping
- Corporate dashboard embedding capabilities
- Advanced alerting and notification systems

---

**Last Updated:** November 2024  
**Version:** 1.0  
**Maintainer:** Oil & Gas Analytics Team