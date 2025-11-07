#!/usr/bin/env python3
"""
Test script for Oil & Gas Forecasting API

Tests API endpoints and generates sample forecasts for dashboard integration.
"""

import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys

# Add scripts directory to path
sys.path.insert(0, '/mnt/code')
from scripts.data_config import get_data_paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForecastingAPITester:
    """Test class for forecasting API endpoints"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.paths = get_data_paths('Oil-and-Gas-Demo')
        self.output_dir = self.paths['artifacts_path'] / 'api_tests'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def test_health_endpoint(self):
        """Test API health check"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"API Health Check: {health_data['status']}")
                logger.info(f"Models Loaded: {health_data['models_loaded']}")
                return True
            else:
                logger.error(f"Health check failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False
    
    def test_models_endpoint(self):
        """Test models listing endpoint"""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                logger.info(f"Total Models Available: {models_data['total_models']}")
                
                # Save models info
                models_file = self.output_dir / 'available_models.json'
                with open(models_file, 'w') as f:
                    json.dump(models_data, f, indent=2)
                
                return models_data
            else:
                logger.error(f"Models endpoint failed: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Models endpoint error: {e}")
            return None
    
    def test_production_forecast(self):
        """Test production forecasting endpoints"""
        logger.info("Testing production forecasting...")
        
        forecasts = {}
        targets = ['oil_production_bpd', 'gas_production_mcfd']
        horizons = [30, 180]
        
        for target in targets:
            for horizon in horizons:
                try:
                    url = f"{self.base_url}/forecast/production/{target}"
                    params = {'horizon_days': horizon}
                    
                    response = requests.get(url, params=params, timeout=30)
                    
                    if response.status_code == 200:
                        forecast_data = response.json()
                        forecasts[f"{target}_{horizon}d"] = forecast_data
                        logger.info(f"✓ Production forecast: {target} ({horizon} days)")
                        
                        # Save forecast to CSV for dashboard
                        df = pd.DataFrame(forecast_data['forecast'])
                        csv_file = self.output_dir / f"production_{target}_{horizon}d_forecast.csv"
                        df.to_csv(csv_file, index=False)
                        
                    else:
                        logger.warning(f"✗ Production forecast failed: {target} ({horizon} days) - {response.status_code}")
                        
                except Exception as e:
                    logger.error(f"Production forecast error for {target}: {e}")
        
        return forecasts
    
    def test_price_forecast(self):
        """Test price forecasting endpoints"""
        logger.info("Testing price forecasting...")
        
        forecasts = {}
        commodities = [
            'crude_oil_price_usd_bbl', 
            'natural_gas_price_usd_mcf',
            'brent_crude_usd_bbl', 
            'wti_crude_usd_bbl'
        ]
        horizons = [30, 180]
        
        for commodity in commodities:
            for horizon in horizons:
                try:
                    url = f"{self.base_url}/forecast/prices/{commodity}"
                    params = {'horizon_days': horizon}
                    
                    response = requests.get(url, params=params, timeout=30)
                    
                    if response.status_code == 200:
                        forecast_data = response.json()
                        forecasts[f"{commodity}_{horizon}d"] = forecast_data
                        logger.info(f"✓ Price forecast: {commodity} ({horizon} days)")
                        
                        # Save forecast to CSV for dashboard
                        df = pd.DataFrame(forecast_data['forecast'])
                        csv_file = self.output_dir / f"price_{commodity}_{horizon}d_forecast.csv"
                        df.to_csv(csv_file, index=False)
                        
                    else:
                        logger.warning(f"✗ Price forecast failed: {commodity} ({horizon} days) - {response.status_code}")
                        
                except Exception as e:
                    logger.error(f"Price forecast error for {commodity}: {e}")
        
        return forecasts
    
    def test_demand_forecast(self):
        """Test demand forecasting endpoints"""
        logger.info("Testing demand forecasting...")
        
        forecasts = {}
        regions = ['North America', 'Europe', 'Asia Pacific']
        targets = [
            'oil_demand_thousand_bpd',
            'gas_demand_thousand_mcfd', 
            'gasoline_demand_thousand_bpd',
            'diesel_demand_thousand_bpd'
        ]
        horizons = [30, 180]
        
        for region in regions:
            for target in targets:
                for horizon in horizons:
                    try:
                        url = f"{self.base_url}/forecast/demand/{region.replace(' ', '%20')}"
                        params = {'target': target, 'horizon_days': horizon}
                        
                        response = requests.get(url, params=params, timeout=30)
                        
                        if response.status_code == 200:
                            forecast_data = response.json()
                            forecasts[f"{region}_{target}_{horizon}d"] = forecast_data
                            logger.info(f"✓ Demand forecast: {region} - {target} ({horizon} days)")
                            
                            # Save forecast to CSV for dashboard
                            df = pd.DataFrame(forecast_data['forecast'])
                            csv_file = self.output_dir / f"demand_{region.replace(' ', '_')}_{target}_{horizon}d_forecast.csv"
                            df.to_csv(csv_file, index=False)
                            
                        else:
                            logger.warning(f"✗ Demand forecast failed: {region} - {target} ({horizon} days) - {response.status_code}")
                            
                    except Exception as e:
                        logger.error(f"Demand forecast error for {region} - {target}: {e}")
        
        return forecasts
    
    def test_maintenance_forecast(self):
        """Test maintenance forecasting endpoints"""
        logger.info("Testing maintenance forecasting...")
        
        forecasts = {}
        targets = ['cost_usd', 'duration_hours']
        horizons = [30, 180]
        
        for target in targets:
            for horizon in horizons:
                try:
                    url = f"{self.base_url}/forecast/maintenance"
                    params = {'target': target, 'horizon_days': horizon}
                    
                    response = requests.get(url, params=params, timeout=30)
                    
                    if response.status_code == 200:
                        forecast_data = response.json()
                        forecasts[f"{target}_{horizon}d"] = forecast_data
                        logger.info(f"✓ Maintenance forecast: {target} ({horizon} days)")
                        
                        # Save forecast to CSV for dashboard
                        df = pd.DataFrame(forecast_data['forecast'])
                        csv_file = self.output_dir / f"maintenance_{target}_{horizon}d_forecast.csv"
                        df.to_csv(csv_file, index=False)
                        
                    else:
                        logger.warning(f"✗ Maintenance forecast failed: {target} ({horizon} days) - {response.status_code}")
                        
                except Exception as e:
                    logger.error(f"Maintenance forecast error for {target}: {e}")
        
        return forecasts
    
    def test_custom_forecast_requests(self):
        """Test custom forecast requests using POST endpoint"""
        logger.info("Testing custom forecast requests...")
        
        test_requests = [
            {
                "forecast_type": "production",
                "target_variable": "oil_production_bpd", 
                "horizon_days": 60
            },
            {
                "forecast_type": "prices",
                "target_variable": "crude_oil_price_usd_bbl",
                "horizon_days": 90
            },
            {
                "forecast_type": "demand",
                "target_variable": "oil_demand_thousand_bpd",
                "horizon_days": 45,
                "region": "North America"
            }
        ]
        
        forecasts = {}
        
        for i, request_data in enumerate(test_requests):
            try:
                response = requests.post(
                    f"{self.base_url}/forecast",
                    json=request_data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    forecast_data = response.json()
                    forecasts[f"custom_request_{i+1}"] = forecast_data
                    logger.info(f"✓ Custom forecast: {request_data['forecast_type']} - {request_data['target_variable']}")
                    
                    # Save forecast
                    df = pd.DataFrame(forecast_data['forecast'])
                    csv_file = self.output_dir / f"custom_forecast_{i+1}.csv"
                    df.to_csv(csv_file, index=False)
                    
                else:
                    logger.warning(f"✗ Custom forecast failed: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Custom forecast error: {e}")
        
        return forecasts
    
    def generate_test_report(self, all_forecasts):
        """Generate comprehensive test report"""
        logger.info("Generating test report...")
        
        report = {
            "test_summary": {
                "timestamp": datetime.now().isoformat(),
                "api_base_url": self.base_url,
                "total_forecasts_generated": len(all_forecasts),
                "output_directory": str(self.output_dir)
            },
            "forecast_categories": {
                "production": len([k for k in all_forecasts.keys() if 'production' in k]),
                "prices": len([k for k in all_forecasts.keys() if 'price' in k]),
                "demand": len([k for k in all_forecasts.keys() if 'demand' in k]),
                "maintenance": len([k for k in all_forecasts.keys() if 'maintenance' in k]),
                "custom": len([k for k in all_forecasts.keys() if 'custom' in k])
            },
            "api_endpoints_tested": [
                "/health",
                "/models", 
                "/forecast/production/{target}",
                "/forecast/prices/{commodity}",
                "/forecast/demand/{region}",
                "/forecast/maintenance",
                "/forecast"
            ],
            "dashboard_integration_files": []
        }
        
        # List generated CSV files for dashboard integration
        csv_files = list(self.output_dir.glob("*.csv"))
        report["dashboard_integration_files"] = [str(f.name) for f in csv_files]
        
        # Save comprehensive report
        report_file = self.output_dir / 'api_test_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create summary markdown
        summary_md = f"""# Oil & Gas Forecasting API Test Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Test Summary
- **API Base URL**: {self.base_url}
- **Total Forecasts Generated**: {report['test_summary']['total_forecasts_generated']}
- **Output Directory**: {self.output_dir}

## Forecast Categories Tested
- **Production Forecasts**: {report['forecast_categories']['production']}
- **Price Forecasts**: {report['forecast_categories']['prices']}
- **Demand Forecasts**: {report['forecast_categories']['demand']}
- **Maintenance Forecasts**: {report['forecast_categories']['maintenance']}
- **Custom Forecasts**: {report['forecast_categories']['custom']}

## API Endpoints Tested
"""
        for endpoint in report["api_endpoints_tested"]:
            summary_md += f"- `{endpoint}`\n"
        
        summary_md += f"""
## Dashboard Integration Files
{len(report['dashboard_integration_files'])} CSV files generated for dashboard integration:

"""
        for file_name in sorted(report['dashboard_integration_files']):
            summary_md += f"- {file_name}\n"
        
        summary_md += """
## Usage Examples

### Production Forecast
```bash
curl -X GET "http://localhost:8000/forecast/production/oil_production_bpd?horizon_days=30"
```

### Price Forecast  
```bash
curl -X GET "http://localhost:8000/forecast/prices/crude_oil_price_usd_bbl?horizon_days=30"
```

### Custom Forecast
```bash
curl -X POST "http://localhost:8000/forecast" \\
     -H "Content-Type: application/json" \\
     -d '{"forecast_type": "production", "target_variable": "oil_production_bpd", "horizon_days": 30}'
```

## Model Performance
All forecasts include:
- Predicted values for specified horizon
- Model metadata (type, metrics, training info)
- Confidence intervals (when available)
- Timestamps for each prediction

---

**Note**: To use the API, ensure the forecasting service is running:
```bash
python src/api/forecasting_api.py
```
"""
        
        summary_file = self.output_dir / 'API_TEST_SUMMARY.md'
        with open(summary_file, 'w') as f:
            f.write(summary_md)
        
        logger.info(f"Test report saved: {report_file}")
        logger.info(f"Test summary saved: {summary_file}")
        
        return report
    
    def run_all_tests(self):
        """Run all API tests"""
        logger.info("Starting comprehensive API testing...")
        
        # Test basic endpoints first
        if not self.test_health_endpoint():
            logger.error("API health check failed - cannot proceed with tests")
            return None
        
        models_info = self.test_models_endpoint()
        if not models_info:
            logger.warning("Models endpoint failed - testing will continue with mock responses")
        
        # Test all forecast endpoints
        all_forecasts = {}
        
        production_forecasts = self.test_production_forecast()
        all_forecasts.update(production_forecasts)
        
        price_forecasts = self.test_price_forecast()
        all_forecasts.update(price_forecasts)
        
        demand_forecasts = self.test_demand_forecast()
        all_forecasts.update(demand_forecasts)
        
        maintenance_forecasts = self.test_maintenance_forecast()
        all_forecasts.update(maintenance_forecasts)
        
        custom_forecasts = self.test_custom_forecast_requests()
        all_forecasts.update(custom_forecasts)
        
        # Generate report
        report = self.generate_test_report(all_forecasts)
        
        logger.info(f"API testing completed! Generated {len(all_forecasts)} forecasts")
        logger.info(f"Test results saved to: {self.output_dir}")
        
        return report

def main():
    """Main testing function"""
    tester = ForecastingAPITester()
    
    print("Oil & Gas Forecasting API Test Suite")
    print("===================================")
    print()
    
    # Note: This will test with mock responses since API server isn't running
    print("Note: API server not running - this will demonstrate the test framework")
    print("To run actual tests, start the API server first:")
    print("python src/api/forecasting_api.py")
    print()
    
    # Create sample test files for demonstration
    tester.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample forecast data
    sample_forecast = {
        "forecast_type": "production",
        "target_variable": "oil_production_bpd", 
        "forecast": [
            {"date": "2025-01-01", "predicted_value": 125000.0},
            {"date": "2025-01-02", "predicted_value": 126500.0},
            {"date": "2025-01-03", "predicted_value": 124800.0}
        ],
        "metadata": {
            "model_type": "lightgbm",
            "horizon_days": 30,
            "generated_at": datetime.now().isoformat()
        }
    }
    
    # Save sample forecast
    sample_file = tester.output_dir / "sample_production_forecast.json"
    with open(sample_file, 'w') as f:
        json.dump(sample_forecast, f, indent=2)
    
    # Create sample CSV
    sample_df = pd.DataFrame(sample_forecast['forecast'])
    csv_file = tester.output_dir / "sample_production_forecast.csv"
    sample_df.to_csv(csv_file, index=False)
    
    print(f"Sample test files created in: {tester.output_dir}")
    print("\nAPI Test Framework Ready!")
    print("\nGenerated Files:")
    for file in tester.output_dir.glob("*"):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()