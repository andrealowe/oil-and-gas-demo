#!/usr/bin/env python3
"""
Model Cards Generator for Oil & Gas Forecasting Models

Creates comprehensive model cards for governance and compliance.
Integrates with MLflow to extract model performance and metadata.
"""

import pandas as pd
import numpy as np
import json
import mlflow
import mlflow.tracking
from datetime import datetime
from pathlib import Path
import sys
import logging

# Add scripts directory to path
sys.path.insert(0, '/mnt/code')
from scripts.data_config import get_data_paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelCardGenerator:
    """Generate model cards for oil & gas forecasting models"""
    
    def __init__(self, project_name="Oil-and-Gas-Demo"):
        self.project_name = project_name
        self.paths = get_data_paths(project_name)
        self.artifacts_dir = self.paths['artifacts_path']
        
        # MLflow setup
        mlflow.set_tracking_uri("http://localhost:8768")
        self.experiment_name = "oil_gas_forecasting_models"
        
        # Output directory
        self.output_dir = self.artifacts_dir / "model_cards"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_model_metadata(self):
        """Extract model metadata from MLflow"""
        client = mlflow.tracking.MlflowClient()
        
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if not experiment:
                logger.error(f"Experiment '{self.experiment_name}' not found")
                return {}
                
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"]
            )
            
            models_metadata = {}
            
            for run in runs:
                run_data = run.data
                run_info = run.info
                
                # Extract model information
                model_category = run_data.tags.get('model_category', 'unknown')
                model_name = run_info.run_name
                
                if model_category != 'unknown':
                    models_metadata[model_name] = {
                        'run_id': run_info.run_id,
                        'category': model_category,
                        'metrics': run_data.metrics,
                        'params': run_data.params,
                        'tags': run_data.tags,
                        'start_time': run_info.start_time,
                        'end_time': run_info.end_time,
                        'status': run_info.status
                    }
            
            logger.info(f"Extracted metadata for {len(models_metadata)} models")
            return models_metadata
            
        except Exception as e:
            logger.error(f"Failed to extract model metadata: {e}")
            return {}
    
    def generate_production_model_card(self, metadata):
        """Generate model card for production forecasting models"""
        
        production_models = {k: v for k, v in metadata.items() 
                           if 'production' in v.get('category', '')}
        
        if not production_models:
            return None
            
        card_content = {
            "model_name": "Oil & Gas Production Forecasting Suite",
            "model_type": "Time Series Forecasting",
            "use_case": "Production Planning and Operations Optimization",
            "version": "1.0.0",
            "created_date": datetime.now().isoformat(),
            
            "model_description": {
                "overview": "Comprehensive forecasting models for oil and gas production planning",
                "target_variables": ["oil_production_bpd", "gas_production_mcfd"],
                "forecast_horizons": ["30 days (short-term)", "180 days (medium-term)"],
                "algorithms": ["LightGBM", "Prophet"],
                "features": [
                    "Time-based features (year, month, day, seasonality)",
                    "Lag features (1, 7, 30 days)",
                    "Rolling statistics (7, 30 day windows)",
                    "Cyclical encoding for seasonality"
                ]
            },
            
            "business_context": {
                "business_objective": "Enable accurate production forecasting for operational planning and resource allocation",
                "success_criteria": [
                    "MAPE < 15% for short-term forecasts",
                    "MAPE < 25% for medium-term forecasts",
                    "Real-time inference capability",
                    "Confidence interval availability"
                ],
                "stakeholders": [
                    "Production Operations Team",
                    "Supply Chain Planning",
                    "Revenue Forecasting",
                    "Executive Leadership"
                ]
            },
            
            "performance_metrics": {},
            "model_specifications": {},
            
            "risk_assessment": {
                "risk_level": "Medium",
                "key_risks": [
                    "Market volatility impact on production decisions",
                    "Model drift due to changing operational conditions",
                    "Data quality dependencies",
                    "Forecast accuracy degradation over time"
                ],
                "mitigation_strategies": [
                    "Regular model retraining schedule",
                    "Comprehensive monitoring and alerting",
                    "Multiple model ensemble approach",
                    "Expert judgment override capability"
                ]
            },
            
            "data_governance": {
                "data_sources": ["Production time series data"],
                "data_quality": "Validated historical production records",
                "data_retention": "3+ years of historical data",
                "privacy_considerations": "Aggregated operational data only",
                "compliance_requirements": ["SOX", "Internal Risk Management"]
            },
            
            "monitoring_plan": {
                "performance_monitoring": [
                    "Daily forecast accuracy tracking",
                    "Weekly model performance reports",
                    "Monthly model drift analysis"
                ],
                "alert_conditions": [
                    "MAPE > 20% for consecutive days",
                    "Prediction service downtime",
                    "Data pipeline failures"
                ],
                "retraining_triggers": [
                    "Performance degradation > 5%",
                    "New facility additions",
                    "Quarterly schedule"
                ]
            }
        }
        
        # Add specific model performance data
        for model_name, model_data in production_models.items():
            metrics = model_data.get('metrics', {})
            params = model_data.get('params', {})
            
            model_key = f"{model_name}_{model_data['run_id'][:8]}"
            
            card_content["performance_metrics"][model_key] = {
                "mae": metrics.get('lgb_mae', metrics.get('prophet_mae', 'N/A')),
                "rmse": metrics.get('lgb_rmse', metrics.get('prophet_rmse', 'N/A')),
                "mape": metrics.get('lgb_mape', metrics.get('prophet_mape', 'N/A')),
                "target_variable": params.get('target_variable', 'N/A'),
                "forecast_horizon": params.get('forecast_horizon', 'N/A')
            }
            
            card_content["model_specifications"][model_key] = {
                "algorithm": "LightGBM" if 'lgb' in model_name else "Prophet",
                "parameters": params,
                "training_time": model_data.get('end_time', 0) - model_data.get('start_time', 0)
            }
        
        return card_content
    
    def generate_price_model_card(self, metadata):
        """Generate model card for price forecasting models"""
        
        price_models = {k: v for k, v in metadata.items() 
                       if 'price' in v.get('category', '')}
        
        if not price_models:
            return None
            
        card_content = {
            "model_name": "Oil & Gas Price Forecasting Suite",
            "model_type": "Time Series Forecasting",
            "use_case": "Market Analysis and Revenue Planning",
            "version": "1.0.0",
            "created_date": datetime.now().isoformat(),
            
            "model_description": {
                "overview": "Comprehensive price forecasting models for oil and gas commodities",
                "target_variables": [
                    "crude_oil_price_usd_bbl", "natural_gas_price_usd_mcf", 
                    "brent_crude_usd_bbl", "wti_crude_usd_bbl"
                ],
                "forecast_horizons": ["30 days (short-term)", "180 days (medium-term)"],
                "algorithms": ["LightGBM", "Prophet"],
                "market_coverage": "Global commodity markets"
            },
            
            "business_context": {
                "business_objective": "Support revenue forecasting and risk management through accurate price predictions",
                "success_criteria": [
                    "MAPE < 20% for short-term price forecasts",
                    "MAPE < 30% for medium-term price forecasts", 
                    "Early warning for significant price movements",
                    "Integration with trading systems"
                ],
                "stakeholders": [
                    "Trading and Marketing Team",
                    "Risk Management",
                    "Finance and Accounting",
                    "Strategic Planning"
                ]
            },
            
            "performance_metrics": {},
            "model_specifications": {},
            
            "risk_assessment": {
                "risk_level": "High",
                "key_risks": [
                    "High market volatility and unpredictability",
                    "Geopolitical events impact",
                    "Model performance sensitivity to market regime changes",
                    "Financial decision dependency on forecasts"
                ],
                "mitigation_strategies": [
                    "Ensemble modeling approach",
                    "Volatility-aware confidence intervals",
                    "Regular expert review and validation",
                    "Scenario-based stress testing"
                ]
            },
            
            "data_governance": {
                "data_sources": ["Market price feeds", "External commodity data"],
                "data_quality": "Real-time validated market data",
                "data_retention": "10+ years of historical prices",
                "privacy_considerations": "Public market data",
                "compliance_requirements": ["Market Data Licensing", "Financial Reporting"]
            }
        }
        
        # Add specific model performance data
        for model_name, model_data in price_models.items():
            metrics = model_data.get('metrics', {})
            params = model_data.get('params', {})
            
            model_key = f"{model_name}_{model_data['run_id'][:8]}"
            
            card_content["performance_metrics"][model_key] = {
                "mae": metrics.get('lgb_mae', metrics.get('prophet_mae', 'N/A')),
                "rmse": metrics.get('lgb_rmse', metrics.get('prophet_rmse', 'N/A')),
                "mape": metrics.get('lgb_mape', metrics.get('prophet_mape', 'N/A')),
                "target_variable": params.get('target_variable', 'N/A'),
                "forecast_horizon": params.get('forecast_horizon', 'N/A')
            }
        
        return card_content
    
    def generate_demand_model_card(self, metadata):
        """Generate model card for demand forecasting models"""
        
        demand_models = {k: v for k, v in metadata.items() 
                        if 'demand' in v.get('category', '')}
        
        if not demand_models:
            return None
            
        card_content = {
            "model_name": "Oil & Gas Demand Forecasting Suite",
            "model_type": "Time Series Forecasting",
            "use_case": "Market Analysis and Supply Planning",
            "version": "1.0.0",
            "created_date": datetime.now().isoformat(),
            
            "model_description": {
                "overview": "Regional demand forecasting models for oil and gas products",
                "target_variables": [
                    "oil_demand_thousand_bpd", "gas_demand_thousand_mcfd",
                    "gasoline_demand_thousand_bpd", "diesel_demand_thousand_bpd"
                ],
                "regional_coverage": ["North America", "Europe", "Asia Pacific"],
                "forecast_horizons": ["30 days (short-term)", "180 days (medium-term)"],
                "algorithms": ["LightGBM", "Prophet"]
            },
            
            "business_context": {
                "business_objective": "Optimize supply chain and inventory management through demand forecasting",
                "success_criteria": [
                    "MAPE < 18% for regional demand forecasts",
                    "Early detection of demand shifts", 
                    "Support for capacity planning",
                    "Integration with logistics optimization"
                ],
                "stakeholders": [
                    "Supply Chain Operations",
                    "Marketing and Sales",
                    "Strategic Planning",
                    "Regional Business Units"
                ]
            },
            
            "performance_metrics": {},
            "model_specifications": {},
            
            "risk_assessment": {
                "risk_level": "Medium",
                "key_risks": [
                    "Economic cycle impact on demand",
                    "Seasonal variation complexity",
                    "Regional market differences",
                    "Supply chain planning dependency"
                ]
            }
        }
        
        return card_content
    
    def generate_maintenance_model_card(self, metadata):
        """Generate model card for maintenance forecasting models"""
        
        maintenance_models = {k: v for k, v in metadata.items() 
                             if 'maintenance' in v.get('category', '')}
        
        if not maintenance_models:
            return None
            
        card_content = {
            "model_name": "Maintenance Scheduling Optimization Suite", 
            "model_type": "Time Series Forecasting",
            "use_case": "Predictive Maintenance and Resource Planning",
            "version": "1.0.0",
            "created_date": datetime.now().isoformat(),
            
            "model_description": {
                "overview": "Forecasting models for maintenance scheduling and cost optimization",
                "target_variables": ["duration_hours", "cost_usd"],
                "maintenance_types": ["Preventive", "Corrective", "Predictive"],
                "forecast_horizons": ["30 days (short-term)", "180 days (medium-term)"],
                "algorithms": ["LightGBM", "Prophet"]
            },
            
            "business_context": {
                "business_objective": "Optimize maintenance operations and resource allocation",
                "success_criteria": [
                    "Reduce unplanned downtime by 20%",
                    "Optimize maintenance cost planning",
                    "Improve resource scheduling efficiency",
                    "Support predictive maintenance programs"
                ],
                "stakeholders": [
                    "Maintenance Operations",
                    "Asset Management", 
                    "Operations Planning",
                    "Finance and Budgeting"
                ]
            },
            
            "performance_metrics": {},
            "model_specifications": {},
            
            "risk_assessment": {
                "risk_level": "Medium-High",
                "key_risks": [
                    "Safety implications of maintenance delays",
                    "Production impact from maintenance scheduling",
                    "Cost overrun from poor planning",
                    "Equipment failure prediction accuracy"
                ]
            }
        }
        
        return card_content
    
    def generate_governance_summary(self, metadata):
        """Generate governance and compliance summary"""
        
        total_models = len(metadata)
        categories = set(v.get('category', 'unknown') for v in metadata.values())
        
        governance_summary = {
            "document_title": "Oil & Gas Forecasting Models - Governance Summary",
            "created_date": datetime.now().isoformat(),
            "version": "1.0.0",
            
            "model_inventory": {
                "total_models": total_models,
                "model_categories": list(categories),
                "framework_compliance": "NIST Risk Management Framework",
                "approval_status": "Pending Review"
            },
            
            "compliance_frameworks": {
                "applicable_frameworks": [
                    "Model Risk Management V3",
                    "NIST Risk Management Framework", 
                    "Internal AI Governance Policy"
                ],
                "risk_classification": "Medium-Risk Models",
                "approval_requirements": [
                    "Model validation review",
                    "Risk assessment approval",
                    "Performance monitoring plan",
                    "Data governance compliance"
                ]
            },
            
            "approval_workflow": {
                "required_approvers": [
                    "modeling-review (Technical Review)",
                    "modeling-leadership (Model Approval)",
                    "it-review (Infrastructure Review)",
                    "lob-leadership (Business Approval)"
                ],
                "approval_timeline": "2-3 weeks",
                "documentation_requirements": [
                    "Model cards for each model category",
                    "Performance validation reports",
                    "Risk assessment documentation",
                    "Monitoring and alerting plan"
                ]
            },
            
            "operational_requirements": {
                "monitoring": "Continuous performance monitoring required",
                "retraining": "Quarterly model refresh schedule",
                "validation": "Monthly model validation reviews", 
                "documentation": "Model registry maintenance",
                "incident_response": "24/7 support for production issues"
            },
            
            "risk_mitigation": {
                "key_controls": [
                    "Multi-model ensemble approach",
                    "Human expert override capability",
                    "Automated performance monitoring",
                    "Regular model validation",
                    "Comprehensive logging and audit trail"
                ],
                "escalation_procedures": [
                    "Performance degradation alerts",
                    "Model drift detection",
                    "Data quality issues",
                    "System availability problems"
                ]
            }
        }
        
        return governance_summary
    
    def generate_all_model_cards(self):
        """Generate all model cards and governance documentation"""
        logger.info("Generating model cards and governance documentation...")
        
        # Extract metadata
        metadata = self.extract_model_metadata()
        
        if not metadata:
            logger.warning("No model metadata found, creating template documentation")
            
        # Generate individual model cards
        cards = {}
        
        cards['production'] = self.generate_production_model_card(metadata)
        cards['prices'] = self.generate_price_model_card(metadata) 
        cards['demand'] = self.generate_demand_model_card(metadata)
        cards['maintenance'] = self.generate_maintenance_model_card(metadata)
        
        # Generate governance summary
        governance = self.generate_governance_summary(metadata)
        
        # Save all documents
        saved_files = []
        
        for card_type, card_content in cards.items():
            if card_content:
                filename = f"{card_type}_model_card.json"
                filepath = self.output_dir / filename
                
                with open(filepath, 'w') as f:
                    json.dump(card_content, f, indent=2, default=str)
                
                saved_files.append(str(filepath))
                logger.info(f"Generated {card_type} model card: {filepath}")
        
        # Save governance summary
        governance_path = self.output_dir / "governance_summary.json"
        with open(governance_path, 'w') as f:
            json.dump(governance, f, indent=2, default=str)
        saved_files.append(str(governance_path))
        
        # Create markdown summary for easy reading
        self.create_markdown_summary(cards, governance)
        
        logger.info(f"Generated {len(saved_files)} governance documents")
        return saved_files
    
    def create_markdown_summary(self, cards, governance):
        """Create a markdown summary of all model cards"""
        
        markdown_content = f"""# Oil & Gas Forecasting Models - Documentation Summary

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This document provides a comprehensive overview of the oil & gas forecasting model suite developed for dashboard integration and real-time decision support.

## Model Categories

"""
        
        for card_type, card in cards.items():
            if card:
                markdown_content += f"### {card_type.title()} Forecasting Models\n\n"
                markdown_content += f"**Model Name**: {card['model_name']}\n\n"
                markdown_content += f"**Use Case**: {card['use_case']}\n\n"
                markdown_content += f"**Business Objective**: {card['business_context']['business_objective']}\n\n"
                
                if 'target_variables' in card['model_description']:
                    variables = card['model_description']['target_variables']
                    markdown_content += f"**Target Variables**:\n"
                    for var in variables:
                        markdown_content += f"- {var}\n"
                    markdown_content += "\n"
                
                markdown_content += "---\n\n"
        
        markdown_content += f"""## Governance and Compliance

**Risk Classification**: {governance['compliance_frameworks']['risk_classification']}

**Applicable Frameworks**:
"""
        for framework in governance['compliance_frameworks']['applicable_frameworks']:
            markdown_content += f"- {framework}\n"
        
        markdown_content += f"""
**Required Approvers**:
"""
        for approver in governance['approval_workflow']['required_approvers']:
            markdown_content += f"- {approver}\n"
        
        markdown_content += f"""
## Model Performance

All models have been trained with comprehensive evaluation metrics including:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE) 
- Mean Absolute Percentage Error (MAPE)

Models support both short-term (30 days) and medium-term (180 days) forecasting horizons.

## API Integration

Models are deployed through FastAPI service endpoints for real-time forecasting:
- `/forecast/production/{{target}}`
- `/forecast/prices/{{commodity}}`
- `/forecast/demand/{{region}}`
- `/forecast/maintenance`

## Monitoring and Maintenance

- **Performance Monitoring**: Continuous tracking of forecast accuracy
- **Model Refresh**: Quarterly retraining schedule
- **Alert System**: Automated alerts for performance degradation
- **Validation**: Monthly model validation reviews

---

For detailed technical specifications, see individual model card JSON files.
"""
        
        # Save markdown summary
        summary_path = self.output_dir / "README.md"
        with open(summary_path, 'w') as f:
            f.write(markdown_content)
        
        logger.info(f"Generated markdown summary: {summary_path}")

def main():
    """Generate all model cards and governance documentation"""
    generator = ModelCardGenerator()
    saved_files = generator.generate_all_model_cards()
    
    print("Generated Model Cards and Governance Documentation:")
    for file_path in saved_files:
        print(f"  - {file_path}")
    
    print(f"\nDocumentation saved to: {generator.output_dir}")

if __name__ == "__main__":
    main()