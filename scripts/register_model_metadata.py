#!/usr/bin/env python3
"""
Register Model Card, Tags, and Specs for Oil & Gas Production Forecasting Champion Model

This script updates the registered model 'oil_gas_production_forecasting_champion' with:
- Professional model card documentation
- Relevant tags for categorization and searchability  
- Technical specifications for governance and compliance

Compatible with Domino Model Registry and MLflow tracking.
"""

import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_mlflow():
    """Setup MLflow tracking"""
    mlflow.set_tracking_uri("http://localhost:8768")
    return MlflowClient()

def register_model_card(client, model_name, model_card_path):
    """Register model card markdown description"""
    try:
        # Read model card content
        with open(model_card_path, 'r', encoding='utf-8') as file:
            markdown_description = file.read()
        
        # Update registered model with model card
        client.update_registered_model(
            name=model_name,
            description=markdown_description
        )
        
        logger.info(f"✓ Successfully registered model card for '{model_name}'")
        logger.info(f"Model card source: {model_card_path}")
        
    except Exception as e:
        logger.error(f"Error registering model card: {e}")
        raise

def add_model_tags(client, model_name):
    """Add relevant tags to the registered model using proper tag format"""
    
    # Define 5 relevant tags for oil & gas forecasting model (not performance metrics)
    tags = {
        "risk-level": "medium",
        "industry": "oil_and_gas", 
        "model-category": "time_series_forecasting",
        "business-criticality": "high",
        "data-sensitivity": "internal"
    }
    
    try:
        # Check if model exists, if not create it with tags
        try:
            model = client.get_registered_model(model_name)
            logger.info(f"Model '{model_name}' already exists, adding tags...")
        except:
            logger.info(f"Model '{model_name}' not found, but it should exist from previous registration")
        
        # Add tags using set_registered_model_tag
        for key, value in tags.items():
            client.set_registered_model_tag(model_name, key, value)
            logger.info(f"✓ Added tag: {key} = {value}")
        
        logger.info(f"Successfully added {len(tags)} tags to '{model_name}'")
        return tags
        
    except Exception as e:
        logger.error(f"Error adding model tags: {e}")
        raise

def add_version_tags(client, model_name, model_version):
    """Add performance tags to specific model version"""
    
    # Get the best run metrics from the champion model version
    try:
        run_id = model_version.run_id
        run = client.get_run(run_id)
        run_metrics = run.data.metrics
        run_params = run.data.params
        
        # Version tags (top 5 most relevant performance metrics) - these go on the model VERSION
        version_tags = {
            "mae": f"{run_metrics.get('best_mae', run_metrics.get('mae', 0)):.4f}",
            "rmse": f"{run_metrics.get('rmse', 0):.4f}",
            "mape": f"{run_metrics.get('mape', 0):.2f}",
            "training_time_min": f"{(run.info.end_time - run.info.start_time) / 60000:.2f}" if run.info.end_time else "unknown",
            "champion_framework": run_params.get('framework', run_params.get('best_config', 'automl_comparison'))
        }
        
        # Add version tags using set_model_version_tag
        for key, value in version_tags.items():
            try:
                client.set_model_version_tag(model_name, model_version.version, key, value)
                logger.info(f"✓ Added version tag: {key} = {value}")
            except Exception as tag_error:
                logger.warning(f"Could not add version tag {key}: {tag_error}")
        
        logger.info(f"Successfully added version tags to '{model_name}' version {model_version.version}")
        return version_tags
        
    except Exception as e:
        logger.error(f"Error retrieving run metrics for version tags: {e}")
        
        # Fallback version tags if metrics can't be retrieved
        fallback_tags = {
            "status": "champion",
            "framework": "automl_comparison", 
            "validation": "temporal_split",
            "selection_method": "automated_mae",
            "data_type": "oil_production"
        }
        
        try:
            for key, value in fallback_tags.items():
                client.set_model_version_tag(model_name, model_version.version, key, value)
                logger.info(f"✓ Added fallback version tag: {key} = {value}")
            logger.info(f"Added {len(fallback_tags)} fallback version tags")
            return fallback_tags
        except Exception as e2:
            logger.error(f"Error adding fallback version tags: {e2}")
            return {}

def add_model_specs(client, model_name):
    """Add technical specifications using Domino specs format"""
    
    # Model Specs tags (registered model level) - use mlflow.domino.specs.* prefix
    model_specs = {
        "mlflow.domino.specs.Framework": "AutoML Time Series Comparison",
        "mlflow.domino.specs.Evaluation Metric": "Mean Absolute Error (MAE)",
        "mlflow.domino.specs.Training Method": "Parallel Framework Comparison via Domino Flows",
        "mlflow.domino.specs.Model Selection": "Automated Champion Selection (Lowest MAE)",
        "mlflow.domino.specs.Retraining Schedule": "Monthly or on Performance Degradation"
    }
    
    try:
        for key, value in model_specs.items():
            client.set_registered_model_tag(model_name, key, value)
            logger.info(f"✓ Added spec: {key.replace('mlflow.domino.specs.', '')} = {value}")
        
        logger.info(f"Successfully added {len(model_specs)} specifications to '{model_name}'")
        return model_specs
        
    except Exception as e:
        logger.error(f"Error adding model specs: {e}")
        raise

def main():
    """Main function to register model metadata"""
    try:
        # Configuration
        model_name = "oil_gas_production_forecasting_champion"
        model_card_path = Path("/mnt/code/model_card_short.md")
        
        logger.info("=== Domino Model Registry Metadata Registration ===")
        logger.info(f"Target Model: {model_name}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        
        # Setup MLflow client
        client = setup_mlflow()
        
        # Verify model exists
        try:
            model = client.get_registered_model(model_name)
            logger.info(f"✓ Found registered model: {model_name}")
            logger.info(f"Current version count: {len(model.latest_versions)}")
        except Exception as e:
            logger.error(f"Model '{model_name}' not found in registry: {e}")
            logger.info("Please ensure the model has been registered first by running the model comparison script")
            return
        
        # Verify model card file exists
        if not model_card_path.exists():
            logger.error(f"Model card file not found: {model_card_path}")
            return
        
        # 1. Register model card
        logger.info("\n1. Registering Model Card...")
        register_model_card(client, model_name, model_card_path)
        
        # 2. Add model-level tags
        logger.info("\n2. Adding Model Tags...")
        model_tags = add_model_tags(client, model_name)
        
        # 3. Add version-level tags (get latest version)
        logger.info("\n3. Adding Version Tags...")
        latest_version = model.latest_versions[0] if model.latest_versions else None
        if latest_version:
            version_tags = add_version_tags(client, model_name, latest_version)
        else:
            logger.warning("No model versions found - skipping version tags")
            version_tags = {}
        
        # 4. Add technical specifications  
        logger.info("\n4. Adding Technical Specifications...")
        model_specs = add_model_specs(client, model_name)
        
        logger.info("\n=== Model Metadata Registration Complete ===")
        logger.info(f"✓ Model card, tags, and specs successfully added to '{model_name}'")
        logger.info(f"✓ View in Domino Model Registry: Models > {model_name}")
        
        # Display summary
        logger.info("\n=== Summary ===")
        logger.info("Model-Level Tags Added:")
        if 'model_tags' in locals():
            for key, value in model_tags.items():
                logger.info(f"  - {key}: {value}")
        
        if version_tags:
            logger.info("\nVersion-Level Tags Added:")
            for key, value in version_tags.items():
                logger.info(f"  - {key}: {value}")
        
        logger.info("\nTechnical Specifications Added:")
        if 'model_specs' in locals():
            for key, value in model_specs.items():
                spec_name = key.replace('mlflow.domino.specs.', '')
                logger.info(f"  - {spec_name}: {value}")
        
        logger.info("\nNote: Version tags contain performance metrics, model tags contain governance/business metadata")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()