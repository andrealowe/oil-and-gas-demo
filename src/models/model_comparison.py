#!/usr/bin/env python3
"""
Model Comparison Script for Oil and Gas Forecasting AutoML Experiment

Compares results from AutoGluon, Prophet/NeuralProphet, Nixtla, and the combined LightGBM+ARIMA model.
Selects the best performing model and registers it in the Domino Model Registry.
"""

import sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import logging
from pathlib import Path
from datetime import datetime
import json

# Add scripts directory to path for data_config import
sys.path.insert(0, '/mnt/code')
from scripts.data_config import get_data_paths
from src.models.forecasting_config import ForecastingConfig
from src.models.workflow_io import WorkflowIO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_mlflow():
    """Setup MLflow tracking"""
    mlflow.set_tracking_uri("http://localhost:8768")
    experiment_name = 'oil_gas_forecasting_models'
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created MLflow experiment: {experiment_name}")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing MLflow experiment: {experiment_name}")
        
        mlflow.set_experiment(experiment_name)
        return experiment_id
    except Exception as e:
        logger.error(f"Error setting up MLflow: {e}")
        raise

def get_experiment_runs(experiment_name):
    """Get all runs from the forecasting experiment"""
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            logger.error(f"Experiment {experiment_name} not found")
            return []
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="",
            order_by=["metrics.mae ASC"]
        )
        
        logger.info(f"Found {len(runs)} runs in experiment {experiment_name}")
        return runs
        
    except Exception as e:
        logger.error(f"Error retrieving experiment runs: {e}")
        return []

def extract_run_metrics(runs):
    """Extract and organize metrics from MLflow runs"""
    model_results = []
    
    for run in runs:
        run_data = {
            'run_id': run.info.run_id,
            'run_name': run.data.tags.get('mlflow.runName', 'Unknown'),
            'model_type': run.data.tags.get('model_type', 'unknown'),
            'status': run.info.status,
            'start_time': run.info.start_time,
            'end_time': run.info.end_time,
            'metrics': {}
        }
        
        # Extract metrics
        for metric_name, metric_value in run.data.metrics.items():
            run_data['metrics'][metric_name] = metric_value
        
        # Extract parameters
        run_data['parameters'] = run.data.params
        
        # Extract tags
        run_data['tags'] = run.data.tags
        
        model_results.append(run_data)
    
    return model_results

def categorize_models(model_results):
    """Categorize models by framework type using standardized tags"""
    categories = {
        'autogluon': [],
        'prophet': [],
        'nixtla': [],
        'combined': []
    }
    
    for result in model_results:
        # Use standardized framework tag if available
        framework = result.get('tags', {}).get('framework', '').lower()
        
        # Fallback to legacy categorization
        if not framework:
            model_type = result.get('model_type', '').lower()
            run_name = result.get('run_name', '').lower()
            
            if 'autogluon' in model_type or 'autogluon' in run_name:
                framework = 'autogluon'
            elif 'prophet' in model_type or 'neuralprophet' in model_type or 'prophet' in run_name:
                framework = 'prophet'
            elif 'nixtla' in model_type or 'neuralforecast' in model_type or 'nixtla' in run_name:
                framework = 'nixtla'
            elif 'lightgbm' in run_name or 'oil_gas_forecasting' in run_name or 'combined' in run_name:
                framework = 'combined'
        
        # Categorize using framework
        if framework in categories:
            categories[framework].append(result)
        else:
            logger.warning(f"Unknown framework for run {result.get('run_name', 'unknown')}: {framework}")
    
    return categories

def find_best_model_per_category(categories):
    """Find the best performing model in each category using standardized primary metric"""
    best_models = {}
    primary_metric = ForecastingConfig.PRIMARY_METRIC
    
    for category, models in categories.items():
        if not models:
            logger.warning(f"No models found in category: {category}")
            continue
        
        # Look for models already tagged as best in framework
        best_tagged_models = [m for m in models if m.get('tags', {}).get('best_model_in_framework') == 'true']
        
        if best_tagged_models:
            # Use the tagged best model
            best_model = best_tagged_models[0]
            best_score = best_model['metrics'].get(primary_metric, float('inf'))
            logger.info(f"Found tagged best {category} model: {best_model['run_name']} ({primary_metric.upper()}: {best_score:.4f})")
        else:
            # Find model with lowest primary metric
            best_model = None
            best_score = float('inf')
            
            for model in models:
                score = model['metrics'].get(primary_metric)
                if score is not None and score < best_score:
                    best_score = score
                    best_model = model
            
            if best_model:
                logger.info(f"Best {category} model: {best_model['run_name']} ({primary_metric.upper()}: {best_score:.4f})")
        
        if best_model:
            best_models[category] = best_model
    
    return best_models

def compare_models(best_models):
    """Compare best models across categories and select overall winner"""
    if not best_models:
        logger.error("No models to compare")
        return None
    
    # Create comparison DataFrame
    comparison_data = []
    
    for category, model in best_models.items():
        metrics = model['metrics']
        comparison_data.append({
            'category': category,
            'run_id': model['run_id'],
            'run_name': model['run_name'],
            'mae': metrics.get('mae', float('inf')),
            'rmse': metrics.get('rmse', float('inf')),
            'mape': metrics.get('mape', float('inf')),
            'training_status': model['tags'].get('training_status', 'unknown'),
            'start_time': model['start_time'],
            'end_time': model['end_time']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by MAE (primary), RMSE (secondary), MAPE (tertiary)
    comparison_df = comparison_df.sort_values(['mae', 'rmse', 'mape'])
    
    # Find overall best model
    overall_best = comparison_df.iloc[0].to_dict()
    
    logger.info("=== MODEL COMPARISON RESULTS ===")
    logger.info(f"Overall Best Model: {overall_best['run_name']} ({overall_best['category']})")
    logger.info(f"MAE: {overall_best['mae']:.4f}, RMSE: {overall_best['rmse']:.4f}, MAPE: {overall_best['mape']:.2f}%")
    
    return overall_best, comparison_df

def register_best_model(best_model, model_results):
    """Register the best model in Domino Model Registry"""
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Model registry name
        model_name = "oil_gas_production_forecasting_champion"
        
        # Get the run details
        run_id = best_model['run_id']
        run = client.get_run(run_id)
        
        logger.info(f"Registering model from run: {run_id}")
        
        # Find model artifact
        artifacts = client.list_artifacts(run_id)
        model_artifact_path = None
        
        for artifact in artifacts:
            if artifact.path.endswith('.pkl') and 'model' in artifact.path:
                model_artifact_path = artifact.path
                break
        
        if not model_artifact_path:
            logger.warning("No model artifact found, creating model registry entry with run reference")
            model_artifact_path = "model"
        
        # Create or get registered model
        try:
            registered_model = client.create_registered_model(
                name=model_name,
                tags={
                    "project": "oil_gas_analytics",
                    "use_case": "production_forecasting",
                    "framework": best_model['category'],
                    "champion_selection_date": datetime.now().isoformat()
                }
            )
            logger.info(f"Created registered model: {model_name}")
        except mlflow.exceptions.RestException as e:
            if "RESOURCE_ALREADY_EXISTS" in str(e):
                registered_model = client.get_registered_model(model_name)
                logger.info(f"Using existing registered model: {model_name}")
            else:
                raise
        
        # Create model version
        model_version = client.create_model_version(
            name=model_name,
            source=f"runs:/{run_id}/{model_artifact_path}",
            run_id=run_id,
            tags={
                "validation_mae": str(best_model['mae']),
                "validation_rmse": str(best_model['rmse']),
                "validation_mape": str(best_model['mape']),
                "model_category": best_model['category'],
                "selection_criteria": "lowest_mae",
                "registered_at": datetime.now().isoformat()
            }
        )
        
        logger.info(f"Created model version {model_version.version} for {model_name}")
        
        # Transition to Production stage
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Production",
            archive_existing_versions=True
        )
        
        logger.info(f"Transitioned model version {model_version.version} to Production stage")
        
        return {
            'model_name': model_name,
            'model_version': model_version.version,
            'run_id': run_id,
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"Error registering model: {e}")
        return {
            'model_name': None,
            'model_version': None,
            'run_id': run_id,
            'status': 'failed',
            'error': str(e)
        }

def save_comparison_results(comparison_df, best_model, registration_result):
    """Save comparison results to artifacts"""
    try:
        # Get data paths
        paths = get_data_paths('Oil-and-Gas-Demo')
        artifacts_dir = paths['artifacts_path']
        results_dir = artifacts_dir / 'model_comparison'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comparison DataFrame
        comparison_path = results_dir / 'model_comparison_results.csv'
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"Saved comparison results to: {comparison_path}")
        
        # Save detailed results
        results_summary = {
            'comparison_timestamp': datetime.now().isoformat(),
            'total_models_compared': len(comparison_df),
            'best_model': {
                'category': best_model['category'],
                'run_name': best_model['run_name'],
                'run_id': best_model['run_id'],
                'mae': best_model['mae'],
                'rmse': best_model['rmse'],
                'mape': best_model['mape']
            },
            'model_registration': registration_result,
            'comparison_criteria': 'Lowest Mean Absolute Error (MAE)',
            'models_by_category': {
                category: len(comparison_df[comparison_df['category'] == category])
                for category in comparison_df['category'].unique()
            }
        }
        
        summary_path = results_dir / 'comparison_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        logger.info(f"Saved summary to: {summary_path}")
        
        return results_summary
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return None

def compare_workflow_summaries(summaries):
    """
    Compare training summaries from workflow inputs

    Args:
        summaries: Dict of {framework_name: summary_data}

    Returns:
        Tuple of (best_summary, comparison_data)
    """
    logger.info("Comparing models from workflow input summaries")

    valid_summaries = {}
    for name, summary in summaries.items():
        if summary and summary.get('status') != 'error' and summary.get('best_mae') is not None:
            valid_summaries[name] = summary
            logger.info(f"{name}: best_mae={summary.get('best_mae'):.4f}, config={summary.get('best_config')}")
        else:
            logger.warning(f"Skipping {name}: invalid or error summary")

    if not valid_summaries:
        logger.error("No valid summaries to compare")
        return None, None

    # Find overall best based on MAE
    best_framework = min(valid_summaries.items(), key=lambda x: x[1]['best_mae'])
    champion_name, champion_summary = best_framework

    # Create comparison data
    comparison_data = {
        'champion_framework': champion_name,
        'champion_config': champion_summary.get('best_config'),
        'champion_mae': champion_summary.get('best_mae'),
        'all_frameworks': {
            name: {
                'best_mae': s.get('best_mae'),
                'best_config': s.get('best_config'),
                'total_configs': s.get('total_configs', 0),
                'successful_configs': s.get('successful_configs', 0)
            }
            for name, s in valid_summaries.items()
        }
    }

    return champion_summary, comparison_data

def read_workflow_input(name: str):
    """Read input from workflow, return as dict"""
    p = Path(f"/workflow/inputs/{name}")
    if not p.exists():
        return None

    content = p.read_text().strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        logger.warning(f"Could not parse {name} as JSON")
        return None

def main():
    """Main function to run model comparison"""
    try:
        logger.info("Starting model comparison for Oil & Gas forecasting experiment")

        # Check if running in workflow mode by checking for input files
        autogluon_summary = read_workflow_input("autogluon_summary")
        prophet_summary = read_workflow_input("prophet_summary")
        nixtla_summary = read_workflow_input("nixtla_summary")
        combined_summary = read_workflow_input("combined_summary")

        # If any inputs exist, we're in workflow mode
        if any([autogluon_summary, prophet_summary, nixtla_summary, combined_summary]):
            logger.info("Running in Domino Flow mode - using workflow inputs")

            summaries = {
                'autogluon': autogluon_summary,
                'prophet_neuralprophet': prophet_summary,
                'nixtla_neuralforecast': nixtla_summary,
                'combined_lightgbm_arima': combined_summary
            }

            logger.info("Loaded training summaries from workflow inputs")

            # Compare summaries directly
            champion_summary, comparison_data = compare_workflow_summaries(summaries)

            if champion_summary is None:
                logger.error("No valid models to compare")
                # Still write an output
                error_result = {
                    'timestamp': datetime.now().isoformat(),
                    'status': 'error',
                    'error': 'No valid models found for comparison',
                    'champion_framework': None,
                    'champion_mae': None
                }
                # Write error output using WorkflowIO
                wf_io = WorkflowIO()
                wf_io.write_output("comparison_results", error_result)
                return

            # Create results summary for workflow output
            results_summary = {
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'champion_framework': comparison_data['champion_framework'],
                'champion_config': comparison_data['champion_config'],
                'champion_mae': comparison_data['champion_mae'],
                'all_frameworks': comparison_data['all_frameworks'],
                'total_frameworks_evaluated': len([s for s in summaries.values() if s and s.get('status') != 'error']),
                'registration_status': 'skipped_in_workflow_mode'
            }

            logger.info("=== COMPARISON COMPLETE (Workflow Mode) ===")
            logger.info(f"Champion Framework: {comparison_data['champion_framework']}")
            logger.info(f"Champion Config: {comparison_data['champion_config']}")
            logger.info(f"Champion MAE: {comparison_data['champion_mae']:.4f}")

            # Write results using WorkflowIO
            wf_io = WorkflowIO()
            wf_io.write_output("comparison_results", results_summary)

            return results_summary

        # STANDALONE MODE: Use MLflow to get experiment runs
        logger.info("Running in standalone mode - querying MLflow experiments")

        # Setup MLflow
        experiment_id = setup_mlflow()
        experiment_name = 'oil_gas_forecasting_models'

        # Start comparison run
        with mlflow.start_run(run_name="model_comparison_and_selection") as comparison_run:
            mlflow.set_tag("comparison_type", "automl_forecasting")
            mlflow.set_tag("data_source", "oil_gas_production")
            mlflow.set_tag("selection_criteria", "lowest_mae")

            # Get all experiment runs
            runs = get_experiment_runs(experiment_name)

            if not runs:
                logger.error("No runs found in experiment")
                return

            mlflow.log_param("total_runs_evaluated", len(runs))
            
            # Extract metrics from runs
            model_results = extract_run_metrics(runs)
            
            # Categorize models by framework
            categories = categorize_models(model_results)
            
            # Log category counts
            for category, models in categories.items():
                mlflow.log_param(f"{category}_count", len(models))
                logger.info(f"Found {len(models)} {category} models")
            
            # Find best model per category
            best_models = find_best_model_per_category(categories)
            
            if not best_models:
                logger.error("No valid models found for comparison")
                return
            
            # Compare models and select overall best
            overall_best, comparison_df = compare_models(best_models)
            
            # Log comparison metrics
            mlflow.log_metric("champion_mae", overall_best['mae'])
            mlflow.log_metric("champion_rmse", overall_best['rmse'])
            mlflow.log_metric("champion_mape", overall_best['mape'])
            mlflow.log_param("champion_category", overall_best['category'])
            mlflow.log_param("champion_run_name", overall_best['run_name'])
            
            # Register best model
            registration_result = register_best_model(overall_best, model_results)
            
            # Log registration results
            mlflow.log_param("registration_status", registration_result['status'])
            if registration_result['status'] == 'success':
                mlflow.log_param("registered_model_name", registration_result['model_name'])
                mlflow.log_param("registered_model_version", registration_result['model_version'])
            
            # Save results
            results_summary = save_comparison_results(comparison_df, overall_best, registration_result)
            
            # Log artifacts
            if results_summary:
                paths = get_data_paths('Oil-and-Gas-Demo')
                results_dir = paths['artifacts_path'] / 'model_comparison'
                
                mlflow.log_artifact(str(results_dir / 'model_comparison_results.csv'))
                mlflow.log_artifact(str(results_dir / 'comparison_summary.json'))
            
            logger.info("=== COMPARISON COMPLETE (Standalone Mode) ===")
            logger.info(f"Champion Model: {overall_best['run_name']} ({overall_best['category']})")
            logger.info(f"Performance: MAE={overall_best['mae']:.4f}")

            if registration_result['status'] == 'success':
                logger.info(f"Registered as: {registration_result['model_name']} v{registration_result['model_version']}")
            else:
                logger.warning(f"Registration failed: {registration_result.get('error', 'Unknown error')}")

            return results_summary
            
    except Exception as e:
        logger.error(f"Error in model comparison: {e}")
        # CRITICAL: Write error output for Flow execution using WorkflowIO
        wf_io = WorkflowIO()
        wf_io.write_error_output("comparison_results", e, "model_comparison")
        raise

if __name__ == "__main__":
    main()