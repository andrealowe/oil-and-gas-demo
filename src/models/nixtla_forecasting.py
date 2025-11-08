#!/usr/bin/env python3
"""
Robust Nixtla NeuralForecast for Oil and Gas Production Data

Tests Nixtla's neuralforecast with different neural network models on daily oil production.
Uses MLflow for experiment tracking with child runs for hyperparameter testing.

Compatible with both standalone execution and Domino Flows.
Includes robust error handling for common Nixtla/NeuralForecast issues.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import mlflow.sklearn
import joblib
import logging
from datetime import datetime, timedelta
import warnings
import argparse
import os
warnings.filterwarnings('ignore')

# Add scripts directory to path for data_config import
sys.path.insert(0, '/mnt/code')
from scripts.data_config import get_data_paths
# Removed ensure_data import - scripts now fail fast if data is missing
from src.models.forecasting_config import ForecastingConfig, get_standard_configs
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

def load_and_prepare_data():
    """Load and prepare oil production data for time series forecasting"""
    try:
        # Get data paths
        paths = get_data_paths('Oil-and-Gas-Demo')
        data_path = paths['base_data_path'] / 'production_timeseries.parquet'

        # Check if data file exists - fail fast if missing
        if not data_path.exists():
            available_files = list(paths['base_data_path'].glob('*.parquet')) if paths['base_data_path'].exists() else []
            error_msg = f"""
🚫 REQUIRED DATA FILE MISSING: {data_path.name}

Expected location: {data_path}
Data directory: {paths['base_data_path']}
Directory exists: {paths['base_data_path'].exists()}
Available files: {[f.name for f in available_files]}

🔧 SOLUTIONS:
- For local development: Run 'python scripts/oil_gas_data_generator.py' first
- For Domino Flows: Ensure the data generation task completed successfully
- For read-only datasets: Verify data was pre-generated and mounted correctly

❌ This script will NOT auto-generate missing data (fail-fast design)
            """
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.info(f"Loading data from: {data_path}")
        df = pd.read_parquet(data_path)
        
        logger.info(f"Original data shape: {df.shape}")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Aggregate daily oil production across all facilities
        daily_production = df.groupby('date').agg({
            'oil_production_bpd': 'sum'
        }).reset_index()
        
        # Rename columns for NeuralForecast format
        daily_production = daily_production.rename(columns={
            'date': 'ds',
            'oil_production_bpd': 'y'
        })
        
        # Add unique_id column (required by NeuralForecast)
        daily_production['unique_id'] = 'oil_production'
        
        # Reorder columns: unique_id, ds, y
        daily_production = daily_production[['unique_id', 'ds', 'y']]
        
        # Sort by date and ensure no gaps
        daily_production = daily_production.sort_values('ds')
        
        logger.info(f"Aggregated data shape: {daily_production.shape}")
        logger.info(f"Target statistics:\n{daily_production['y'].describe()}")
        
        return daily_production
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def calculate_metrics(y_true, y_pred):
    """Calculate forecasting metrics"""
    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {'mae': mae, 'rmse': rmse, 'mape': mape}
    
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {'mae': float('inf'), 'rmse': float('inf'), 'mape': float('inf')}

def create_neuralforecast_model(model_class, params, horizon):
    """Create a NeuralForecast model with robust error handling"""
    try:
        if model_class == 'MLP':
            from neuralforecast.models import MLP
            return MLP(h=horizon, **params)
        elif model_class == 'NBEATS':
            from neuralforecast.models import NBEATS
            return NBEATS(h=horizon, **params)
        elif model_class == 'NHITS':
            from neuralforecast.models import NHITS
            return NHITS(h=horizon, **params)
        elif model_class == 'LSTM':
            from neuralforecast.models import LSTM
            return LSTM(h=horizon, **params)
        elif model_class == 'TFT':
            # TFT (Temporal Fusion Transformer) requires transformers library
            # which has Keras 3 compatibility issues - skip for now
            logger.warning(f"TFT model skipped due to TensorFlow/Keras compatibility issues")
            logger.warning("Install tf-keras with: pip install tf-keras")
            return None
        else:
            logger.error(f"Unknown model class: {model_class}")
            return None
    except ImportError as e:
        logger.error(f"Could not import {model_class}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error creating {model_class} model: {e}")
        return None

def test_neuralforecast_config(data, train_end_date, config):
    """Test specific NeuralForecast configuration with robust error handling"""
    try:
        from neuralforecast import NeuralForecast
        
        # Prepare train/test split
        train_data = data[data['ds'] <= train_end_date].copy()
        test_data = data[data['ds'] > train_end_date].copy()
        
        if len(test_data) == 0:
            logger.warning(f"No test data available for {config['name']}")
            return None
        
        horizon = len(test_data)
        logger.info(f"Training {config['name']}: Train size: {len(train_data)}, Test size: {len(test_data)}, Horizon: {horizon}")
        
        # Create model with error handling
        model = create_neuralforecast_model(config['model_class'], config['params'], horizon)
        if model is None:
            logger.error(f"Failed to create model for {config['name']}")
            return None
        
        # Create NeuralForecast with robust configuration
        try:
            nf = NeuralForecast(
                models=[model],
                freq='D'
            )
        except Exception as e:
            logger.error(f"Error creating NeuralForecast instance: {e}")
            return None
        
        # Fit model with timeout and error handling
        try:
            logger.info(f"Starting training for {config['name']}...")
            nf.fit(train_data)
            logger.info(f"Training completed for {config['name']}")
        except Exception as e:
            logger.error(f"Training failed for {config['name']}: {e}")
            return None
        
        # Make predictions with error handling
        try:
            forecasts = nf.predict(h=horizon)
            logger.info(f"Prediction completed for {config['name']}")
        except Exception as e:
            logger.error(f"Prediction failed for {config['name']}: {e}")
            return None
        
        # Extract predictions with robust column detection
        pred_col = None
        for col in forecasts.columns:
            if col not in ['unique_id', 'ds'] and config['model_class'] in col:
                pred_col = col
                break
        
        # Fallback: use any non-id/date column
        if pred_col is None:
            for col in forecasts.columns:
                if col not in ['unique_id', 'ds']:
                    pred_col = col
                    break
        
        if pred_col is None:
            logger.error(f"Could not find prediction column in forecasts: {forecasts.columns.tolist()}")
            return None
        
        y_pred = forecasts[pred_col].values
        y_true = test_data['y'].values
        
        # Ensure same length
        min_length = min(len(y_true), len(y_pred))
        y_true = y_true[:min_length]
        y_pred = y_pred[:min_length]
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred)
        
        logger.info(f"{config['name']} metrics: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, MAPE={metrics['mape']:.2f}%")
        
        return {
            'model': nf,
            'metrics': metrics,
            'predictions': y_pred,
            'y_true': y_true,
            'forecasts': forecasts,
            'train_data': train_data,
            'test_data': test_data
        }
        
    except Exception as e:
        logger.error(f"Error in NeuralForecast config {config['name']}: {e}")
        return None

def create_fallback_models(horizon):
    """Create fallback statistical models if neural models fail"""
    try:
        from statsforecast.models import AutoARIMA, Naive, SeasonalNaive
        from statsforecast import StatsForecast
        
        fallback_configs = [
            {
                'name': 'naive_fallback',
                'model_class': 'Naive',
                'params': {},
                'models': [Naive()],
                'framework': 'statsforecast'
            },
            {
                'name': 'seasonal_naive_fallback', 
                'model_class': 'SeasonalNaive',
                'params': {'season_length': 7},
                'models': [SeasonalNaive(season_length=7)],
                'framework': 'statsforecast'
            },
            {
                'name': 'autoarima_fallback',
                'model_class': 'AutoARIMA',
                'params': {'season_length': 7},
                'models': [AutoARIMA(season_length=7)],
                'framework': 'statsforecast'
            }
        ]
        
        logger.info("Created fallback statistical models")
        return fallback_configs
        
    except Exception as e:
        logger.error(f"Error creating fallback models: {e}")
        return []

def test_statsforecast_config(data, train_end_date, config):
    """Test StatsForecast configuration (fallback)"""
    try:
        from statsforecast import StatsForecast
        
        # Prepare train/test split
        train_data = data[data['ds'] <= train_end_date].copy()
        test_data = data[data['ds'] > train_end_date].copy()
        
        if len(test_data) == 0:
            logger.warning(f"No test data available for {config['name']}")
            return None
        
        horizon = len(test_data)
        logger.info(f"Training {config['name']}: Train size: {len(train_data)}, Test size: {len(test_data)}, Horizon: {horizon}")
        
        # Create StatsForecast model
        sf = StatsForecast(
            models=config['models'],
            freq='D'
        )
        
        # Fit and predict
        forecasts = sf.forecast(df=train_data, h=horizon)
        
        # Extract predictions
        pred_col = None
        for col in forecasts.columns:
            if col not in ['unique_id', 'ds']:
                pred_col = col
                break
        
        if pred_col is None:
            logger.error(f"Could not find prediction column in forecasts: {forecasts.columns.tolist()}")
            return None
        
        y_pred = forecasts[pred_col].values
        y_true = test_data['y'].values
        
        # Ensure same length
        min_length = min(len(y_true), len(y_pred))
        y_true = y_true[:min_length]
        y_pred = y_pred[:min_length]
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred)
        
        logger.info(f"{config['name']} metrics: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, MAPE={metrics['mape']:.2f}%")
        
        return {
            'model': sf,
            'metrics': metrics,
            'predictions': y_pred,
            'y_true': y_true,
            'forecasts': forecasts,
            'train_data': train_data,
            'test_data': test_data
        }
        
    except Exception as e:
        logger.error(f"Error in StatsForecast config {config['name']}: {e}")
        return None

def parse_arguments():
    """Parse command line arguments for standalone and Flow execution"""
    parser = argparse.ArgumentParser(description='Nixtla NeuralForecast for Oil & Gas Production')
    parser.add_argument('--input-data', type=str, help='Input data file path (for Flow execution)')
    parser.add_argument('--output-dir', type=str, help='Output directory path (for Flow execution)')
    parser.add_argument('--training-summary', type=str, help='Training summary output file (for Flow execution)')
    return parser.parse_args()

def write_training_summary(results, output_path):
    """Write training summary for Flow execution"""
    try:
        summary = {
            'timestamp': datetime.now().isoformat(),
            'framework': 'nixtla_neuralforecast',
            'total_configs': len(results),
            'successful_configs': len([r for r in results.values() if r is not None]),
            'best_config': None,
            'best_mae': float('inf'),
            'models_saved': []
        }

        # Find best configuration
        for config_name, result in results.items():
            if result is not None and result['metrics']['mae'] < summary['best_mae']:
                summary['best_mae'] = result['metrics']['mae']
                summary['best_config'] = config_name

        # Save summary to normal location
        import json
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Training summary saved to: {output_path}")

        # Write to workflow outputs if directory exists (Flow mode)
        workflow_output = Path("/workflow/outputs/training_summary")
        if workflow_output.parent.exists():
            workflow_output.write_text(json.dumps(summary))
            logger.info(f"✓ Wrote workflow output to {workflow_output}")

        return summary

    except Exception as e:
        logger.error(f"Error writing training summary: {e}")
        return None

def main(args=None):
    """Main function to run NeuralForecast experiments"""
    try:
        # Parse arguments
        if args is None:
            args = parse_arguments()

        logger.info("Starting Nixtla NeuralForecast experiment")

        # Data must be available from previous flow task
        # Script will fail fast with descriptive error if data is missing
        logger.info("Loading pre-generated data from previous flow task...")

        # Setup
        experiment_id = setup_mlflow()
        
        # Load data - support both standalone and Flow modes
        if args.input_data and Path(args.input_data).exists():
            # Flow mode - load from specified input
            logger.info(f"Loading data from Flow input: {args.input_data}")
            data = pd.read_parquet(args.input_data)
            # Convert to NeuralForecast format
            if 'date' in data.columns and 'oil_production_bpd' in data.columns:
                daily_production = data.groupby('date').agg({
                    'oil_production_bpd': 'sum'
                }).reset_index()
                daily_production = daily_production.rename(columns={
                    'date': 'ds',
                    'oil_production_bpd': 'y'
                })
                daily_production['unique_id'] = 'oil_production'
                data = daily_production[['unique_id', 'ds', 'y']].sort_values('ds')
            else:
                # Data already in correct format
                pass
        else:
            # Standalone mode - load from default location
            data = load_and_prepare_data()
        
        # Get output paths - use WorkflowIO for Flow compatibility
        wf_io = WorkflowIO()
        if args.output_dir:
            # Flow mode - use specified output directory
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            models_dir = output_dir / 'models'
            models_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Use WorkflowIO for automatic path detection
            models_dir = wf_io.ensure_model_directory('nixtla')
        
        # Use standardized train/test split
        train_data, test_data, train_end_date = ForecastingConfig.get_train_test_split(data, 'ds')
        
        # Validate data quality
        checks, is_valid = ForecastingConfig.validate_data_quality(
            data, target_column='y', date_column='ds'
        )
        
        if not is_valid:
            logger.error(f"Data quality validation failed: {checks}")
            raise ValueError("Data does not meet quality requirements")
        
        # Use standardized test configurations
        test_configs = get_standard_configs('nixtla')
        
        # Store results for Flow output
        all_results = {}
        
        # Start parent run
        with mlflow.start_run(run_name="nixtla_forecasting_suite") as parent_run:
            # Use standardized MLflow tags
            parent_tags = ForecastingConfig.get_mlflow_tags('nixtla', 'suite')
            for key, value in parent_tags.items():
                mlflow.set_tag(key, value)
            
            # Use standardized MLflow params
            parent_params = ForecastingConfig.get_mlflow_params('nixtla', {})
            for key, value in parent_params.items():
                if value is not None:
                    mlflow.log_param(key, value)
            
            mlflow.log_param("total_configs", len(test_configs))
            mlflow.log_param("train_end_date", str(train_end_date))
            mlflow.log_param("train_size", len(train_data))
            mlflow.log_param("test_size", len(test_data))
            
            best_config = None
            best_score = float('inf')
            best_run_id = None
            successful_configs = 0
            
            # Test each configuration as child runs
            for config in test_configs:
                with mlflow.start_run(run_name=f"nixtla_{config['name']}", nested=True) as child_run:
                    try:
                        # Use standardized MLflow tags and params
                        child_tags = ForecastingConfig.get_mlflow_tags('nixtla', config['name'])
                        for key, value in child_tags.items():
                            mlflow.set_tag(key, value)
                        
                        child_params = ForecastingConfig.get_mlflow_params('nixtla', config)
                        for key, value in child_params.items():
                            if value is not None:
                                mlflow.log_param(key, value)
                        
                        # Train and evaluate
                        result = test_neuralforecast_config(
                            data=data,
                            train_end_date=train_end_date,
                            config=config
                        )
                        
                        if result is not None:
                            successful_configs += 1
                            
                            # Store result for Flow output
                            all_results[config['name']] = result
                            
                            # Log metrics
                            metrics = result['metrics']
                            mlflow.log_metric("mae", metrics['mae'])
                            mlflow.log_metric("rmse", metrics['rmse'])
                            mlflow.log_metric("mape", metrics['mape'])
                            
                            # Save model
                            try:
                                model_path = models_dir / f"nixtla_{config['name']}_model.pkl"
                                joblib.dump(result['model'], model_path)
                                mlflow.log_artifact(str(model_path))
                                logger.info(f"Saved model to {model_path}")
                            except Exception as e:
                                logger.warning(f"Could not save model: {e}")
                            
                            # Save forecasts
                            try:
                                forecast_path = models_dir / f"nixtla_{config['name']}_forecasts.csv"
                                result['forecasts'].to_csv(forecast_path, index=False)
                                mlflow.log_artifact(str(forecast_path))
                            except Exception as e:
                                logger.warning(f"Could not save forecasts: {e}")
                            
                            # Save predictions
                            try:
                                pred_df = pd.DataFrame({
                                    'y_true': result['y_true'],
                                    'y_pred': result['predictions']
                                })
                                pred_path = models_dir / f"nixtla_{config['name']}_predictions.csv"
                                pred_df.to_csv(pred_path, index=False)
                                mlflow.log_artifact(str(pred_path))
                            except Exception as e:
                                logger.warning(f"Could not save predictions: {e}")
                            
                            # Track best model
                            if metrics['mae'] < best_score:
                                best_score = metrics['mae']
                                best_config = config['name']
                                best_run_id = child_run.info.run_id
                            
                            mlflow.set_tag("training_status", "success")
                            
                        else:
                            all_results[config['name']] = None
                            mlflow.set_tag("training_status", "failed")
                            logger.warning(f"Failed to train {config['name']}")
                    
                    except Exception as e:
                        all_results[config['name']] = None
                        mlflow.set_tag("training_status", "error")
                        mlflow.log_param("error_message", str(e))
                        logger.error(f"Error in child run {config['name']}: {e}")
            
            # If all neural models failed, try fallback models
            if successful_configs == 0:
                logger.warning("All NeuralForecast models failed, trying fallback models...")
                fallback_configs = create_fallback_models(len(test_data))
                
                for config in fallback_configs:
                    with mlflow.start_run(run_name=f"nixtla_{config['name']}", nested=True) as child_run:
                        try:
                            mlflow.set_tag("model_type", "fallback_statistical")
                            mlflow.log_param("config_name", config['name'])
                            
                            result = test_statsforecast_config(
                                data=data,
                                train_end_date=train_end_date,
                                config=config
                            )
                            
                            if result is not None:
                                successful_configs += 1
                                all_results[config['name']] = result
                                
                                # Log metrics
                                metrics = result['metrics']
                                mlflow.log_metric("mae", metrics['mae'])
                                mlflow.log_metric("rmse", metrics['rmse'])
                                mlflow.log_metric("mape", metrics['mape'])
                                
                                # Track best model
                                if metrics['mae'] < best_score:
                                    best_score = metrics['mae']
                                    best_config = config['name']
                                    best_run_id = child_run.info.run_id
                                
                                mlflow.set_tag("training_status", "success")
                                logger.info(f"Fallback model {config['name']} succeeded")
                        
                        except Exception as e:
                            logger.error(f"Fallback model {config['name']} also failed: {e}")
            
            # Log best model information and tag the best run
            if best_config and best_run_id:
                mlflow.log_param("best_config", best_config)
                mlflow.log_metric("best_mae", best_score)
                mlflow.log_param("successful_configs", successful_configs)
                
                # Tag the best child run
                client = mlflow.tracking.MlflowClient()
                best_tags = ForecastingConfig.get_mlflow_tags('nixtla', best_config, is_best=True)
                for key, value in best_tags.items():
                    client.set_tag(best_run_id, key, value)
                
                logger.info(f"Best Nixtla configuration: {best_config} with MAE: {best_score:.2f}")
                logger.info(f"Tagged best run ID: {best_run_id}")
                logger.info(f"Successful configurations: {successful_configs}/{len(test_configs)}")
        
        # Write training summary for Flow execution
        if args.training_summary:
            summary = write_training_summary(all_results, args.training_summary)
        
        logger.info("Nixtla NeuralForecast experiment completed successfully")
        return all_results
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        # CRITICAL: Write error output for Flow execution
        # This ensures sidecar uploader has a file even if script fails
        workflow_output = Path("/workflow/outputs/training_summary")
        if workflow_output.parent.exists():
            error_data = {
                'timestamp': datetime.now().isoformat(),
                'framework': 'nixtla_neuralforecast',
                'status': 'error',
                'error_message': str(e),
                'error_type': type(e).__name__
            }
            import json
            workflow_output.write_text(json.dumps(error_data))
            logger.info(f"✓ Wrote error output to {workflow_output}")
        raise

if __name__ == "__main__":
    main()