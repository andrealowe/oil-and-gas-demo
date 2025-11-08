#!/usr/bin/env python3
"""
AutoGluon Time Series Forecasting for Oil and Gas Production Data

Tests AutoGluon TimeSeries with different presets and time limits on daily oil production.
Uses MLflow for experiment tracking with child runs for hyperparameter testing.

Compatible with both standalone execution and Domino Flows.
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
from src.models.forecasting_config import ForecastingConfig, get_standard_configs

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
        
        logger.info(f"Loading data from: {data_path}")
        df = pd.read_parquet(data_path)
        
        logger.info(f"Original data shape: {df.shape}")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Aggregate daily oil production across all facilities
        daily_production = df.groupby('date').agg({
            'oil_production_bpd': 'sum'
        }).reset_index()
        
        # Rename columns for AutoGluon format
        daily_production = daily_production.rename(columns={
            'date': 'timestamp',
            'oil_production_bpd': 'target'
        })
        
        # Add item_id column required by AutoGluon
        daily_production['item_id'] = 'oil_production'
        
        # Reorder columns for AutoGluon: [item_id, timestamp, target]
        daily_production = daily_production[['item_id', 'timestamp', 'target']]
        
        # Sort by timestamp and ensure no gaps
        daily_production = daily_production.sort_values('timestamp')
        
        logger.info(f"Aggregated data shape: {daily_production.shape}")
        logger.info(f"Target statistics:\n{daily_production['target'].describe()}")
        
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

def test_autogluon_config(data, train_end_date, config_name, preset, time_limit, hyperparameters=None, models_dir=None):
    """Test specific AutoGluon configuration"""
    try:
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

        # Prepare train/test split
        train_data = data[data['timestamp'] <= train_end_date].copy()
        test_data = data[data['timestamp'] > train_end_date].copy()

        if len(test_data) == 0:
            logger.warning(f"No test data available for {config_name}")
            return None

        logger.info(f"Training {config_name}: Train size: {len(train_data)}, Test size: {len(test_data)}")

        # Convert to AutoGluon TimeSeriesDataFrame
        train_ts = TimeSeriesDataFrame(train_data)

        # Set up model save path in /mnt/artifacts
        if models_dir is None:
            paths = get_data_paths('Oil-and-Gas-Demo')
            models_dir = paths['artifacts_path'] / 'models' / 'autogluon'
        else:
            models_dir = Path(models_dir) / 'autogluon'

        models_dir.mkdir(parents=True, exist_ok=True)
        model_save_path = models_dir / f"ag_{config_name}"

        # Configure predictor
        predictor_kwargs = {
            'target': 'target',
            'prediction_length': len(test_data),
            'freq': 'D',
            'eval_metric': 'MASE',
            'path': str(model_save_path)  # Save to /mnt/artifacts
        }

        if hyperparameters:
            predictor_kwargs['hyperparameters'] = hyperparameters

        # Create and train predictor
        predictor = TimeSeriesPredictor(
            **predictor_kwargs
        )
        
        predictor.fit(
            train_data=train_ts,
            presets=preset,
            time_limit=time_limit,
            verbosity=1
        )
        
        # Make predictions
        predictions = predictor.predict(train_ts)
        
        # Extract predictions as array
        if hasattr(predictions, 'values'):
            y_pred = predictions.values.flatten()
        else:
            y_pred = predictions.flatten()
        
        y_true = test_data['target'].values
        
        # Ensure same length
        min_length = min(len(y_true), len(y_pred))
        y_true = y_true[:min_length]
        y_pred = y_pred[:min_length]
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred)
        
        logger.info(f"{config_name} metrics: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, MAPE={metrics['mape']:.2f}%")
        
        return {
            'predictor': predictor,
            'metrics': metrics,
            'predictions': y_pred,
            'y_true': y_true,
            'leaderboard': predictor.leaderboard() if hasattr(predictor, 'leaderboard') else None
        }
        
    except Exception as e:
        logger.error(f"Error in AutoGluon config {config_name}: {e}")
        return None

def parse_arguments():
    """Parse command line arguments for standalone and Flow execution"""
    parser = argparse.ArgumentParser(description='AutoGluon Forecasting for Oil & Gas Production')
    parser.add_argument('--input-data', type=str, help='Input data file path (for Flow execution)')
    parser.add_argument('--output-dir', type=str, help='Output directory path (for Flow execution)')
    parser.add_argument('--training-summary', type=str, help='Training summary output file (for Flow execution)')
    return parser.parse_args()

def write_training_summary(results, output_path):
    """Write training summary for Flow execution"""
    try:
        summary = {
            'timestamp': datetime.now().isoformat(),
            'framework': 'autogluon',
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
                
        # Save summary
        import json
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training summary saved to: {output_path}")
        return summary
        
    except Exception as e:
        logger.error(f"Error writing training summary: {e}")
        return None

def main(args=None):
    """Main function to run AutoGluon forecasting experiments"""
    try:
        # Parse arguments
        if args is None:
            args = parse_arguments()
        
        logger.info("Starting AutoGluon forecasting experiment")
        
        # Setup
        experiment_id = setup_mlflow()
        
        # Load data - support both standalone and Flow modes
        if args.input_data and Path(args.input_data).exists():
            # Flow mode - load from specified input
            logger.info(f"Loading data from Flow input: {args.input_data}")
            data = pd.read_parquet(args.input_data)
            # Convert to AutoGluon format
            if 'date' in data.columns and 'oil_production_bpd' in data.columns:
                daily_production = data.groupby('date').agg({
                    'oil_production_bpd': 'sum'
                }).reset_index()
                daily_production = daily_production.rename(columns={
                    'date': 'timestamp',
                    'oil_production_bpd': 'target'
                })
                daily_production['item_id'] = 'oil_production'
                data = daily_production[['item_id', 'timestamp', 'target']]
            else:
                # Data already in correct format
                pass
        else:
            # Standalone mode - load from default location
            data = load_and_prepare_data()
        
        # Get output paths - support both standalone and Flow modes
        if args.output_dir:
            # Flow mode
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            models_dir = output_dir / 'models'
        else:
            # Standalone mode
            paths = get_data_paths('Oil-and-Gas-Demo')
            artifacts_dir = paths['artifacts_path']
            models_dir = artifacts_dir / 'models'
        
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Use standardized train/test split
        train_data, test_data, train_end_date = ForecastingConfig.get_train_test_split(data, 'timestamp')
        
        # Validate data quality
        checks, is_valid = ForecastingConfig.validate_data_quality(
            data, target_column='target', date_column='timestamp'
        )
        
        if not is_valid:
            logger.error(f"Data quality validation failed: {checks}")
            raise ValueError("Data does not meet quality requirements")
        
        # Use standardized test configurations
        test_configs = get_standard_configs('autogluon')
        
        # Store results for Flow output
        all_results = {}
        
        # Start parent run
        with mlflow.start_run(run_name="autogluon_forecasting_suite") as parent_run:
            # Use standardized MLflow tags
            parent_tags = ForecastingConfig.get_mlflow_tags('autogluon', 'suite')
            for key, value in parent_tags.items():
                mlflow.set_tag(key, value)
            
            # Use standardized MLflow params
            parent_params = ForecastingConfig.get_mlflow_params('autogluon', {})
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
            
            # Test each configuration as child runs
            for config in test_configs:
                with mlflow.start_run(run_name=f"autogluon_{config['name']}", nested=True) as child_run:
                    try:
                        # Use standardized MLflow tags and params
                        child_tags = ForecastingConfig.get_mlflow_tags('autogluon', config['name'])
                        for key, value in child_tags.items():
                            mlflow.set_tag(key, value)
                        
                        child_params = ForecastingConfig.get_mlflow_params('autogluon', config)
                        for key, value in child_params.items():
                            if value is not None:
                                mlflow.log_param(key, value)
                        
                        # Train and evaluate using updated data splits
                        result = test_autogluon_config(
                            data=data,
                            train_end_date=train_end_date,
                            config_name=config['name'],
                            preset=config['preset'],
                            time_limit=config['time_limit'],
                            hyperparameters=config['hyperparameters'],
                            models_dir=models_dir
                        )
                        
                        if result is not None:
                            # Store result for Flow output
                            all_results[config['name']] = result
                            
                            # Log metrics
                            metrics = result['metrics']
                            mlflow.log_metric("mae", metrics['mae'])
                            mlflow.log_metric("rmse", metrics['rmse'])
                            mlflow.log_metric("mape", metrics['mape'])
                            
                            # Log leaderboard if available
                            if result['leaderboard'] is not None:
                                try:
                                    leaderboard_path = models_dir / f"autogluon_{config['name']}_leaderboard.csv"
                                    result['leaderboard'].to_csv(leaderboard_path, index=False)
                                    mlflow.log_artifact(str(leaderboard_path))
                                except Exception as e:
                                    logger.warning(f"Could not save leaderboard: {e}")
                            
                            # Save model
                            try:
                                model_path = models_dir / f"autogluon_{config['name']}_model.pkl"
                                joblib.dump(result['predictor'], model_path)
                                mlflow.log_artifact(str(model_path))
                                logger.info(f"Saved model to {model_path}")
                            except Exception as e:
                                logger.warning(f"Could not save model: {e}")
                            
                            # Save predictions
                            try:
                                pred_df = pd.DataFrame({
                                    'y_true': result['y_true'],
                                    'y_pred': result['predictions']
                                })
                                pred_path = models_dir / f"autogluon_{config['name']}_predictions.csv"
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
                        mlflow.set_tag("training_status", "error")
                        mlflow.log_param("error_message", str(e))
                        logger.error(f"Error in child run {config['name']}: {e}")
            
            # Log best model information and tag the best run
            if best_config and best_run_id:
                mlflow.log_param("best_config", best_config)
                mlflow.log_metric("best_mae", best_score)
                
                # Tag the best child run
                client = mlflow.tracking.MlflowClient()
                best_tags = ForecastingConfig.get_mlflow_tags('autogluon', best_config, is_best=True)
                for key, value in best_tags.items():
                    client.set_tag(best_run_id, key, value)
                
                logger.info(f"Best AutoGluon configuration: {best_config} with MAE: {best_score:.2f}")
                logger.info(f"Tagged best run ID: {best_run_id}")
        
        # Write training summary for Flow execution
        if args.training_summary:
            summary = write_training_summary(all_results, args.training_summary)
        
        logger.info("AutoGluon forecasting experiment completed successfully")
        return all_results
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        # Write error summary for Flow execution
        if args and args.training_summary:
            error_summary = {
                'timestamp': datetime.now().isoformat(),
                'framework': 'autogluon',
                'status': 'error',
                'error_message': str(e),
                'total_configs': 0,
                'successful_configs': 0
            }
            import json
            with open(args.training_summary, 'w') as f:
                json.dump(error_summary, f, indent=2)
        raise

if __name__ == "__main__":
    main()