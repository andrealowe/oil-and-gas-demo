#!/usr/bin/env python3
"""
Nixtla NeuralForecast for Oil and Gas Production Data

Tests Nixtla's neuralforecast with different neural network models on daily oil production.
Uses MLflow for experiment tracking with child runs for hyperparameter testing.
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
warnings.filterwarnings('ignore')

# Add scripts directory to path for data_config import
sys.path.insert(0, '/mnt/code')
from scripts.data_config import get_data_paths

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

def test_neuralforecast_config(data, train_end_date, config_name, models, horizon):
    """Test specific NeuralForecast configuration"""
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.utils import AirPassengersDF
        
        # Prepare train/test split
        train_data = data[data['ds'] <= train_end_date].copy()
        test_data = data[data['ds'] > train_end_date].copy()
        
        if len(test_data) == 0:
            logger.warning(f"No test data available for {config_name}")
            return None
        
        # Adjust horizon to match test data size
        actual_horizon = min(horizon, len(test_data))
        test_data = test_data.head(actual_horizon)
        
        logger.info(f"Training {config_name}: Train size: {len(train_data)}, Test size: {len(test_data)}, Horizon: {actual_horizon}")
        
        # Create NeuralForecast model
        nf = NeuralForecast(
            models=models,
            freq='D'
        )
        
        # Fit model
        nf.fit(train_data)
        
        # Make predictions
        forecasts = nf.predict(h=actual_horizon)
        
        # Extract predictions (use the first model's predictions)
        model_name = list(models[0].__dict__.keys())[0] if hasattr(models[0], '__dict__') else str(models[0]).split('(')[0]
        
        # Find the correct prediction column
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
        
        logger.info(f"{config_name} metrics: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, MAPE={metrics['mape']:.2f}%")
        
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
        logger.error(f"Error in NeuralForecast config {config_name}: {e}")
        return None

def main():
    """Main function to run NeuralForecast experiments"""
    try:
        logger.info("Starting Nixtla NeuralForecast experiment")
        
        # Setup
        experiment_id = setup_mlflow()
        data = load_and_prepare_data()
        
        # Get data paths for saving artifacts
        paths = get_data_paths('Oil-and-Gas-Demo')
        artifacts_dir = paths['artifacts_path']
        models_dir = artifacts_dir / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Define train/test split (use 80% for training)
        train_size = int(len(data) * 0.8)
        train_end_date = data.iloc[train_size]['ds']
        horizon = len(data) - train_size
        
        # Define test configurations
        test_configs = [
            {
                'name': 'mlp_simple',
                'models': [],  # Will be populated below
                'horizon': horizon,
                'model_params': {'input_size': 14, 'h': horizon, 'max_steps': 100}
            },
            {
                'name': 'nbeats_simple',
                'models': [],
                'horizon': horizon,
                'model_params': {'input_size': 14, 'h': horizon, 'max_steps': 100}
            },
            {
                'name': 'nhits_simple',
                'models': [],
                'horizon': horizon,
                'model_params': {'input_size': 14, 'h': horizon, 'max_steps': 100}
            },
            {
                'name': 'lstm_simple',
                'models': [],
                'horizon': horizon,
                'model_params': {'input_size': 14, 'h': horizon, 'max_steps': 100}
            },
            {
                'name': 'tft_simple',
                'models': [],
                'horizon': horizon,
                'model_params': {'input_size': 14, 'h': horizon, 'max_steps': 50}  # Reduced for TFT
            }
        ]
        
        # Import and configure models
        try:
            from neuralforecast.models import MLP, NBEATS, NHITS, LSTM, TFT
            
            # Update configurations with actual model instances
            for config in test_configs:
                if config['name'] == 'mlp_simple':
                    config['models'] = [MLP(**config['model_params'])]
                elif config['name'] == 'nbeats_simple':
                    config['models'] = [NBEATS(**config['model_params'])]
                elif config['name'] == 'nhits_simple':
                    config['models'] = [NHITS(**config['model_params'])]
                elif config['name'] == 'lstm_simple':
                    config['models'] = [LSTM(**config['model_params'])]
                elif config['name'] == 'tft_simple':
                    config['models'] = [TFT(**config['model_params'])]
                    
        except ImportError as e:
            logger.error(f"Could not import NeuralForecast models: {e}")
            logger.info("Trying alternative approach with StatsForecast models")
            
            # Fallback to StatsForecast models if NeuralForecast is not available
            try:
                from statsforecast.models import AutoARIMA, ETS, Theta
                from statsforecast import StatsForecast
                
                # Reconfigure with StatsForecast models
                test_configs = [
                    {
                        'name': 'autoarima',
                        'models': [AutoARIMA(season_length=7)],
                        'horizon': horizon,
                        'framework': 'statsforecast'
                    },
                    {
                        'name': 'ets',
                        'models': [ETS(season_length=7)],
                        'horizon': horizon,
                        'framework': 'statsforecast'
                    },
                    {
                        'name': 'theta',
                        'models': [Theta(season_length=7)],
                        'horizon': horizon,
                        'framework': 'statsforecast'
                    }
                ]
                
            except ImportError as e2:
                logger.error(f"Could not import StatsForecast either: {e2}")
                raise ImportError("Neither NeuralForecast nor StatsForecast is available")
        
        # Start parent run
        with mlflow.start_run(run_name="nixtla_forecasting_suite") as parent_run:
            mlflow.set_tag("model_type", "nixtla_neuralforecast")
            mlflow.set_tag("data_source", "oil_gas_production")
            mlflow.log_param("total_configs", len(test_configs))
            mlflow.log_param("train_end_date", str(train_end_date))
            mlflow.log_param("train_size", train_size)
            mlflow.log_param("test_size", horizon)
            
            best_config = None
            best_score = float('inf')
            
            # Test each configuration as child runs
            for config in test_configs:
                with mlflow.start_run(run_name=f"nixtla_{config['name']}", nested=True) as child_run:
                    try:
                        # Log configuration parameters
                        mlflow.log_param("config_name", config['name'])
                        mlflow.log_param("horizon", config['horizon'])
                        mlflow.log_param("framework", config.get('framework', 'neuralforecast'))
                        
                        if 'model_params' in config:
                            for key, value in config['model_params'].items():
                                mlflow.log_param(key, value)
                        
                        # Handle different frameworks
                        if config.get('framework') == 'statsforecast':
                            result = test_statsforecast_config(
                                data=data,
                                train_end_date=train_end_date,
                                config_name=config['name'],
                                models=config['models'],
                                horizon=config['horizon']
                            )
                        else:
                            result = test_neuralforecast_config(
                                data=data,
                                train_end_date=train_end_date,
                                config_name=config['name'],
                                models=config['models'],
                                horizon=config['horizon']
                            )
                        
                        if result is not None:
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
                                mlflow.set_tag("best_model_candidate", "true")
                            
                            mlflow.set_tag("training_status", "success")
                            
                        else:
                            mlflow.set_tag("training_status", "failed")
                            logger.warning(f"Failed to train {config['name']}")
                    
                    except Exception as e:
                        mlflow.set_tag("training_status", "error")
                        mlflow.log_param("error_message", str(e))
                        logger.error(f"Error in child run {config['name']}: {e}")
            
            # Log best model information
            if best_config:
                mlflow.log_param("best_config", best_config)
                mlflow.log_metric("best_mae", best_score)
                logger.info(f"Best Nixtla configuration: {best_config} with MAE: {best_score:.2f}")
            
        logger.info("Nixtla NeuralForecast experiment completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

def test_statsforecast_config(data, train_end_date, config_name, models, horizon):
    """Test specific StatsForecast configuration (fallback)"""
    try:
        from statsforecast import StatsForecast
        
        # Prepare train/test split
        train_data = data[data['ds'] <= train_end_date].copy()
        test_data = data[data['ds'] > train_end_date].copy()
        
        if len(test_data) == 0:
            logger.warning(f"No test data available for {config_name}")
            return None
        
        # Adjust horizon to match test data size
        actual_horizon = min(horizon, len(test_data))
        test_data = test_data.head(actual_horizon)
        
        logger.info(f"Training {config_name}: Train size: {len(train_data)}, Test size: {len(test_data)}, Horizon: {actual_horizon}")
        
        # Create StatsForecast model
        sf = StatsForecast(
            models=models,
            freq='D'
        )
        
        # Fit and predict
        forecasts = sf.forecast(df=train_data, h=actual_horizon)
        
        # Extract predictions (use the first model's predictions)
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
        
        logger.info(f"{config_name} metrics: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, MAPE={metrics['mape']:.2f}%")
        
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
        logger.error(f"Error in StatsForecast config {config_name}: {e}")
        return None

if __name__ == "__main__":
    main()