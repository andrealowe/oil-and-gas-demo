#!/usr/bin/env python3
"""
Standardized Configuration for Oil & Gas AutoML Forecasting Experiments

This module provides consistent parameters across all AutoML frameworks
to ensure fair comparison and reproducible results.
"""

from datetime import datetime, timedelta
from pathlib import Path

class ForecastingConfig:
    """Standardized configuration for all AutoML forecasting experiments"""
    
    # Data configuration
    DATA_CONFIG = {
        'train_split_ratio': 0.8,  # 80% for training, 20% for testing
        'target_column': 'oil_production_bpd',
        'date_column': 'date',
        'aggregation_method': 'sum',  # How to aggregate daily production
        'min_data_points': 100,  # Minimum data points required for training
    }
    
    # Experiment configuration
    EXPERIMENT_CONFIG = {
        'experiment_name': 'oil_gas_forecasting_models',
        'max_training_time_minutes': 10,  # Maximum training time per configuration
        'random_seed': 42,  # For reproducibility
        'cross_validation_folds': 3,  # For model validation
        'early_stopping_rounds': 50,  # For gradient boosting models
    }
    
    # Evaluation metrics (consistent across all frameworks)
    EVALUATION_METRICS = ['mae', 'rmse', 'mape']
    PRIMARY_METRIC = 'mae'  # Primary metric for model selection
    
    # AutoGluon standardized configurations
    AUTOGLUON_CONFIGS = [
        {
            'name': 'fast_training',
            'preset': 'fast_training',
            'time_limit': 120,  # 2 minutes
            'hyperparameters': None
        },
        {
            'name': 'medium_quality',
            'preset': 'medium_quality', 
            'time_limit': 300,  # 5 minutes
            'hyperparameters': None
        },
        {
            'name': 'high_quality',
            'preset': 'high_quality',
            'time_limit': 600,  # 10 minutes
            'hyperparameters': None
        },
        {
            'name': 'statistical_ensemble',
            'preset': 'medium_quality',
            'time_limit': 300,
            'hyperparameters': {
                'ETS': {'seasonal': 'add'},
                'ARIMA': {'seasonal': True},
                'Theta': {},
                'SeasonalNaive': {}
            }
        },
        {
            'name': 'neural_ensemble', 
            'preset': 'high_quality',
            'time_limit': 600,
            'hyperparameters': {
                'DeepAR': {'epochs': 50, 'learning_rate': 0.001},
                'SimpleFeedForward': {'hidden_size': 64},
                'TemporalFusionTransformer': {'hidden_size': 32}
            }
        }
    ]
    
    # Prophet standardized configurations
    PROPHET_CONFIGS = [
        {
            'name': 'prophet_default',
            'model_type': 'prophet',
            'params': {
                'daily_seasonality': True,
                'weekly_seasonality': True, 
                'yearly_seasonality': True,
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0,
                'interval_width': 0.95
            }
        },
        {
            'name': 'prophet_strong_seasonality',
            'model_type': 'prophet',
            'params': {
                'daily_seasonality': True,
                'weekly_seasonality': True,
                'yearly_seasonality': True, 
                'changepoint_prior_scale': 0.1,
                'seasonality_prior_scale': 20.0,
                'holidays_prior_scale': 15.0,
                'interval_width': 0.95
            }
        },
        {
            'name': 'prophet_flexible_trend',
            'model_type': 'prophet',
            'params': {
                'daily_seasonality': True,
                'weekly_seasonality': True,
                'yearly_seasonality': True,
                'changepoint_prior_scale': 0.5,
                'n_changepoints': 50,
                'seasonality_prior_scale': 10.0,
                'interval_width': 0.95
            }
        },
        {
            'name': 'neuralprophet_default',
            'model_type': 'neuralprophet',
            'params': {
                'growth': 'linear',
                'n_forecasts': 1,
                'epochs': 100,
                'learning_rate': 0.1,
                'batch_size': 32
            }
        },
        {
            'name': 'neuralprophet_autoregressive',
            'model_type': 'neuralprophet', 
            'params': {
                'growth': 'linear',
                'n_forecasts': 1,
                'n_lags': 14,  # 2 weeks of autoregressive terms
                'epochs': 150,
                'learning_rate': 0.05,
                'batch_size': 64
            }
        }
    ]
    
    # Nixtla NeuralForecast standardized configurations
    NIXTLA_CONFIGS = [
        {
            'name': 'mlp_medium',
            'model_class': 'MLP',
            'params': {
                'input_size': 14,  # 2 weeks lookback
                'max_steps': 200,
                'learning_rate': 0.001,
                'batch_size': 32,
                'hidden_size': 64
            }
        },
        {
            'name': 'nbeats_medium',
            'model_class': 'NBEATS', 
            'params': {
                'input_size': 14,
                'max_steps': 200,
                'learning_rate': 0.001,
                'batch_size': 32,
                'stack_types': ['trend', 'seasonality']
            }
        },
        {
            'name': 'nhits_medium',
            'model_class': 'NHITS',
            'params': {
                'input_size': 14,
                'max_steps': 200,
                'learning_rate': 0.001,
                'batch_size': 32,
                'n_pool_kernel_size': [2, 2, 2]
            }
        },
        {
            'name': 'lstm_medium',
            'model_class': 'LSTM',
            'params': {
                'input_size': 14,
                'max_steps': 200,
                'learning_rate': 0.001,
                'batch_size': 32,
                'hidden_size': 64,
                'num_layers': 2
            }
        },
        {
            'name': 'tft_medium',
            'model_class': 'TFT',
            'params': {
                'input_size': 14,
                'max_steps': 100,  # Reduced for TFT complexity
                'learning_rate': 0.001,
                'batch_size': 16,
                'hidden_size': 32
            }
        }
    ]
    
    # Combined LightGBM + ARIMA standardized configuration
    COMBINED_CONFIG = {
        'name': 'lightgbm_arima_ensemble',
        'lightgbm_params': {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'num_boost_round': 1000,
            'early_stopping_rounds': 50
        },
        'arima_params': {
            'seasonal': True,
            'seasonal_periods': 7,  # Weekly seasonality
            'max_p': 3,
            'max_q': 3,
            'max_d': 2
        },
        'feature_engineering': {
            'lag_features': [1, 7, 14, 30],  # 1 day, 1 week, 2 weeks, 1 month
            'rolling_windows': [7, 14, 30],  # Rolling statistics windows
            'seasonal_decomposition': True,
            'time_features': True  # Year, month, day, etc.
        },
        'ensemble_weights': {
            'lightgbm': 0.7,
            'arima': 0.3
        }
    }
    
    @classmethod
    def get_train_test_split(cls, data, date_column='ds'):
        """Get standardized train/test split"""
        train_size = int(len(data) * cls.DATA_CONFIG['train_split_ratio'])
        train_end_date = data.iloc[train_size][date_column]
        
        train_data = data[data[date_column] <= train_end_date].copy()
        test_data = data[data[date_column] > train_end_date].copy()
        
        return train_data, test_data, train_end_date
    
    @classmethod
    def get_horizon_days(cls, test_data):
        """Get forecast horizon in days"""
        return len(test_data)
    
    @classmethod
    def validate_data_quality(cls, data, target_column=None, date_column=None):
        """Validate data meets minimum quality requirements"""
        target_col = target_column or cls.DATA_CONFIG['target_column']
        date_col = date_column or cls.DATA_CONFIG['date_column'] 
        
        checks = {
            'sufficient_data': len(data) >= cls.DATA_CONFIG['min_data_points'],
            'has_target': target_col in data.columns,
            'has_date': date_col in data.columns,
            'no_null_target': data[target_col].notna().all() if target_col in data.columns else False,
            'no_null_date': data[date_col].notna().all() if date_col in data.columns else False
        }
        
        return checks, all(checks.values())
    
    @classmethod
    def get_mlflow_tags(cls, framework, config_name, is_best=False):
        """Get standardized MLflow tags"""
        tags = {
            'framework': framework,
            'config_name': config_name,
            'experiment_version': '1.0',
            'data_source': 'oil_gas_production',
            'train_split_ratio': str(cls.DATA_CONFIG['train_split_ratio']),
            'primary_metric': cls.PRIMARY_METRIC
        }
        
        if is_best:
            tags['best_model_in_framework'] = 'true'
            tags['champion_candidate'] = 'true'
        
        return tags
    
    @classmethod
    def get_mlflow_params(cls, framework, config):
        """Get standardized MLflow parameters"""
        base_params = {
            'framework': framework,
            'train_split_ratio': cls.DATA_CONFIG['train_split_ratio'],
            'random_seed': cls.EXPERIMENT_CONFIG['random_seed'],
            'primary_metric': cls.PRIMARY_METRIC
        }
        
        # Add framework-specific parameters
        if framework == 'autogluon':
            base_params.update({
                'preset': config.get('preset'),
                'time_limit': config.get('time_limit'),
                'hyperparameters': str(config.get('hyperparameters'))
            })
        elif framework in ['prophet', 'neuralprophet']:
            base_params.update({
                'model_type': config.get('model_type'),
                **config.get('params', {})
            })
        elif framework == 'nixtla':
            base_params.update({
                'model_class': config.get('model_class'),
                **config.get('params', {})
            })
        elif framework == 'combined':
            base_params.update({
                'ensemble_type': 'lightgbm_arima',
                'lightgbm_learning_rate': config.get('lightgbm_params', {}).get('learning_rate'),
                'ensemble_weights': str(config.get('ensemble_weights'))
            })
        
        return base_params

# Convenience functions for easy import
def get_standard_configs(framework):
    """Get standardized configurations for a specific framework"""
    config_map = {
        'autogluon': ForecastingConfig.AUTOGLUON_CONFIGS,
        'prophet': ForecastingConfig.PROPHET_CONFIGS,
        'nixtla': ForecastingConfig.NIXTLA_CONFIGS,
        'combined': [ForecastingConfig.COMBINED_CONFIG]
    }
    return config_map.get(framework, [])

def get_experiment_config():
    """Get experiment configuration"""
    return ForecastingConfig.EXPERIMENT_CONFIG

def get_data_config():
    """Get data configuration"""
    return ForecastingConfig.DATA_CONFIG

if __name__ == "__main__":
    # Print configuration summary
    print("Oil & Gas AutoML Forecasting - Standardized Configuration")
    print("=" * 60)
    print(f"Training Split: {ForecastingConfig.DATA_CONFIG['train_split_ratio']}")
    print(f"Primary Metric: {ForecastingConfig.PRIMARY_METRIC}")
    print(f"Random Seed: {ForecastingConfig.EXPERIMENT_CONFIG['random_seed']}")
    print()
    
    frameworks = ['autogluon', 'prophet', 'nixtla', 'combined']
    for framework in frameworks:
        configs = get_standard_configs(framework)
        print(f"{framework.upper()}: {len(configs)} configurations")
        for config in configs:
            print(f"  - {config['name']}")
    print()
    print("All frameworks use identical:")
    print("- Train/test split ratios")
    print("- Evaluation metrics") 
    print("- Random seeds")
    print("- Data preprocessing")
    print("- MLflow logging standards")