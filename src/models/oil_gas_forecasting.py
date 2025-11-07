#!/usr/bin/env python3
"""
Comprehensive Time Series Forecasting Models for Oil & Gas Dashboard

This module develops forecasting models for:
1. Oil/Gas production forecasting
2. Price prediction for multiple commodities
3. Demand forecasting by region and product
4. Maintenance scheduling optimization

Uses modern time series techniques with MLflow integration.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import sys
import os
import json
import joblib
from datetime import datetime, timedelta
import logging

# Add scripts directory to path for data_config import
sys.path.insert(0, '/mnt/code')
from scripts.data_config import get_data_paths, ensure_directories

# Time series libraries
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import lightgbm as lgb

# Statistical forecasting
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available, ARIMA models will be skipped")

# Prophet for time series forecasting
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: Prophet not available, Prophet models will be skipped")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OilGasTimeSeriesForecaster:
    """Comprehensive time series forecasting for oil & gas operations"""
    
    def __init__(self, project_name="Oil-and-Gas-Demo"):
        self.project_name = project_name
        self.paths = get_data_paths(project_name)
        self.directories = ensure_directories(project_name)
        
        # MLflow setup
        mlflow.set_tracking_uri("http://localhost:8768")
        self.experiment_name = "oil_gas_forecasting_models"
        mlflow.set_experiment(self.experiment_name)
        
        # Data paths
        self.data_dir = Path('/mnt/artifacts/Oil-and-Gas-Demo')
        self.models_dir = self.directories['models'] / 'forecasting'
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configurations
        self.forecast_horizons = {
            'short_term': 30,    # 30 days
            'medium_term': 180   # 6 months
        }
        
        self.models = {}
        self.evaluation_results = {}
        
    def load_data(self):
        """Load all time series datasets"""
        logger.info("Loading time series datasets...")
        
        self.data = {}
        
        # Production data
        production_path = self.data_dir / 'production_timeseries.parquet'
        if production_path.exists():
            df_prod = pd.read_parquet(production_path)
            df_prod['date'] = pd.to_datetime(df_prod['date'])
            self.data['production'] = df_prod
            logger.info(f"Loaded production data: {df_prod.shape}")
        
        # Prices data
        prices_path = self.data_dir / 'prices_timeseries.parquet'
        if prices_path.exists():
            df_prices = pd.read_parquet(prices_path)
            df_prices['date'] = pd.to_datetime(df_prices['date'])
            self.data['prices'] = df_prices
            logger.info(f"Loaded prices data: {df_prices.shape}")
        
        # Demand data
        demand_path = self.data_dir / 'demand_timeseries.parquet'
        if demand_path.exists():
            df_demand = pd.read_parquet(demand_path)
            df_demand['date'] = pd.to_datetime(df_demand['date'])
            self.data['demand'] = df_demand
            logger.info(f"Loaded demand data: {df_demand.shape}")
        
        # Maintenance data
        maintenance_path = self.data_dir / 'maintenance_timeseries.parquet'
        if maintenance_path.exists():
            df_maint = pd.read_parquet(maintenance_path)
            df_maint['date'] = pd.to_datetime(df_maint['date'])
            self.data['maintenance'] = df_maint
            logger.info(f"Loaded maintenance data: {df_maint.shape}")
    
    def create_time_features(self, df, date_col='date'):
        """Create time-based features for machine learning models"""
        df = df.copy()
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['day_of_year'] = df[date_col].dt.dayofyear
        df['week_of_year'] = df[date_col].dt.isocalendar().week
        df['quarter'] = df[date_col].dt.quarter
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Cyclical encoding for seasonal patterns
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        return df
    
    def create_lag_features(self, df, target_col, lags=[1, 7, 30]):
        """Create lag features for time series"""
        df = df.copy()
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        # Rolling window features
        for window in [7, 30]:
            df[f'{target_col}_roll_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col}_roll_std_{window}'] = df[target_col].rolling(window=window).std()
        
        return df
    
    def evaluate_forecast(self, y_true, y_pred, model_name, horizon):
        """Evaluate forecast performance"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape
        }
        
        # Store evaluation results
        if model_name not in self.evaluation_results:
            self.evaluation_results[model_name] = {}
        self.evaluation_results[model_name][horizon] = metrics
        
        return metrics
    
    def train_lightgbm_forecast(self, df, target_col, horizon_days=30):
        """Train LightGBM model for time series forecasting"""
        logger.info(f"Training LightGBM model for {target_col}, horizon: {horizon_days} days")
        
        # Prepare data with features
        df_features = self.create_time_features(df)
        df_features = self.create_lag_features(df_features, target_col)
        
        # Remove rows with NaN values (due to lag features)
        df_features = df_features.dropna()
        
        if len(df_features) < 100:
            logger.warning(f"Insufficient data for {target_col} after feature engineering")
            return None
        
        # Feature columns (exclude categorical and non-numeric columns)
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols 
                       if col not in ['date', target_col] and not col.startswith('facility_id')]
        
        # Train/test split (use last 30% for testing)
        split_idx = int(len(df_features) * 0.7)
        train_data = df_features.iloc[:split_idx]
        test_data = df_features.iloc[split_idx:]
        
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]
        
        # Train LightGBM model
        train_dataset = lgb.Dataset(X_train, label=y_train)
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        model = lgb.train(
            params,
            train_dataset,
            num_boost_round=1000,
            valid_sets=[train_dataset],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
        )
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate
        metrics = self.evaluate_forecast(y_test, y_pred, f'lightgbm_{target_col}', f'{horizon_days}_days')
        
        # Generate forecasts for the specified horizon
        last_date = df['date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                   periods=horizon_days, freq='D')
        
        # Create future features (simplified approach)
        future_df = pd.DataFrame({'date': future_dates})
        future_df = self.create_time_features(future_df)
        
        # For lag features, use the last known values (simple approach)
        last_values = df_features[feature_cols].iloc[-1:].values
        future_features = np.tile(last_values, (len(future_dates), 1))
        
        # Update time features for future dates
        time_feature_cols = ['year', 'month', 'day', 'day_of_week', 'day_of_year', 
                           'week_of_year', 'quarter', 'is_weekend', 'month_sin', 
                           'month_cos', 'day_sin', 'day_cos']
        
        for i, col in enumerate(feature_cols):
            if col in time_feature_cols:
                idx = feature_cols.index(col)
                future_features[:, idx] = future_df[col].values
        
        future_predictions = model.predict(future_features)
        
        forecast_df = pd.DataFrame({
            'date': future_dates,
            f'{target_col}_forecast': future_predictions
        })
        
        model_info = {
            'model': model,
            'feature_columns': feature_cols,
            'metrics': metrics,
            'forecast': forecast_df,
            'model_type': 'lightgbm'
        }
        
        return model_info
    
    def train_prophet_forecast(self, df, target_col, horizon_days=30):
        """Train Prophet model for time series forecasting"""
        if not PROPHET_AVAILABLE:
            logger.warning("Prophet not available, skipping Prophet model")
            return None
            
        logger.info(f"Training Prophet model for {target_col}, horizon: {horizon_days} days")
        
        # Prepare data for Prophet (needs 'ds' and 'y' columns)
        prophet_df = df[['date', target_col]].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df = prophet_df.dropna()
        
        if len(prophet_df) < 100:
            logger.warning(f"Insufficient data for {target_col} Prophet model")
            return None
        
        # Train/test split
        split_idx = int(len(prophet_df) * 0.7)
        train_data = prophet_df.iloc[:split_idx]
        test_data = prophet_df.iloc[split_idx:]
        
        # Train Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.95
        )
        
        model.fit(train_data)
        
        # Make predictions on test set
        test_future = test_data[['ds']].copy()
        test_forecast = model.predict(test_future)
        
        # Evaluate
        y_test = test_data['y'].values
        y_pred = test_forecast['yhat'].values
        metrics = self.evaluate_forecast(y_test, y_pred, f'prophet_{target_col}', f'{horizon_days}_days')
        
        # Generate future forecasts
        future = model.make_future_dataframe(periods=horizon_days, freq='D')
        forecast = model.predict(future)
        
        # Extract future predictions
        future_forecast = forecast.tail(horizon_days)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        future_forecast.columns = ['date', f'{target_col}_forecast', 
                                 f'{target_col}_lower', f'{target_col}_upper']
        
        model_info = {
            'model': model,
            'metrics': metrics,
            'forecast': future_forecast,
            'model_type': 'prophet',
            'full_forecast': forecast
        }
        
        return model_info
    
    def train_production_forecasting_models(self):
        """Train forecasting models for oil and gas production"""
        logger.info("Training production forecasting models...")
        
        if 'production' not in self.data:
            logger.error("Production data not available")
            return
        
        df_prod = self.data['production'].copy()
        
        with mlflow.start_run(run_name="production_forecasting_suite") as parent_run:
            mlflow.set_tag("model_category", "production_forecasting")
            mlflow.set_tag("data_type", "time_series")
            
            # Aggregate production by date for overall forecasting
            daily_prod = df_prod.groupby('date').agg({
                'oil_production_bpd': 'sum',
                'gas_production_mcfd': 'sum'
            }).reset_index()
            
            production_targets = ['oil_production_bpd', 'gas_production_mcfd']
            
            for target in production_targets:
                for horizon_name, horizon_days in self.forecast_horizons.items():
                    
                    with mlflow.start_run(run_name=f"{target}_{horizon_name}", nested=True):
                        mlflow.log_param("target_variable", target)
                        mlflow.log_param("forecast_horizon", horizon_days)
                        mlflow.log_param("horizon_type", horizon_name)
                        
                        models_for_target = {}
                        
                        # Train LightGBM model
                        lgb_model = self.train_lightgbm_forecast(daily_prod, target, horizon_days)
                        if lgb_model:
                            models_for_target['lightgbm'] = lgb_model
                            
                            # Log metrics
                            for metric, value in lgb_model['metrics'].items():
                                mlflow.log_metric(f"lgb_{metric}", value)
                            
                            # Save model
                            model_path = self.models_dir / f"production_{target}_lgb_{horizon_name}.pkl"
                            joblib.dump(lgb_model, model_path)
                            mlflow.log_artifact(str(model_path))
                        
                        # Train Prophet model
                        prophet_model = self.train_prophet_forecast(daily_prod, target, horizon_days)
                        if prophet_model:
                            models_for_target['prophet'] = prophet_model
                            
                            # Log metrics
                            for metric, value in prophet_model['metrics'].items():
                                mlflow.log_metric(f"prophet_{metric}", value)
                            
                            # Save model
                            model_path = self.models_dir / f"production_{target}_prophet_{horizon_name}.pkl"
                            joblib.dump(prophet_model, model_path)
                            mlflow.log_artifact(str(model_path))
                        
                        # Store models
                        model_key = f"production_{target}_{horizon_name}"
                        self.models[model_key] = models_for_target
                        
                        # Save forecast data
                        if models_for_target:
                            best_model = min(models_for_target.values(), 
                                           key=lambda x: x['metrics']['mape'])
                            forecast_path = self.models_dir / f"forecast_{target}_{horizon_name}.csv"
                            best_model['forecast'].to_csv(forecast_path, index=False)
                            mlflow.log_artifact(str(forecast_path))
                            
                            mlflow.log_metric("best_mape", best_model['metrics']['mape'])
                            mlflow.set_tag("best_model_type", best_model['model_type'])
    
    def train_price_forecasting_models(self):
        """Train forecasting models for commodity prices"""
        logger.info("Training price forecasting models...")
        
        if 'prices' not in self.data:
            logger.error("Prices data not available")
            return
        
        df_prices = self.data['prices'].copy()
        
        with mlflow.start_run(run_name="price_forecasting_suite") as parent_run:
            mlflow.set_tag("model_category", "price_forecasting")
            mlflow.set_tag("data_type", "time_series")
            
            price_targets = ['crude_oil_price_usd_bbl', 'natural_gas_price_usd_mcf', 
                           'brent_crude_usd_bbl', 'wti_crude_usd_bbl']
            
            for target in price_targets:
                for horizon_name, horizon_days in self.forecast_horizons.items():
                    
                    with mlflow.start_run(run_name=f"{target}_{horizon_name}", nested=True):
                        mlflow.log_param("target_variable", target)
                        mlflow.log_param("forecast_horizon", horizon_days)
                        mlflow.log_param("horizon_type", horizon_name)
                        
                        models_for_target = {}
                        
                        # Train LightGBM model
                        lgb_model = self.train_lightgbm_forecast(df_prices, target, horizon_days)
                        if lgb_model:
                            models_for_target['lightgbm'] = lgb_model
                            
                            for metric, value in lgb_model['metrics'].items():
                                mlflow.log_metric(f"lgb_{metric}", value)
                            
                            model_path = self.models_dir / f"prices_{target}_lgb_{horizon_name}.pkl"
                            joblib.dump(lgb_model, model_path)
                            mlflow.log_artifact(str(model_path))
                        
                        # Train Prophet model
                        prophet_model = self.train_prophet_forecast(df_prices, target, horizon_days)
                        if prophet_model:
                            models_for_target['prophet'] = prophet_model
                            
                            for metric, value in prophet_model['metrics'].items():
                                mlflow.log_metric(f"prophet_{metric}", value)
                            
                            model_path = self.models_dir / f"prices_{target}_prophet_{horizon_name}.pkl"
                            joblib.dump(prophet_model, model_path)
                            mlflow.log_artifact(str(model_path))
                        
                        # Store models
                        model_key = f"prices_{target}_{horizon_name}"
                        self.models[model_key] = models_for_target
                        
                        # Save forecast data
                        if models_for_target:
                            best_model = min(models_for_target.values(), 
                                           key=lambda x: x['metrics']['mape'])
                            forecast_path = self.models_dir / f"forecast_{target}_{horizon_name}.csv"
                            best_model['forecast'].to_csv(forecast_path, index=False)
                            mlflow.log_artifact(str(forecast_path))
                            
                            mlflow.log_metric("best_mape", best_model['metrics']['mape'])
                            mlflow.set_tag("best_model_type", best_model['model_type'])
    
    def train_demand_forecasting_models(self):
        """Train forecasting models for demand by region"""
        logger.info("Training demand forecasting models...")
        
        if 'demand' not in self.data:
            logger.error("Demand data not available")
            return
        
        df_demand = self.data['demand'].copy()
        
        with mlflow.start_run(run_name="demand_forecasting_suite") as parent_run:
            mlflow.set_tag("model_category", "demand_forecasting")
            mlflow.set_tag("data_type", "time_series")
            
            demand_targets = ['oil_demand_thousand_bpd', 'gas_demand_thousand_mcfd',
                            'gasoline_demand_thousand_bpd', 'diesel_demand_thousand_bpd']
            
            regions = df_demand['region'].unique()
            
            for region in regions:
                region_data = df_demand[df_demand['region'] == region].copy()
                
                with mlflow.start_run(run_name=f"demand_{region}", nested=True):
                    mlflow.log_param("region", region)
                    
                    for target in demand_targets:
                        for horizon_name, horizon_days in self.forecast_horizons.items():
                            
                            with mlflow.start_run(run_name=f"{target}_{horizon_name}", nested=True):
                                mlflow.log_param("target_variable", target)
                                mlflow.log_param("forecast_horizon", horizon_days)
                                mlflow.log_param("horizon_type", horizon_name)
                                mlflow.log_param("region", region)
                                
                                models_for_target = {}
                                
                                # Train LightGBM model
                                lgb_model = self.train_lightgbm_forecast(region_data, target, horizon_days)
                                if lgb_model:
                                    models_for_target['lightgbm'] = lgb_model
                                    
                                    for metric, value in lgb_model['metrics'].items():
                                        mlflow.log_metric(f"lgb_{metric}", value)
                                    
                                    model_path = self.models_dir / f"demand_{region}_{target}_lgb_{horizon_name}.pkl"
                                    joblib.dump(lgb_model, model_path)
                                    mlflow.log_artifact(str(model_path))
                                
                                # Train Prophet model  
                                prophet_model = self.train_prophet_forecast(region_data, target, horizon_days)
                                if prophet_model:
                                    models_for_target['prophet'] = prophet_model
                                    
                                    for metric, value in prophet_model['metrics'].items():
                                        mlflow.log_metric(f"prophet_{metric}", value)
                                    
                                    model_path = self.models_dir / f"demand_{region}_{target}_prophet_{horizon_name}.pkl"
                                    joblib.dump(prophet_model, model_path)
                                    mlflow.log_artifact(str(model_path))
                                
                                # Store models
                                model_key = f"demand_{region}_{target}_{horizon_name}"
                                self.models[model_key] = models_for_target
                                
                                # Save forecast data
                                if models_for_target:
                                    best_model = min(models_for_target.values(), 
                                                   key=lambda x: x['metrics']['mape'])
                                    forecast_path = self.models_dir / f"forecast_{region}_{target}_{horizon_name}.csv"
                                    best_model['forecast'].to_csv(forecast_path, index=False)
                                    mlflow.log_artifact(str(forecast_path))
                                    
                                    mlflow.log_metric("best_mape", best_model['metrics']['mape'])
                                    mlflow.set_tag("best_model_type", best_model['model_type'])
    
    def train_maintenance_forecasting_models(self):
        """Train maintenance scheduling optimization models"""
        logger.info("Training maintenance forecasting models...")
        
        if 'maintenance' not in self.data:
            logger.error("Maintenance data not available")
            return
        
        df_maint = self.data['maintenance'].copy()
        
        with mlflow.start_run(run_name="maintenance_forecasting_suite") as parent_run:
            mlflow.set_tag("model_category", "maintenance_forecasting")
            mlflow.set_tag("data_type", "time_series")
            
            # Aggregate maintenance by date
            daily_maint = df_maint.groupby('date').agg({
                'duration_hours': 'sum',
                'cost_usd': 'sum'
            }).reset_index()
            
            # Also aggregate by maintenance type
            type_maint = df_maint.groupby(['date', 'maintenance_type']).agg({
                'duration_hours': 'sum',
                'cost_usd': 'sum'
            }).reset_index()
            
            maintenance_targets = ['duration_hours', 'cost_usd']
            
            # Overall maintenance forecasting
            for target in maintenance_targets:
                for horizon_name, horizon_days in self.forecast_horizons.items():
                    
                    with mlflow.start_run(run_name=f"overall_{target}_{horizon_name}", nested=True):
                        mlflow.log_param("target_variable", target)
                        mlflow.log_param("forecast_horizon", horizon_days)
                        mlflow.log_param("horizon_type", horizon_name)
                        mlflow.log_param("scope", "overall")
                        
                        models_for_target = {}
                        
                        # Train LightGBM model
                        lgb_model = self.train_lightgbm_forecast(daily_maint, target, horizon_days)
                        if lgb_model:
                            models_for_target['lightgbm'] = lgb_model
                            
                            for metric, value in lgb_model['metrics'].items():
                                mlflow.log_metric(f"lgb_{metric}", value)
                            
                            model_path = self.models_dir / f"maintenance_overall_{target}_lgb_{horizon_name}.pkl"
                            joblib.dump(lgb_model, model_path)
                            mlflow.log_artifact(str(model_path))
                        
                        # Train Prophet model
                        prophet_model = self.train_prophet_forecast(daily_maint, target, horizon_days)
                        if prophet_model:
                            models_for_target['prophet'] = prophet_model
                            
                            for metric, value in prophet_model['metrics'].items():
                                mlflow.log_metric(f"prophet_{metric}", value)
                            
                            model_path = self.models_dir / f"maintenance_overall_{target}_prophet_{horizon_name}.pkl"
                            joblib.dump(prophet_model, model_path)
                            mlflow.log_artifact(str(model_path))
                        
                        # Store models
                        model_key = f"maintenance_overall_{target}_{horizon_name}"
                        self.models[model_key] = models_for_target
                        
                        if models_for_target:
                            best_model = min(models_for_target.values(), 
                                           key=lambda x: x['metrics']['mape'])
                            forecast_path = self.models_dir / f"forecast_maintenance_{target}_{horizon_name}.csv"
                            best_model['forecast'].to_csv(forecast_path, index=False)
                            mlflow.log_artifact(str(forecast_path))
                            
                            mlflow.log_metric("best_mape", best_model['metrics']['mape'])
                            mlflow.set_tag("best_model_type", best_model['model_type'])
    
    def generate_model_summary(self):
        """Generate comprehensive model performance summary"""
        logger.info("Generating model performance summary...")
        
        summary = {
            'total_models_trained': len(self.models),
            'model_categories': {
                'production': len([k for k in self.models.keys() if 'production' in k]),
                'prices': len([k for k in self.models.keys() if 'prices' in k]),
                'demand': len([k for k in self.models.keys() if 'demand' in k]),
                'maintenance': len([k for k in self.models.keys() if 'maintenance' in k])
            },
            'evaluation_results': self.evaluation_results,
            'forecast_horizons': self.forecast_horizons,
            'model_directory': str(self.models_dir)
        }
        
        # Save summary
        summary_path = self.models_dir / 'model_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary
    
    def register_best_models(self):
        """Register best performing models to MLflow Model Registry"""
        logger.info("Registering best models to MLflow Model Registry...")
        
        client = mlflow.tracking.MlflowClient()
        
        # Find best models for each category
        categories = ['production', 'prices', 'demand', 'maintenance']
        
        for category in categories:
            category_models = {k: v for k, v in self.evaluation_results.items() 
                             if category in k}
            
            if not category_models:
                continue
            
            # Find best model by MAPE
            best_model_key = None
            best_mape = float('inf')
            
            for model_key, horizons in category_models.items():
                for horizon, metrics in horizons.items():
                    if metrics['mape'] < best_mape:
                        best_mape = metrics['mape']
                        best_model_key = model_key
            
            if best_model_key:
                # Register model
                model_name = f"oil_gas_{category}_forecasting"
                
                try:
                    client.create_registered_model(
                        model_name,
                        tags={'category': category, 'use_case': 'oil_gas_forecasting'}
                    )
                except:
                    pass  # Model already exists
                
                logger.info(f"Registered best {category} model: {best_model_key} (MAPE: {best_mape:.2f}%)")
    
    def create_api_serving_models(self):
        """Create API-ready model serving infrastructure"""
        logger.info("Creating API serving infrastructure...")
        
        # Create model serving configuration
        serving_config = {
            'models': {},
            'endpoints': {},
            'deployment_ready': True
        }
        
        # Process each model category
        for model_key, model_variants in self.models.items():
            if model_variants:
                # Select best model variant
                best_variant = min(model_variants.values(), 
                                 key=lambda x: x['metrics']['mape'])
                
                serving_config['models'][model_key] = {
                    'model_type': best_variant['model_type'],
                    'metrics': best_variant['metrics'],
                    'model_path': str(self.models_dir / f"{model_key}_{best_variant['model_type']}.pkl")
                }
        
        # Save serving configuration
        config_path = self.models_dir / 'serving_config.json'
        with open(config_path, 'w') as f:
            json.dump(serving_config, f, indent=2)
        
        return serving_config
    
    def run_complete_forecasting_pipeline(self):
        """Run the complete forecasting model development pipeline"""
        logger.info("Starting complete oil & gas forecasting pipeline...")
        
        try:
            # Load data
            self.load_data()
            
            # Train models for each category
            self.train_production_forecasting_models()
            self.train_price_forecasting_models()
            self.train_demand_forecasting_models()
            self.train_maintenance_forecasting_models()
            
            # Generate summary and register models
            summary = self.generate_model_summary()
            self.register_best_models()
            serving_config = self.create_api_serving_models()
            
            logger.info("Forecasting pipeline completed successfully!")
            logger.info(f"Total models trained: {summary['total_models_trained']}")
            logger.info(f"Models saved to: {self.models_dir}")
            
            return {
                'status': 'success',
                'summary': summary,
                'serving_config': serving_config,
                'models_directory': str(self.models_dir)
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

def main():
    """Main execution function"""
    forecaster = OilGasTimeSeriesForecaster()
    result = forecaster.run_complete_forecasting_pipeline()
    print("Pipeline Result:")
    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    main()