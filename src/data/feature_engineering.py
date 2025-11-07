"""
Feature Engineering for Oil & Gas Geospatial Data
Provides reusable functions for creating features for ML models
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from datetime import datetime

def calculate_production_efficiency(df):
    """Calculate production efficiency metrics"""
    df = df.copy()
    
    # Basic efficiency for wells with capacity data
    df['production_efficiency'] = np.where(
        (df['facility_type'] == 'oil_well') & (df['capacity_bpd'].notna()) & (df['capacity_bpd'] > 0),
        (df['oil_production_bpd'] / df['capacity_bpd']).clip(0, 1),
        np.nan
    )
    
    # Throughput efficiency
    df['throughput_efficiency'] = np.where(
        (df['throughput'].notna()) & (df['capacity'].notna()) & (df['capacity'] > 0),
        (df['throughput'] / df['capacity']).clip(0, 1),
        np.nan
    )
    
    return df

def create_location_clusters(df, n_clusters='auto', random_state=42):
    """Create geographic clusters for facilities"""
    coords_df = df.dropna(subset=['latitude', 'longitude']).copy()
    
    if len(coords_df) < 10:
        coords_df['location_cluster'] = 0
        return coords_df
    
    coords = coords_df[['latitude', 'longitude']].values
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    
    if n_clusters == 'auto':
        # Find optimal clusters using silhouette analysis
        from sklearn.metrics import silhouette_score
        best_score = -1
        best_k = 3
        
        for k in range(3, min(12, len(coords_df)//10)):
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(coords_scaled)
            score = silhouette_score(coords_scaled, labels)
            if score > best_score:
                best_score = score
                best_k = k
        n_clusters = best_k
    
    # Apply clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    coords_df['location_cluster'] = kmeans.fit_predict(coords_scaled)
    
    return coords_df

def calculate_proximity_features(df, max_distance_deg=5.0):
    """Calculate proximity-based features"""
    df = df.copy()
    
    # Initialize proximity features
    df['nearest_facility_distance'] = np.nan
    df['facilities_within_5deg'] = 0
    df['cluster_density'] = 0
    
    coords = df[['latitude', 'longitude']].dropna()
    
    if len(coords) > 1:
        # Calculate distance matrix (sample for large datasets)
        sample_size = min(1000, len(coords))
        sample_indices = np.random.choice(coords.index, size=sample_size, replace=False)
        sample_coords = coords.loc[sample_indices]
        
        for idx in sample_indices:
            facility_coords = coords.loc[idx].values.reshape(1, -1)
            other_coords = coords.drop(idx).values
            
            if len(other_coords) > 0:
                # Calculate distances to all other facilities
                distances = np.sqrt(np.sum((other_coords - facility_coords) ** 2, axis=1))
                
                df.loc[idx, 'nearest_facility_distance'] = np.min(distances)
                df.loc[idx, 'facilities_within_5deg'] = np.sum(distances <= max_distance_deg)
                df.loc[idx, 'cluster_density'] = 1.0 / (np.mean(distances[:min(5, len(distances))]) + 1e-6)
    
    return df

def create_environmental_risk_score(df):
    """Create environmental risk assessment score"""
    df = df.copy()
    
    env_cols = ['h2s_concentration_ppm', 'co2_concentration_ppm', 'noise_level_db']
    df['environmental_risk_score'] = 0
    
    # Normalize each environmental metric
    for col in env_cols:
        if col in df.columns and df[col].notna().any():
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max > col_min:
                normalized = (df[col] - col_min) / (col_max - col_min)
                df['environmental_risk_score'] += normalized.fillna(0)
    
    # Average risk score
    num_metrics = sum(1 for col in env_cols if col in df.columns and df[col].notna().any())
    if num_metrics > 0:
        df['environmental_risk_score'] = df['environmental_risk_score'] / num_metrics
    
    return df

def create_maintenance_features(df):
    """Create maintenance-related features"""
    df = df.copy()
    
    # Convert dates
    df['last_maintenance_date'] = pd.to_datetime(df['last_maintenance_date'], errors='coerce')
    df['next_maintenance_date'] = pd.to_datetime(df['next_maintenance_date'], errors='coerce')
    current_date = datetime.now()
    
    # Days since/until maintenance
    df['days_since_maintenance'] = (current_date - df['last_maintenance_date']).dt.days
    df['days_until_maintenance'] = (df['next_maintenance_date'] - current_date).dt.days
    
    # Maintenance status flags
    df['maintenance_overdue'] = df['days_since_maintenance'].fillna(0) > 365
    df['maintenance_due_soon'] = (df['days_until_maintenance'] >= 0) & (df['days_until_maintenance'] <= 30)
    
    # Maintenance interval
    df['maintenance_interval'] = (df['next_maintenance_date'] - df['last_maintenance_date']).dt.days
    
    return df

def create_operational_features(df):
    """Create operational performance features"""
    df = df.copy()
    
    # Age-based features
    if 'well_age_years' in df.columns:
        df['age_category'] = pd.cut(df['well_age_years'].fillna(0), 
                                   bins=[0, 5, 10, 20, float('inf')], 
                                   labels=['new', 'young', 'mature', 'old'])
    
    # Health score categories
    df['health_category'] = pd.cut(df['equipment_health_score'], 
                                  bins=[0, 0.6, 0.8, 1.0], 
                                  labels=['poor', 'good', 'excellent'])
    
    # Production categories (for oil wells)
    if 'oil_production_bpd' in df.columns:
        oil_wells = df['facility_type'] == 'oil_well'
        production_median = df.loc[oil_wells, 'oil_production_bpd'].median()
        df['production_category'] = 'none'
        df.loc[oil_wells, 'production_category'] = np.where(
            df.loc[oil_wells, 'oil_production_bpd'] >= production_median, 
            'high', 'low'
        )
    
    # Utilization rate
    if 'current_throughput_bpd' in df.columns and 'capacity_bpd' in df.columns:
        df['utilization_rate_calculated'] = np.where(
            df['capacity_bpd'] > 0,
            (df['current_throughput_bpd'] / df['capacity_bpd']).clip(0, 1),
            np.nan
        )
    
    return df

def engineer_all_features(df, include_clustering=True):
    """Apply all feature engineering functions"""
    print("Starting feature engineering...")
    
    # Apply all feature engineering functions
    df = calculate_production_efficiency(df)
    print("✓ Production efficiency features created")
    
    df = create_environmental_risk_score(df)
    print("✓ Environmental risk features created")
    
    df = create_maintenance_features(df)
    print("✓ Maintenance features created")
    
    df = create_operational_features(df)
    print("✓ Operational features created")
    
    if include_clustering:
        df = create_location_clusters(df)
        print("✓ Location clustering completed")
        
        df = calculate_proximity_features(df)
        print("✓ Proximity features created")
    
    print(f"Feature engineering complete. Final dataset shape: {df.shape}")
    return df

def get_feature_list():
    """Return list of all engineered features"""
    return [
        'production_efficiency',
        'throughput_efficiency', 
        'environmental_risk_score',
        'days_since_maintenance',
        'days_until_maintenance',
        'maintenance_overdue',
        'maintenance_due_soon',
        'maintenance_interval',
        'age_category',
        'health_category',
        'production_category',
        'utilization_rate_calculated',
        'location_cluster',
        'nearest_facility_distance',
        'facilities_within_5deg',
        'cluster_density'
    ]

if __name__ == "__main__":
    # Example usage
    print("Oil & Gas Feature Engineering Module")
    print("Available functions:")
    print("- engineer_all_features(df): Apply all feature engineering")
    print("- calculate_production_efficiency(df): Production metrics")
    print("- create_location_clusters(df): Geographic clustering")
    print("- create_environmental_risk_score(df): Environmental assessment")
    print("- create_maintenance_features(df): Maintenance analysis")
    print("- create_operational_features(df): Operational categorization")