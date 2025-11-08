#!/usr/bin/env python3
"""
Comprehensive Exploratory Data Analysis for Oil & Gas Geospatial Dataset
Focus on geospatial analysis, feature engineering, and interactive visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from pathlib import Path
import json
import warnings
from datetime import datetime
import sys
import mlflow
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
import folium
from folium.plugins import HeatMap, MarkerCluster
import geopandas as gpd

warnings.filterwarnings('ignore')

# Set up paths using data config
sys.path.insert(0, '/mnt/code')
from scripts.data_config import get_data_paths

# Set up MLflow
mlflow.set_tracking_uri("http://localhost:8768")

def setup_environment(project_name='Oil-and-Gas-Demo'):
    """Setup directory structure and data paths"""
    paths = get_data_paths(project_name)

    # Create necessary directories
    notebooks_dir = Path("/mnt/code/notebooks")
    scripts_dir = Path("/mnt/code/scripts")

    # Visualization and analysis directories
    viz_dir = paths['artifacts_path'] / "visualizations" / "geospatial"
    analysis_dir = paths['artifacts_path'] / "analysis" / "geospatial"

    for directory in [notebooks_dir, scripts_dir, viz_dir, analysis_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    return paths, viz_dir, analysis_dir

def load_and_prepare_data(project_name='Oil-and-Gas-Demo'):
    """Load and prepare the oil & gas geospatial dataset"""
    print("Loading oil & gas geospatial dataset...")

    # Get correct data path
    paths = get_data_paths(project_name)

    # Load the dataset
    df = pd.read_parquet(paths['base_data_path'] / 'geospatial_facilities.parquet')
    
    print(f"Dataset loaded: {df.shape[0]} facilities, {df.shape[1]} features")
    
    # Data preparation
    df['last_maintenance_date'] = pd.to_datetime(df['last_maintenance_date'], errors='coerce')
    df['next_maintenance_date'] = pd.to_datetime(df['next_maintenance_date'], errors='coerce')
    
    # Calculate days since last maintenance
    df['days_since_maintenance'] = (datetime.now() - df['last_maintenance_date']).dt.days
    
    # Calculate production efficiency
    df['production_efficiency'] = df['oil_production_bpd'] / (df['capacity_bpd'] + 1e-6)
    df['production_efficiency'] = df['production_efficiency'].fillna(0).clip(0, 1)
    
    # Environmental risk score
    df['environmental_risk'] = (
        (df['h2s_concentration_ppm'].fillna(0) / 100) +
        (df['co2_concentration_ppm'].fillna(0) / 1000) +
        (df['noise_level_db'].fillna(0) / 100)
    ) / 3
    
    return df

def analyze_facility_distribution(df, viz_dir):
    """Analyze facility distribution by region and country"""
    print("\n=== FACILITY DISTRIBUTION ANALYSIS ===")
    
    # Regional distribution
    region_stats = df.groupby('region').agg({
        'facility_id': 'count',
        'oil_production_bpd': ['mean', 'sum'],
        'gas_production_mcfd': ['mean', 'sum'],
        'equipment_health_score': 'mean'
    }).round(2)
    
    region_stats.columns = ['facility_count', 'avg_oil_prod', 'total_oil_prod', 
                           'avg_gas_prod', 'total_gas_prod', 'avg_health_score']
    
    print("Regional Statistics:")
    print(region_stats)
    
    # Country distribution  
    country_stats = df.groupby('country').agg({
        'facility_id': 'count',
        'oil_production_bpd': 'sum',
        'equipment_health_score': 'mean'
    }).round(2).sort_values('oil_production_bpd', ascending=False)
    
    print(f"\nTop 10 Countries by Oil Production:")
    print(country_stats.head(10))
    
    # Facility type distribution
    facility_type_stats = df.groupby('facility_type').agg({
        'facility_id': 'count',
        'oil_production_bpd': ['mean', 'sum'],
        'equipment_health_score': 'mean'
    }).round(2)
    
    print(f"\nFacility Type Statistics:")
    print(facility_type_stats)
    
    # Create visualizations
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Facilities by Region', 'Production by Country (Top 15)', 
                       'Facility Type Distribution', 'Regional Health Scores'],
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "pie"}, {"type": "box"}]]
    )
    
    # Regional facility count
    fig.add_trace(
        go.Bar(x=region_stats.index, y=region_stats['facility_count'],
               name='Facilities', marker_color='lightblue'),
        row=1, col=1
    )
    
    # Top countries by production
    top_countries = country_stats.head(15)
    fig.add_trace(
        go.Bar(x=top_countries.index, y=top_countries['oil_production_bpd'],
               name='Oil Production', marker_color='orange'),
        row=1, col=2
    )
    
    # Facility type pie chart
    facility_counts = df['facility_type'].value_counts()
    fig.add_trace(
        go.Pie(labels=facility_counts.index, values=facility_counts.values,
               name="Facility Types"),
        row=2, col=1
    )
    
    # Regional health scores box plot
    for region in df['region'].unique():
        region_health = df[df['region'] == region]['equipment_health_score']
        fig.add_trace(
            go.Box(y=region_health, name=region, showlegend=False),
            row=2, col=2
        )
    
    fig.update_layout(height=800, title_text="Oil & Gas Facility Distribution Analysis")
    fig.write_html(str(viz_dir / "facility_distribution_analysis.html"))
    
    return region_stats, country_stats, facility_type_stats

def create_geospatial_maps(df, viz_dir):
    """Create interactive geospatial maps"""
    print("\n=== CREATING GEOSPATIAL MAPS ===")
    
    # Clean data for visualization
    df_viz = df.copy()
    df_viz['oil_production_bpd_clean'] = df_viz['oil_production_bpd'].fillna(0)
    df_viz['size_marker'] = df_viz['oil_production_bpd_clean'] + 1  # Add 1 to avoid zero sizes
    
    # 1. World map with facilities colored by production
    fig_world = px.scatter_mapbox(
        df_viz,
        lat="latitude",
        lon="longitude",
        size="size_marker",
        color="equipment_health_score",
        hover_name="facility_name",
        hover_data=["country", "facility_type", "oil_production_bpd"],
        color_continuous_scale="RdYlGn",
        size_max=15,
        zoom=1,
        title="Global Oil & Gas Facilities - Production & Health"
    )
    fig_world.update_layout(mapbox_style="open-street-map", height=600)
    fig_world.write_html(str(viz_dir / "world_facilities_map.html"))
    
    # 2. Production efficiency heatmap
    df_viz['production_efficiency_clean'] = df_viz['production_efficiency'].fillna(0)
    fig_efficiency = px.density_mapbox(
        df_viz,
        lat="latitude",
        lon="longitude",
        z="production_efficiency_clean",
        radius=10,
        center=dict(lat=df['latitude'].mean(), lon=df['longitude'].mean()),
        zoom=1,
        mapbox_style="open-street-map",
        title="Production Efficiency Density Map"
    )
    fig_efficiency.write_html(str(viz_dir / "production_efficiency_heatmap.html"))
    
    # 3. Environmental risk map
    df_viz['env_risk_size'] = (df_viz['environmental_risk'].fillna(0) * 100) + 1  # Scale and avoid zeros
    fig_env_risk = px.scatter_mapbox(
        df_viz,
        lat="latitude",
        lon="longitude",
        size="env_risk_size",
        color="environmental_risk",
        hover_name="facility_name",
        hover_data=["h2s_concentration_ppm", "co2_concentration_ppm", "noise_level_db"],
        color_continuous_scale="Reds",
        size_max=20,
        zoom=1,
        title="Environmental Risk Assessment Map"
    )
    fig_env_risk.update_layout(mapbox_style="open-street-map", height=600)
    fig_env_risk.write_html(str(viz_dir / "environmental_risk_map.html"))
    
    # 4. Facility status map
    fig_status = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        color="status",
        hover_name="facility_name",
        hover_data=["country", "facility_type"],
        zoom=1,
        title="Facility Operational Status Map"
    )
    fig_status.update_layout(mapbox_style="open-street-map", height=600)
    fig_status.write_html(str(viz_dir / "facility_status_map.html"))
    
    print("✓ Interactive maps created and saved")

def perform_location_clustering(df, viz_dir):
    """Perform location clustering for operational efficiency"""
    print("\n=== LOCATION CLUSTERING ANALYSIS ===")
    
    # Prepare coordinates for clustering
    coords = df[['latitude', 'longitude']].dropna()
    
    # Standardize coordinates
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    
    # Determine optimal number of clusters using silhouette analysis
    silhouette_scores = []
    K_range = range(3, 15)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(coords_scaled)
        silhouette_avg = silhouette_score(coords_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_k}")
    
    # Apply optimal clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    df_coords = df.dropna(subset=['latitude', 'longitude']).copy()
    df_coords['cluster'] = kmeans.fit_predict(
        scaler.transform(df_coords[['latitude', 'longitude']])
    )
    
    # Calculate cluster statistics
    cluster_stats = df_coords.groupby('cluster').agg({
        'facility_id': 'count',
        'oil_production_bpd': ['mean', 'sum'],
        'equipment_health_score': 'mean',
        'latitude': 'mean',
        'longitude': 'mean'
    }).round(3)
    
    cluster_stats.columns = ['facility_count', 'avg_oil_prod', 'total_oil_prod',
                            'avg_health_score', 'cluster_lat', 'cluster_lon']
    
    print("Cluster Statistics:")
    print(cluster_stats)
    
    # Visualize clusters
    fig = px.scatter_mapbox(
        df_coords,
        lat="latitude",
        lon="longitude",
        color="cluster",
        size="oil_production_bpd",
        hover_name="facility_name",
        hover_data=["country", "equipment_health_score"],
        zoom=1,
        title="Facility Location Clusters"
    )
    fig.update_layout(mapbox_style="open-street-map", height=600)
    fig.write_html(str(viz_dir / "location_clusters_map.html"))
    
    return cluster_stats, df_coords

def calculate_proximity_metrics(df, viz_dir):
    """Calculate facility proximity metrics"""
    print("\n=== PROXIMITY METRICS ANALYSIS ===")
    
    # Calculate distance matrix for a sample of facilities (to manage computation)
    sample_size = min(500, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    # Extract coordinates
    coords = df_sample[['latitude', 'longitude']].values
    
    # Calculate pairwise distances (in degrees, can be converted to km)
    distances = pdist(coords, metric='euclidean')
    distance_matrix = squareform(distances)
    
    # Calculate proximity statistics for each facility
    proximity_stats = []
    for i, facility in df_sample.iterrows():
        facility_distances = distance_matrix[df_sample.index.get_loc(i)]
        facility_distances = facility_distances[facility_distances > 0]  # Exclude self
        
        if len(facility_distances) > 0:
            proximity_stats.append({
                'facility_id': facility['facility_id'],
                'nearest_distance': np.min(facility_distances),
                'avg_distance_to_others': np.mean(facility_distances),
                'facilities_within_1deg': np.sum(facility_distances < 1.0),
                'facilities_within_5deg': np.sum(facility_distances < 5.0)
            })
    
    proximity_df = pd.DataFrame(proximity_stats)
    
    print("Proximity Statistics Summary:")
    print(proximity_df.describe())
    
    # Create proximity visualization
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Distance to Nearest Facility', 'Facility Density (within 5°)']
    )
    
    fig.add_trace(
        go.Histogram(x=proximity_df['nearest_distance'], nbinsx=20,
                    name='Nearest Distance', marker_color='lightblue'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Histogram(x=proximity_df['facilities_within_5deg'], nbinsx=15,
                    name='Nearby Facilities', marker_color='lightgreen'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, title_text="Facility Proximity Analysis")
    fig.write_html(str(viz_dir / "proximity_analysis.html"))
    
    return proximity_df

def analyze_production_patterns(df, viz_dir):
    """Analyze production efficiency and patterns"""
    print("\n=== PRODUCTION PATTERNS ANALYSIS ===")
    
    # Production efficiency by region
    efficiency_by_region = df.groupby('region')['production_efficiency'].agg(['mean', 'std', 'count']).round(3)
    print("Production Efficiency by Region:")
    print(efficiency_by_region)
    
    # Production vs equipment health correlation
    production_health_corr = pearsonr(
        df['oil_production_bpd'].fillna(0),
        df['equipment_health_score']
    )
    print(f"\nProduction vs Equipment Health Correlation: {production_health_corr[0]:.3f} (p={production_health_corr[1]:.3f})")
    
    # Create production analysis visualizations
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Production Efficiency by Region', 'Production vs Health Score',
                       'Production Distribution by Facility Type', 'Age vs Production Efficiency'],
        specs=[[{"type": "box"}, {"type": "scatter"}],
               [{"type": "violin"}, {"type": "scatter"}]]
    )
    
    # Production efficiency by region
    for region in df['region'].unique():
        region_data = df[df['region'] == region]['production_efficiency']
        fig.add_trace(
            go.Box(y=region_data, name=region, showlegend=False),
            row=1, col=1
        )
    
    # Production vs health scatter
    fig.add_trace(
        go.Scatter(
            x=df['equipment_health_score'],
            y=df['oil_production_bpd'],
            mode='markers',
            name='Facilities',
            opacity=0.6
        ),
        row=1, col=2
    )
    
    # Production by facility type
    for ftype in df['facility_type'].unique():
        type_data = df[df['facility_type'] == ftype]['oil_production_bpd'].fillna(0)
        fig.add_trace(
            go.Violin(y=type_data, name=ftype, showlegend=False),
            row=2, col=1
        )
    
    # Age vs efficiency
    fig.add_trace(
        go.Scatter(
            x=df['well_age_years'],
            y=df['production_efficiency'],
            mode='markers',
            name='Age vs Efficiency',
            opacity=0.6
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title_text="Production Patterns Analysis")
    fig.write_html(str(viz_dir / "production_patterns_analysis.html"))
    
    return efficiency_by_region

def create_environmental_monitoring_dashboard(df, viz_dir):
    """Create environmental monitoring visualizations"""
    print("\n=== ENVIRONMENTAL MONITORING ANALYSIS ===")
    
    # Environmental metrics correlation
    env_cols = ['h2s_concentration_ppm', 'co2_concentration_ppm', 'noise_level_db', 
                'temperature_celsius', 'pressure_psi', 'vibration_level_mm_s']
    env_data = df[env_cols].dropna()
    
    if len(env_data) > 0:
        env_corr = env_data.corr()
        
        # Environmental correlation heatmap
        fig_corr = px.imshow(
            env_corr,
            title="Environmental Parameters Correlation Matrix",
            color_continuous_scale="RdBu_r"
        )
        fig_corr.write_html(str(viz_dir / "environmental_correlation_heatmap.html"))
        
        # Environmental risk distribution
        fig_env = make_subplots(
            rows=2, cols=3,
            subplot_titles=['H2S Concentration', 'CO2 Concentration', 'Noise Level',
                           'Temperature', 'Pressure', 'Vibration Level']
        )
        
        positions = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]
        for i, col in enumerate(env_cols):
            row, col_pos = positions[i]
            fig_env.add_trace(
                go.Histogram(x=df[col].dropna(), nbinsx=20, 
                           name=col, showlegend=False),
                row=row, col=col_pos
            )
        
        fig_env.update_layout(height=600, title_text="Environmental Parameters Distribution")
        fig_env.write_html(str(viz_dir / "environmental_parameters_dashboard.html"))
        
        # Environmental risk by region
        env_risk_by_region = df.groupby('region')['environmental_risk'].agg(['mean', 'std']).round(3)
        print("Environmental Risk by Region:")
        print(env_risk_by_region)
    
    else:
        print("Insufficient environmental data for detailed analysis")

def create_equipment_health_analysis(df, viz_dir):
    """Analyze equipment health patterns"""
    print("\n=== EQUIPMENT HEALTH ANALYSIS ===")
    
    # Health score distribution
    health_stats = df['equipment_health_score'].describe()
    print("Equipment Health Score Statistics:")
    print(health_stats)
    
    # Health by facility type
    health_by_type = df.groupby('facility_type')['equipment_health_score'].agg(['mean', 'std', 'count']).round(3)
    print("\nHealth Score by Facility Type:")
    print(health_by_type)
    
    # Health vs age correlation
    age_health_corr = pearsonr(
        df['well_age_years'].fillna(0),
        df['equipment_health_score']
    )
    print(f"\nAge vs Health Correlation: {age_health_corr[0]:.3f} (p={age_health_corr[1]:.3f})")
    
    # Create equipment health visualizations
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Health Score Distribution', 'Health by Facility Type',
                       'Health vs Age', 'Health Score Geographic Distribution']
    )
    
    # Health score histogram
    fig.add_trace(
        go.Histogram(x=df['equipment_health_score'], nbinsx=30,
                    name='Health Score', marker_color='lightgreen'),
        row=1, col=1
    )
    
    # Health by facility type
    for ftype in df['facility_type'].unique():
        type_health = df[df['facility_type'] == ftype]['equipment_health_score']
        fig.add_trace(
            go.Box(y=type_health, name=ftype, showlegend=False),
            row=1, col=2
        )
    
    # Health vs age scatter
    fig.add_trace(
        go.Scatter(
            x=df['well_age_years'],
            y=df['equipment_health_score'],
            mode='markers',
            name='Age vs Health',
            opacity=0.6
        ),
        row=2, col=1
    )
    
    # Geographic health distribution
    fig.add_trace(
        go.Scatter(
            x=df['longitude'],
            y=df['latitude'],
            mode='markers',
            marker=dict(
                size=8,
                color=df['equipment_health_score'],
                colorscale='RdYlGn',
                showscale=True
            ),
            name='Geographic Health'
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title_text="Equipment Health Analysis")
    fig.write_html(str(viz_dir / "equipment_health_analysis.html"))
    
    return health_by_type

def generate_comprehensive_insights(df, analysis_results, analysis_dir):
    """Generate comprehensive insights and recommendations"""
    print("\n=== GENERATING INSIGHTS & RECOMMENDATIONS ===")
    
    insights = {
        'dataset_summary': {
            'total_facilities': len(df),
            'regions': df['region'].nunique(),
            'countries': df['country'].nunique(),
            'facility_types': df['facility_type'].nunique(),
            'data_quality': {
                'missing_production_data': df['oil_production_bpd'].isna().sum(),
                'missing_coordinates': df[['latitude', 'longitude']].isna().any(axis=1).sum(),
                'complete_records': len(df) - df[['latitude', 'longitude', 'equipment_health_score']].isna().any(axis=1).sum()
            }
        },
        
        'geospatial_insights': {
            'facility_distribution': 'Facilities are distributed across multiple regions with varying densities',
            'production_hotspots': 'Certain geographic clusters show higher production efficiency',
            'environmental_concerns': 'Some regions show elevated environmental risk indicators'
        },
        
        'operational_insights': {
            'health_score_average': df['equipment_health_score'].mean(),
            'production_efficiency_average': df['production_efficiency'].mean(),
            'top_performing_region': df.groupby('region')['production_efficiency'].mean().idxmax(),
            'maintenance_backlog': df['days_since_maintenance'].fillna(0).mean()
        },
        
        'recommendations': [
            'Implement predictive maintenance based on equipment health scores and age',
            'Focus operational improvements on low-efficiency production clusters',
            'Enhance environmental monitoring in high-risk areas',
            'Consider consolidation opportunities in over-dense facility clusters',
            'Develop region-specific operational strategies based on performance patterns'
        ]
    }
    
    # Save insights to JSON
    with open(analysis_dir / 'comprehensive_insights.json', 'w') as f:
        json.dump(insights, f, indent=2, default=str)
    
    print("Key Insights Generated:")
    for category, data in insights.items():
        if category != 'recommendations':
            print(f"\n{category.replace('_', ' ').title()}:")
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, dict):
                        print(f"  {key}:")
                        for subkey, subvalue in value.items():
                            print(f"    {subkey}: {subvalue}")
                    else:
                        print(f"  {key}: {value}")
    
    print("\nRecommendations:")
    for i, rec in enumerate(insights['recommendations'], 1):
        print(f"{i}. {rec}")
    
    return insights

def main():
    """Main execution function"""
    print("=== OIL & GAS GEOSPATIAL DATA ANALYSIS ===")
    
    # Initialize MLflow experiment
    mlflow.set_experiment("oil_gas_geospatial_eda")
    
    with mlflow.start_run(run_name="comprehensive_geospatial_eda") as run:
        # Setup environment
        paths, viz_dir, analysis_dir = setup_environment()
        mlflow.log_param("project_name", "oil_gas_dashboards")
        mlflow.log_param("analysis_type", "comprehensive_geospatial_eda")
        
        # Load and prepare data
        df = load_and_prepare_data()
        mlflow.log_param("dataset_shape", f"{df.shape[0]}x{df.shape[1]}")
        mlflow.log_metric("total_facilities", len(df))
        mlflow.log_metric("regions_count", df['region'].nunique())
        mlflow.log_metric("countries_count", df['country'].nunique())
        
        # Save prepared dataset
        prepared_data_path = paths['base_data_path'] / "prepared_geospatial_data.parquet"
        df.to_parquet(prepared_data_path)
        mlflow.log_artifact(str(prepared_data_path))
        
        analysis_results = {}
        
        try:
            # 1. Facility Distribution Analysis
            region_stats, country_stats, facility_type_stats = analyze_facility_distribution(df, viz_dir)
            analysis_results['regional_analysis'] = region_stats
            mlflow.log_metric("avg_health_score", df['equipment_health_score'].mean())
            
            # 2. Geospatial Maps
            create_geospatial_maps(df, viz_dir)
            
            # 3. Location Clustering
            cluster_stats, df_clustered = perform_location_clustering(df, viz_dir)
            analysis_results['clustering'] = cluster_stats
            mlflow.log_metric("optimal_clusters", len(cluster_stats))
            
            # 4. Proximity Analysis
            proximity_stats = calculate_proximity_metrics(df, viz_dir)
            analysis_results['proximity'] = proximity_stats
            
            # 5. Production Patterns
            production_analysis = analyze_production_patterns(df, viz_dir)
            analysis_results['production'] = production_analysis
            mlflow.log_metric("avg_production_efficiency", df['production_efficiency'].mean())
            
            # 6. Environmental Monitoring
            create_environmental_monitoring_dashboard(df, viz_dir)
            mlflow.log_metric("avg_environmental_risk", df['environmental_risk'].mean())
            
            # 7. Equipment Health Analysis
            health_analysis = create_equipment_health_analysis(df, viz_dir)
            analysis_results['equipment_health'] = health_analysis
            
            # 8. Generate Comprehensive Insights
            insights = generate_comprehensive_insights(df, analysis_results, analysis_dir)
            
            # Log key insights to MLflow
            for key, value in insights['operational_insights'].items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
            
            # Save all analysis results
            results_path = analysis_dir / "complete_analysis_results.json"
            with open(results_path, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            mlflow.log_artifact(str(results_path))
            
            # Log visualization artifacts
            for viz_file in viz_dir.glob("*.html"):
                mlflow.log_artifact(str(viz_file))
            
            mlflow.set_tag("analysis_status", "success")
            print(f"\n✓ Analysis complete! Results saved to {analysis_dir}")
            print(f"✓ Visualizations saved to {viz_dir}")
            
        except Exception as e:
            mlflow.log_param("error", str(e))
            mlflow.set_tag("analysis_status", "failed")
            print(f"❌ Analysis failed: {e}")
            raise

if __name__ == "__main__":
    main()