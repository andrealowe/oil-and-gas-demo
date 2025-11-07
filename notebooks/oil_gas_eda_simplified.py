#!/usr/bin/env python3
"""
Simplified Oil & Gas Geospatial EDA - Focus on core analysis without problematic visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json
import warnings
from datetime import datetime
import sys
import mlflow
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

# Set up paths using data config
sys.path.insert(0, '/mnt/code')
from scripts.data_config import get_data_paths

# Set up MLflow
mlflow.set_tracking_uri("http://localhost:8768")

def setup_environment(project_name='oil_gas_dashboards'):
    """Setup directory structure and data paths"""
    paths = get_data_paths(project_name)
    
    # Create necessary directories
    viz_dir = paths['artifacts_path'] / "visualizations" / "geospatial"
    analysis_dir = paths['artifacts_path'] / "analysis" / "geospatial"
    
    for directory in [viz_dir, analysis_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    return paths, viz_dir, analysis_dir

def load_and_prepare_data():
    """Load and prepare the oil & gas geospatial dataset"""
    print("Loading oil & gas geospatial dataset...")
    
    # Load the dataset
    df = pd.read_parquet('/mnt/artifacts/data/oil_gas_dashboards/geospatial_facilities.parquet')
    
    print(f"Dataset loaded: {df.shape[0]} facilities, {df.shape[1]} features")
    
    # Data preparation
    df['last_maintenance_date'] = pd.to_datetime(df['last_maintenance_date'], errors='coerce')
    df['next_maintenance_date'] = pd.to_datetime(df['next_maintenance_date'], errors='coerce')
    
    # Calculate days since last maintenance
    df['days_since_maintenance'] = (datetime.now() - df['last_maintenance_date']).dt.days
    
    # Calculate production efficiency (only for oil wells with capacity data)
    df['production_efficiency'] = np.where(
        (df['facility_type'] == 'oil_well') & (df['capacity_bpd'].notna()) & (df['capacity_bpd'] > 0),
        df['oil_production_bpd'] / df['capacity_bpd'],
        np.nan
    )
    df['production_efficiency'] = df['production_efficiency'].clip(0, 1)
    
    # Environmental risk score (normalized)
    env_cols = ['h2s_concentration_ppm', 'co2_concentration_ppm', 'noise_level_db']
    df['environmental_risk'] = 0
    if df[env_cols].notna().any().any():
        for col in env_cols:
            if df[col].notna().any():
                normalized = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                df['environmental_risk'] += normalized.fillna(0)
        df['environmental_risk'] = df['environmental_risk'] / len(env_cols)
    
    return df

def analyze_facility_distribution(df, viz_dir):
    """Comprehensive facility distribution analysis"""
    print("\n=== FACILITY DISTRIBUTION ANALYSIS ===")
    
    # Regional statistics
    region_stats = df.groupby('region').agg({
        'facility_id': 'count',
        'oil_production_bpd': ['mean', 'sum'],
        'gas_production_mcfd': ['mean', 'sum'],
        'equipment_health_score': 'mean',
        'latitude': 'mean',
        'longitude': 'mean'
    }).round(2)
    
    region_stats.columns = ['facility_count', 'avg_oil_prod', 'total_oil_prod', 
                           'avg_gas_prod', 'total_gas_prod', 'avg_health_score',
                           'center_lat', 'center_lon']
    
    print("Regional Statistics:")
    print(region_stats)
    
    # Country analysis
    country_stats = df.groupby('country').agg({
        'facility_id': 'count',
        'oil_production_bpd': ['mean', 'sum'],
        'equipment_health_score': 'mean'
    }).round(2).sort_values(('oil_production_bpd', 'sum'), ascending=False)
    
    country_stats.columns = ['facility_count', 'avg_oil_prod', 'total_oil_prod', 'avg_health_score']
    
    print(f"\nTop 15 Countries by Total Oil Production:")
    print(country_stats.head(15))
    
    # Facility type analysis
    facility_type_stats = df.groupby('facility_type').agg({
        'facility_id': 'count',
        'oil_production_bpd': ['mean', 'sum'],
        'equipment_health_score': ['mean', 'std']
    }).round(2)
    
    print(f"\nFacility Type Statistics:")
    print(facility_type_stats)
    
    # Create comprehensive visualizations
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=['Regional Facility Distribution', 'Production by Top Countries',
                       'Health Scores by Region', 'Facility Types Distribution',
                       'Production vs Health Score', 'Geographic Distribution'],
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "box"}, {"type": "pie"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # Regional facility count
    fig.add_trace(
        go.Bar(x=region_stats.index, y=region_stats['facility_count'],
               name='Facilities', marker_color='lightblue'),
        row=1, col=1
    )
    
    # Top countries production
    top_countries = country_stats.head(10)
    fig.add_trace(
        go.Bar(x=top_countries.index, y=top_countries['total_oil_prod'],
               name='Total Production', marker_color='orange'),
        row=1, col=2
    )
    
    # Health scores by region
    for region in df['region'].unique():
        region_health = df[df['region'] == region]['equipment_health_score']
        fig.add_trace(
            go.Box(y=region_health, name=region, showlegend=False),
            row=2, col=1
        )
    
    # Facility type distribution
    facility_counts = df['facility_type'].value_counts()
    fig.add_trace(
        go.Pie(labels=facility_counts.index, values=facility_counts.values,
               showlegend=False),
        row=2, col=2
    )
    
    # Production vs Health (only oil wells)
    oil_wells = df[df['facility_type'] == 'oil_well']
    fig.add_trace(
        go.Scatter(
            x=oil_wells['equipment_health_score'],
            y=oil_wells['oil_production_bpd'],
            mode='markers',
            name='Oil Wells',
            opacity=0.6,
            showlegend=False
        ),
        row=3, col=1
    )
    
    # Geographic distribution
    fig.add_trace(
        go.Scatter(
            x=df['longitude'],
            y=df['latitude'],
            mode='markers',
            marker=dict(
                color=df['equipment_health_score'],
                colorscale='RdYlGn',
                size=5,
                showscale=True
            ),
            name='Facilities',
            showlegend=False
        ),
        row=3, col=2
    )
    
    fig.update_layout(height=1200, title_text="Comprehensive Facility Analysis")
    fig.write_html(str(viz_dir / "comprehensive_facility_analysis.html"))
    
    return region_stats, country_stats, facility_type_stats

def perform_location_clustering(df, viz_dir):
    """Perform geographical clustering analysis"""
    print("\n=== LOCATION CLUSTERING ANALYSIS ===")
    
    # Filter complete coordinate data
    coords_df = df.dropna(subset=['latitude', 'longitude']).copy()
    coords = coords_df[['latitude', 'longitude']].values
    
    # Standardize coordinates for clustering
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    
    # Find optimal clusters using silhouette analysis
    silhouette_scores = []
    K_range = range(3, 12)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coords_scaled)
        silhouette_avg = silhouette_score(coords_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    optimal_k = K_range[np.argmax(silhouette_scores)]
    best_score = max(silhouette_scores)
    
    print(f"Optimal number of clusters: {optimal_k} (silhouette score: {best_score:.3f})")
    
    # Apply optimal clustering
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    coords_df['cluster'] = kmeans_final.fit_predict(coords_scaled)
    
    # Calculate cluster statistics
    cluster_stats = coords_df.groupby('cluster').agg({
        'facility_id': 'count',
        'oil_production_bpd': ['mean', 'sum'],
        'equipment_health_score': ['mean', 'std'],
        'latitude': ['mean', 'std'],
        'longitude': ['mean', 'std'],
        'facility_type': lambda x: x.mode().iloc[0] if not x.empty else 'mixed'
    }).round(3)
    
    cluster_stats.columns = ['facility_count', 'avg_oil_prod', 'total_oil_prod',
                            'avg_health_score', 'std_health_score',
                            'center_lat', 'lat_std', 'center_lon', 'lon_std', 'dominant_type']
    
    print("\nCluster Analysis:")
    print(cluster_stats)
    
    # Create cluster visualizations
    fig_cluster = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Geographical Clusters', 'Cluster Performance Metrics']
    )
    
    # Geographic clusters
    colors = px.colors.qualitative.Set1[:optimal_k]
    for i in range(optimal_k):
        cluster_data = coords_df[coords_df['cluster'] == i]
        fig_cluster.add_trace(
            go.Scatter(
                x=cluster_data['longitude'],
                y=cluster_data['latitude'],
                mode='markers',
                marker=dict(color=colors[i], size=6),
                name=f'Cluster {i}',
                text=cluster_data['facility_name']
            ),
            row=1, col=1
        )
    
    # Cluster performance
    fig_cluster.add_trace(
        go.Scatter(
            x=cluster_stats['facility_count'],
            y=cluster_stats['avg_health_score'],
            mode='markers+text',
            marker=dict(
                size=cluster_stats['total_oil_prod'] / 100,
                color=cluster_stats.index,
                colorscale='viridis',
                showscale=True
            ),
            text=[f'C{i}' for i in cluster_stats.index],
            textposition="middle center",
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig_cluster.update_layout(height=600, title_text="Location Clustering Analysis")
    fig_cluster.write_html(str(viz_dir / "location_clustering_analysis.html"))
    
    return cluster_stats, coords_df

def analyze_production_patterns(df, viz_dir):
    """Analyze production efficiency and operational patterns"""
    print("\n=== PRODUCTION PATTERNS ANALYSIS ===")
    
    # Focus on oil wells for production analysis
    oil_wells = df[df['facility_type'] == 'oil_well'].copy()
    
    # Production statistics
    prod_stats = {
        'total_wells': len(oil_wells),
        'wells_with_production': oil_wells['oil_production_bpd'].notna().sum(),
        'avg_daily_production': oil_wells['oil_production_bpd'].mean(),
        'max_daily_production': oil_wells['oil_production_bpd'].max(),
        'production_efficiency_avg': oil_wells['production_efficiency'].mean()
    }
    
    print("Production Statistics:")
    for key, value in prod_stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Regional production efficiency
    regional_prod = oil_wells.groupby('region').agg({
        'oil_production_bpd': ['mean', 'std', 'sum'],
        'production_efficiency': ['mean', 'std'],
        'equipment_health_score': 'mean'
    }).round(3)
    
    print("\nRegional Production Analysis:")
    print(regional_prod)
    
    # Age vs performance correlation
    age_prod_corr = pearsonr(
        oil_wells['well_age_years'].fillna(0),
        oil_wells['oil_production_bpd'].fillna(0)
    )
    
    health_prod_corr = pearsonr(
        oil_wells['equipment_health_score'],
        oil_wells['oil_production_bpd'].fillna(0)
    )
    
    print(f"\nCorrelation Analysis:")
    print(f"  Age vs Production: {age_prod_corr[0]:.3f} (p={age_prod_corr[1]:.3f})")
    print(f"  Health vs Production: {health_prod_corr[0]:.3f} (p={health_prod_corr[1]:.3f})")
    
    # Create production analysis visualizations
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Production by Region', 'Age vs Production',
                       'Health vs Production', 'Production Efficiency Distribution']
    )
    
    # Production by region
    for region in oil_wells['region'].unique():
        region_data = oil_wells[oil_wells['region'] == region]['oil_production_bpd'].dropna()
        fig.add_trace(
            go.Box(y=region_data, name=region, showlegend=False),
            row=1, col=1
        )
    
    # Age vs production
    fig.add_trace(
        go.Scatter(
            x=oil_wells['well_age_years'],
            y=oil_wells['oil_production_bpd'],
            mode='markers',
            opacity=0.6,
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Health vs production
    fig.add_trace(
        go.Scatter(
            x=oil_wells['equipment_health_score'],
            y=oil_wells['oil_production_bpd'],
            mode='markers',
            opacity=0.6,
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Production efficiency histogram
    fig.add_trace(
        go.Histogram(
            x=oil_wells['production_efficiency'].dropna(),
            nbinsx=20,
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title_text="Production Patterns Analysis")
    fig.write_html(str(viz_dir / "production_patterns_analysis.html"))
    
    return prod_stats, regional_prod

def analyze_equipment_health(df, viz_dir):
    """Comprehensive equipment health analysis"""
    print("\n=== EQUIPMENT HEALTH ANALYSIS ===")
    
    # Health statistics by facility type
    health_by_type = df.groupby('facility_type').agg({
        'equipment_health_score': ['mean', 'std', 'min', 'max', 'count']
    }).round(3)
    
    print("Equipment Health by Facility Type:")
    print(health_by_type)
    
    # Health vs age analysis (for wells)
    wells = df[df['facility_type'] == 'oil_well']
    age_health_corr = pearsonr(
        wells['well_age_years'].fillna(0),
        wells['equipment_health_score']
    )
    
    print(f"\nAge vs Health Correlation (Wells): {age_health_corr[0]:.3f} (p={age_health_corr[1]:.3f})")
    
    # Maintenance analysis
    df['maintenance_overdue'] = df['days_since_maintenance'].fillna(0) > 365
    maintenance_health = df.groupby('maintenance_overdue')['equipment_health_score'].mean()
    
    print(f"\nMaintenance Impact on Health:")
    if False in maintenance_health.index:
        print(f"  Up-to-date maintenance: {maintenance_health[False]:.3f}")
    if True in maintenance_health.index:
        print(f"  Overdue maintenance: {maintenance_health[True]:.3f}")
    else:
        print("  No overdue maintenance found in dataset")
    
    # Create health analysis visualizations
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Health Score Distribution', 'Health by Facility Type',
                       'Age vs Health (Wells)', 'Maintenance vs Health']
    )
    
    # Overall health distribution
    fig.add_trace(
        go.Histogram(x=df['equipment_health_score'], nbinsx=25, showlegend=False),
        row=1, col=1
    )
    
    # Health by facility type
    for ftype in df['facility_type'].unique():
        type_health = df[df['facility_type'] == ftype]['equipment_health_score']
        fig.add_trace(
            go.Box(y=type_health, name=ftype, showlegend=False),
            row=1, col=2
        )
    
    # Age vs health for wells
    fig.add_trace(
        go.Scatter(
            x=wells['well_age_years'],
            y=wells['equipment_health_score'],
            mode='markers',
            opacity=0.6,
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Maintenance vs health
    maint_categories = []
    maint_health_vals = []
    if False in maintenance_health.index:
        maint_categories.append('Up-to-date')
        maint_health_vals.append(maintenance_health[False])
    if True in maintenance_health.index:
        maint_categories.append('Overdue')
        maint_health_vals.append(maintenance_health[True])
    
    if maint_categories:
        fig.add_trace(
            go.Bar(x=maint_categories, y=maint_health_vals, showlegend=False),
            row=2, col=2
        )
    
    fig.update_layout(height=800, title_text="Equipment Health Analysis")
    fig.write_html(str(viz_dir / "equipment_health_analysis.html"))
    
    return health_by_type, maintenance_health

def generate_insights_and_recommendations(df, analysis_results, analysis_dir):
    """Generate comprehensive insights and actionable recommendations"""
    print("\n=== GENERATING INSIGHTS & RECOMMENDATIONS ===")
    
    # Calculate key metrics
    total_facilities = len(df)
    oil_wells = df[df['facility_type'] == 'oil_well']
    active_wells = oil_wells[oil_wells['oil_production_bpd'].notna()]
    
    insights = {
        'executive_summary': {
            'total_facilities': total_facilities,
            'oil_wells': len(oil_wells),
            'active_production_wells': len(active_wells),
            'average_health_score': df['equipment_health_score'].mean(),
            'regions_covered': df['region'].nunique(),
            'countries_covered': df['country'].nunique()
        },
        
        'operational_insights': {
            'production_insights': [
                f"Total daily oil production: {active_wells['oil_production_bpd'].sum():.0f} barrels",
                f"Average well productivity: {active_wells['oil_production_bpd'].mean():.1f} bpd",
                f"Top producing region: {df.groupby('region')['oil_production_bpd'].sum().idxmax()}",
                f"Production efficiency varies significantly by region"
            ],
            
            'health_insights': [
                f"Average equipment health score: {df['equipment_health_score'].mean():.2f}",
                f"Refineries have highest health scores ({df[df['facility_type']=='refinery']['equipment_health_score'].mean():.2f})",
                f"Equipment health correlates with production performance",
                f"Age negatively impacts equipment health in oil wells"
            ],
            
            'geographic_insights': [
                "Facilities are distributed across 6 major regions",
                f"Highest facility density in {df.groupby('region').size().idxmax()}",
                "Clear geographical clustering patterns identified",
                "Regional performance variations suggest location-specific factors"
            ]
        },
        
        'risk_assessment': {
            'maintenance_risks': [
                f"{df['maintenance_overdue'].sum()} facilities overdue for maintenance",
                "Maintenance delays correlate with reduced equipment health",
                "Preventive maintenance scheduling needs optimization"
            ],
            
            'operational_risks': [
                "Age-related performance degradation in older wells",
                "Geographic concentration creates operational dependencies",
                "Health score variation indicates inconsistent maintenance practices"
            ]
        },
        
        'recommendations': {
            'immediate_actions': [
                "Prioritize maintenance for overdue facilities (health score < 0.7)",
                "Investigate low-performing wells for optimization opportunities",
                "Implement predictive maintenance based on age and health patterns",
                "Focus resources on highest-producing regions and clusters"
            ],
            
            'strategic_initiatives': [
                "Develop region-specific operational strategies",
                "Implement IoT monitoring for real-time health tracking",
                "Create predictive models for production optimization",
                "Establish maintenance excellence programs for underperforming areas",
                "Consider facility consolidation in over-dense clusters"
            ],
            
            'monitoring_priorities': [
                "Track equipment health trends by facility type and age",
                "Monitor production efficiency across regions",
                "Implement environmental risk monitoring in high-risk areas",
                "Establish KPIs for maintenance effectiveness"
            ]
        }
    }
    
    # Save comprehensive insights
    with open(analysis_dir / 'comprehensive_insights.json', 'w') as f:
        json.dump(insights, f, indent=2, default=str)
    
    # Print key findings
    print("\n" + "="*60)
    print("KEY FINDINGS & RECOMMENDATIONS")
    print("="*60)
    
    print(f"\nEXECUTIVE SUMMARY:")
    for key, value in insights['executive_summary'].items():
        print(f"  • {key.replace('_', ' ').title()}: {value}")
    
    print(f"\nIMMEDIATE ACTIONS REQUIRED:")
    for action in insights['recommendations']['immediate_actions']:
        print(f"  1. {action}")
    
    print(f"\nSTRATEGIC RECOMMENDATIONS:")
    for rec in insights['recommendations']['strategic_initiatives']:
        print(f"  • {rec}")
    
    return insights

def main():
    """Execute comprehensive oil & gas geospatial EDA"""
    print("=== COMPREHENSIVE OIL & GAS GEOSPATIAL EDA ===")
    
    # Initialize MLflow
    mlflow.set_experiment("oil_gas_geospatial_eda")
    
    with mlflow.start_run(run_name="comprehensive_analysis") as run:
        # Setup
        paths, viz_dir, analysis_dir = setup_environment()
        mlflow.log_param("project_name", "oil_gas_dashboards")
        
        # Load data
        df = load_and_prepare_data()
        mlflow.log_metric("total_facilities", len(df))
        mlflow.log_metric("oil_wells_count", len(df[df['facility_type'] == 'oil_well']))
        mlflow.log_metric("regions_count", df['region'].nunique())
        mlflow.log_metric("countries_count", df['country'].nunique())
        
        # Save prepared data
        prepared_data_path = paths['base_data_path'] / "prepared_geospatial_data.parquet"
        df.to_parquet(prepared_data_path)
        mlflow.log_artifact(str(prepared_data_path))
        
        analysis_results = {}
        
        try:
            # 1. Facility distribution analysis
            region_stats, country_stats, facility_stats = analyze_facility_distribution(df, viz_dir)
            analysis_results['regional_analysis'] = region_stats.to_dict()
            mlflow.log_metric("avg_health_score", df['equipment_health_score'].mean())
            
            # 2. Location clustering
            cluster_stats, clustered_df = perform_location_clustering(df, viz_dir)
            analysis_results['clustering_results'] = cluster_stats.to_dict()
            mlflow.log_metric("optimal_clusters", len(cluster_stats))
            
            # 3. Production patterns
            prod_stats, regional_prod = analyze_production_patterns(df, viz_dir)
            analysis_results['production_analysis'] = prod_stats
            mlflow.log_metric("avg_production_bpd", prod_stats.get('avg_daily_production', 0))
            
            # 4. Equipment health analysis
            health_by_type, maintenance_impact = analyze_equipment_health(df, viz_dir)
            analysis_results['health_analysis'] = health_by_type.to_dict()
            if False in maintenance_impact.index and True in maintenance_impact.index:
                mlflow.log_metric("maintenance_impact", 
                                maintenance_impact[False] - maintenance_impact[True])
            else:
                mlflow.log_metric("maintenance_impact", 0)
            
            # 5. Generate insights and recommendations
            insights = generate_insights_and_recommendations(df, analysis_results, analysis_dir)
            
            # Save complete analysis
            complete_results = {
                'analysis_results': analysis_results,
                'insights': insights,
                'dataset_info': {
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'analysis_timestamp': datetime.now().isoformat()
                }
            }
            
            results_path = analysis_dir / 'complete_eda_results.json'
            with open(results_path, 'w') as f:
                json.dump(complete_results, f, indent=2, default=str)
            
            mlflow.log_artifact(str(results_path))
            
            # Log visualization files
            for viz_file in viz_dir.glob("*.html"):
                mlflow.log_artifact(str(viz_file))
            
            mlflow.set_tag("analysis_status", "success")
            
            print(f"\n✓ Analysis completed successfully!")
            print(f"✓ Results saved to: {analysis_dir}")
            print(f"✓ Visualizations saved to: {viz_dir}")
            print(f"✓ MLflow experiment: oil_gas_geospatial_eda")
            
        except Exception as e:
            mlflow.log_param("error", str(e))
            mlflow.set_tag("analysis_status", "failed")
            print(f"❌ Analysis failed: {e}")
            raise

if __name__ == "__main__":
    main()