#!/usr/bin/env python3
"""
Data Availability Checker and Generator

This module ensures that required data exists before model training.
If data is missing, it automatically generates it.

Use this in Domino Flows to handle data generation within each task.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/mnt/code')
from scripts.data_config import get_data_paths
from scripts.oil_gas_data_generator import OilGasDataGenerator

logger = logging.getLogger(__name__)

def ensure_data_exists(project_name: str = 'Oil-and-Gas-Demo', force_regenerate: bool = False):
    """
    Ensure required data files exist, generating them if necessary.

    Args:
        project_name: Name of the project
        force_regenerate: If True, regenerate data even if it exists

    Returns:
        dict: Paths to data files

    Raises:
        RuntimeError: If data generation fails
    """
    paths = get_data_paths(project_name)
    data_dir = paths['base_data_path']

    required_files = [
        'production_timeseries.parquet',
        'prices_timeseries.parquet',
        'demand_timeseries.parquet',
        'maintenance_timeseries.parquet'
    ]

    # Check if all required files exist
    missing_files = []
    for filename in required_files:
        filepath = data_dir / filename
        if not filepath.exists():
            missing_files.append(filename)

    if not missing_files and not force_regenerate:
        logger.info(f"All required data files found in {data_dir}")
        return {
            'data_dir': data_dir,
            'files': {f: data_dir / f for f in required_files}
        }

    # Data is missing or regeneration requested
    if missing_files:
        logger.warning(f"Missing data files: {missing_files}")
        logger.info(f"Generating data in {data_dir}...")
    else:
        logger.info(f"Force regenerating data in {data_dir}...")

    try:
        # Generate data
        generator = OilGasDataGenerator(project_name)

        # Generate geospatial data
        logger.info("Generating geospatial facility data...")
        geospatial_df = generator.generate_geospatial_data(
            n_wells=1500,
            n_refineries=25,
            n_facilities=200
        )

        # Generate time series data
        logger.info("Generating time series data...")
        timeseries_dict = generator.generate_time_series_data(
            geospatial_df,
            start_date='2022-01-01',
            end_date='2025-11-01'
        )

        # Save all data
        logger.info("Saving generated data...")
        saved_paths = generator.save_datasets(geospatial_df, timeseries_dict)

        logger.info("Data generation completed successfully")
        logger.info(f"Generated files: {list(saved_paths.keys())}")

        return {
            'data_dir': data_dir,
            'files': {f: data_dir / f for f in required_files},
            'saved_paths': saved_paths
        }

    except Exception as e:
        error_msg = f"Failed to generate data: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def check_data_availability(project_name: str = 'Oil-and-Gas-Demo'):
    """
    Check which data files are available without generating.

    Args:
        project_name: Name of the project

    Returns:
        dict: Status of each required file
    """
    paths = get_data_paths(project_name)
    data_dir = paths['base_data_path']

    required_files = [
        'production_timeseries.parquet',
        'prices_timeseries.parquet',
        'demand_timeseries.parquet',
        'maintenance_timeseries.parquet',
        'geospatial_facilities.parquet'
    ]

    status = {
        'data_dir': str(data_dir),
        'data_dir_exists': data_dir.exists(),
        'files': {}
    }

    for filename in required_files:
        filepath = data_dir / filename
        status['files'][filename] = {
            'path': str(filepath),
            'exists': filepath.exists(),
            'size_mb': round(filepath.stat().st_size / (1024 * 1024), 2) if filepath.exists() else 0
        }

    return status


if __name__ == "__main__":
    import json

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("Data Availability Check")
    print("=" * 60)

    # Check current status
    status = check_data_availability()
    print(json.dumps(status, indent=2))

    # Check if generation is needed
    missing = [f for f, info in status['files'].items() if not info['exists']]

    if missing:
        print(f"\nMissing files: {missing}")
        print("Generating data...")
        result = ensure_data_exists()
        print("Data generation complete!")
    else:
        print("\nAll required data files are available.")
