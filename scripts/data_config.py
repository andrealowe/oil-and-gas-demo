"""
Data Configuration Utility for Domino Data Lab Projects

This utility determines correct storage paths based on project type:
- Git-based projects (DOMINO_WORKING_DIR=/mnt/code)
- DFS projects (DOMINO_WORKING_DIR=/mnt)

Usage:
    from scripts.data_config import get_data_paths
    paths = get_data_paths('my_project')
    data_dir = paths['base_data_path']
    artifacts_dir = paths['artifacts_path']
"""

import os
from pathlib import Path
from typing import Dict, Union

def get_data_paths(project_name: str) -> Dict[str, Union[Path, bool]]:
    """
    Get correct data storage paths based on Domino project type.
    
    Args:
        project_name: Name of the project for data organization
        
    Returns:
        Dictionary containing:
        - base_data_path: Where to store datasets
        - artifacts_path: Where to store models/reports
        - is_git_based: Boolean indicating project type
    """
    # Determine project type based on DOMINO_WORKING_DIR
    working_dir = os.environ.get('DOMINO_WORKING_DIR', '/mnt/code')
    is_git_based = working_dir == '/mnt/code'
    
    if is_git_based:
        # Git-based project
        # For Oil-and-Gas-Demo project, data is stored in /mnt/data/
        if project_name == 'Oil-and-Gas-Demo' and Path('/mnt/data/Oil-and-Gas-Demo').exists():
            base_data_path = Path('/mnt/data/Oil-and-Gas-Demo')
        else:
            # Check if /mnt/data is writable, otherwise use /mnt/artifacts/data
            test_data_dir = Path('/mnt/data')
            if test_data_dir.exists() and os.access(test_data_dir, os.W_OK):
                datasets_dir = os.environ.get('DOMINO_DATASETS_DIR', f'/mnt/data')
                base_data_path = Path(datasets_dir) / project_name
            else:
                # Fallback to artifacts directory for data storage
                base_data_path = Path('/mnt/artifacts') / 'data' / project_name
        artifacts_path = Path('/mnt/artifacts')
    else:
        # DFS project
        base_data_path = Path(f'/domino/datasets/local/{project_name}')
        artifacts_path = Path('/mnt')
    
    return {
        'base_data_path': base_data_path,
        'artifacts_path': artifacts_path,
        'is_git_based': is_git_based
    }

def ensure_directories(project_name: str) -> Dict[str, Path]:
    """
    Ensure all necessary directories exist and return paths.
    
    Args:
        project_name: Name of the project
        
    Returns:
        Dictionary of created directory paths
    """
    paths = get_data_paths(project_name)
    
    # Create base directories
    directories = {
        'data': paths['base_data_path'],
        'artifacts': paths['artifacts_path'],
        'models': paths['artifacts_path'] / 'models',
        'reports': paths['artifacts_path'] / 'reports',
        'visualizations': paths['artifacts_path'] / 'visualizations'
    }
    
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return directories