# Data Path Troubleshooting Guide

## Issue: FileNotFoundError for production_timeseries.parquet

### Problem
When running forecasting scripts (especially in Domino Flows), you may encounter:
```
FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/Oil-and-Gas-Demo/production_timeseries.parquet'
```

### Root Cause
The error occurs because:
1. **Domino Flows execution context**: Flows run in isolated job environments where `/mnt/data/Oil-and-Gas-Demo/` may not exist unless:
   - Data is generated as part of the workflow
   - OR it's configured as a Domino Dataset and mounted

2. **Missing data generation step**: The original workflow assumed data already existed

### Solution Applied

#### 1. Fixed Data Path Configuration
Updated `/mnt/code/scripts/data_config.py` to always use correct paths:
- **Data files** (.parquet, .csv): `/mnt/data/Oil-and-Gas-Demo/`
- **Artifacts** (.pkl, models): `/mnt/artifacts/`

#### 2. Added Data Generation to Workflow
Updated `scripts/flows.py` - the main `oil_gas_automl_forecasting_workflow` now includes:
```python
# Step 1: Generate data BEFORE training
data_prep_task = DominoJobTask(
    name="Generate Oil & Gas Synthetic Data",
    domino_job_config=DominoJobConfig(
        Command="python scripts/oil_gas_data_generator.py"
    ),
    ...
)
```

### How to Use

#### Option 1: Local Development (Recommended for Testing)
```bash
# 1. Generate data first
python scripts/oil_gas_data_generator.py

# 2. Run individual model training scripts
python src/models/autogluon_forecasting.py
python src/models/prophet_forecasting.py
python src/models/nixtla_forecasting.py
python src/models/oil_gas_forecasting.py

# 3. Compare models
python src/models/model_comparison.py
```

#### Option 2: Domino Flows (Production)
The `oil_gas_automl_forecasting_workflow` now automatically:
1. Generates data (if needed)
2. Trains all models in parallel
3. Compares and selects champion model

No manual data generation required!

### Verification

To verify data exists and is accessible:
```bash
# Check data directory
ls -la /mnt/data/Oil-and-Gas-Demo/

# Test data access from Python
python3 -c "
import sys
sys.path.insert(0, '/mnt/code')
from scripts.data_config import get_data_paths
import pandas as pd

paths = get_data_paths('Oil-and-Gas-Demo')
data_path = paths['base_data_path'] / 'production_timeseries.parquet'
print(f'Data path: {data_path}')
print(f'Exists: {data_path.exists()}')

if data_path.exists():
    df = pd.read_parquet(data_path)
    print(f'Shape: {df.shape}')
    print('✓ Data accessible')
"
```

### Files Updated

1. **scripts/data_config.py** - Simplified data path logic
2. **scripts/flows.py** - Added data generation step to main workflow
3. **src/models/explore_timeseries_data.py** - Fixed hardcoded path
4. **src/models/oil_gas_geospatial_models.py** - Fixed hardcoded path
5. **notebooks/oil_gas_model_analysis.ipynb** - Updated to use dynamic paths

### Path Configuration Summary

All scripts now use the centralized `get_data_paths()` utility:

```python
from scripts.data_config import get_data_paths

paths = get_data_paths('Oil-and-Gas-Demo')
data_dir = paths['base_data_path']      # /mnt/data/Oil-and-Gas-Demo/
artifacts_dir = paths['artifacts_path']  # /mnt/artifacts/
```

### For Domino Dataset Integration (Optional)

If you want to persist data as a Domino Dataset:
1. Create a Domino Dataset named "Oil-and-Gas-Demo"
2. Mount it to `/mnt/data/Oil-and-Gas-Demo`
3. Run data generator once to populate it
4. Future workflow runs will use the existing data (via caching)

### Contact

If you continue to experience issues:
1. Check that `/mnt/data/Oil-and-Gas-Demo/` directory exists and is writable
2. Verify `DOMINO_WORKING_DIR` environment variable is set to `/mnt/code`
3. Ensure all scripts import from `scripts.data_config` module
