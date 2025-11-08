# Domino Flows Data Path Fix - Complete Summary

## Problem Solved

**Issue**: When running `scripts/flows.py` AutoML workflow in Domino Flows, training tasks failed with:
```
FileNotFoundError: /mnt/data/Oil-and-Gas-Demo/production_timeseries.parquet
```

**Root Cause**: In Domino Flows, each task runs in an isolated environment. Data written by one task isn't automatically available to subsequent tasks unless stored in a shared Domino Dataset.

---

## Solution Implemented

We implemented **TWO complementary solutions** to ensure the workflow works in both scenarios:

### 1. Fixed Task Dependencies in Workflow ✅
**File**: `scripts/flows.py`

**Changes**:
- Added explicit `inputs={"data_prep": FlyteFile}` to all training tasks
- Training tasks now receive `data_result["data_summary"]` as input
- This creates a dependency chain: data generation → training tasks

**Result**: Tasks execute in the correct order, but data still needs to persist.

### 2. Auto-Generate Data if Missing ✅
**New File**: `src/models/ensure_data.py`

**Functionality**:
- Checks if required data files exist
- Automatically generates data if missing
- Provides detailed status reporting

**Updated Files** (all forecasting scripts):
- `src/models/autogluon_forecasting.py`
- `src/models/prophet_forecasting.py`
- `src/models/nixtla_forecasting.py`
- `src/models/oil_gas_forecasting.py`

Each script now calls `ensure_data_exists()` at startup, guaranteeing data availability.

---

## How It Works Now

### Scenario A: With Domino Dataset (Recommended)
1. User creates Domino Dataset "Oil-and-Gas-Demo" mounted to `/mnt/data/Oil-and-Gas-Demo`
2. Data generation task writes to shared dataset
3. All training tasks read from shared dataset
4. **Fast & efficient** - data generated once

### Scenario B: Without Domino Dataset (Auto-fallback)
1. Data generation task completes but data doesn't persist
2. Each training task detects missing data
3. Each task auto-generates its own data copy
4. **Works but slower** - data generated 4 times (once per task)

**Both scenarios work!** The workflow is now resilient to either configuration.

---

## Files Modified

### Core Fixes
1. **scripts/data_config.py**
   - Simplified to always use `/mnt/data/{project}` for data
   - Removed incorrect fallback to `/mnt/artifacts/data`

2. **scripts/flows.py**
   - Added data generation task as first step
   - Added explicit task dependencies via inputs/outputs
   - Updated documentation

3. **src/models/ensure_data.py** (NEW)
   - Auto-detection and generation of missing data
   - Status checking and reporting

### Forecasting Scripts Updated
4. **src/models/autogluon_forecasting.py**
   - Added `ensure_data_exists()` call
   - Improved error messages

5. **src/models/prophet_forecasting.py**
   - Added `ensure_data_exists()` call

6. **src/models/nixtla_forecasting.py**
   - Added `ensure_data_exists()` call

7. **src/models/oil_gas_forecasting.py**
   - Added `ensure_data_exists()` call

### Fixed Hardcoded Paths
8. **src/models/explore_timeseries_data.py**
   - Changed from hardcoded `/mnt/artifacts/...` to dynamic `paths['base_data_path']`

9. **src/models/oil_gas_geospatial_models.py**
   - Fixed hardcoded data path

10. **notebooks/oil_gas_model_analysis.ipynb**
    - Updated to use `get_data_paths()`

11. **notebooks/oil_gas_model_analysis.py**
    - Updated notebook generator

### Documentation Added
12. **docs/DATA_PATH_TROUBLESHOOTING.md**
    - Detailed troubleshooting guide

13. **docs/DOMINO_FLOWS_DATA_SETUP.md**
    - Complete setup guide for both scenarios
    - Step-by-step Domino Dataset configuration
    - Verification procedures

---

## Testing & Verification

### Test Data Availability
```bash
# Check current data status
python src/models/ensure_data.py
```

### Test Individual Scripts
```bash
# Each should work independently now
python src/models/autogluon_forecasting.py
python src/models/prophet_forecasting.py
python src/models/nixtla_forecasting.py
python src/models/oil_gas_forecasting.py
```

### Test Workflow Info
```bash
# View workflow documentation
python scripts/flows.py
```

### Verify Path Configuration
```bash
python3 -c "
import sys
sys.path.insert(0, '/mnt/code')
from scripts.data_config import get_data_paths

paths = get_data_paths('Oil-and-Gas-Demo')
print(f'Data directory: {paths[\"base_data_path\"]}')
print(f'Artifacts directory: {paths[\"artifacts_path\"]}')
print(f'Data exists: {paths[\"base_data_path\"].exists()}')
"
```

---

## Recommended Next Steps

### For Immediate Use (Current State)
✅ **Workflow works now** - run it in Domino Flows
- Auto-generation will handle missing data
- Expect 4× data generation (2-3 min per task)
- Total workflow time: ~30-40 minutes

### For Optimized Production Use
1. **Create Domino Dataset**:
   - Name: `Oil-and-Gas-Demo`
   - Type: Domino File System (DFS)
   - Mount: `/mnt/data/Oil-and-Gas-Demo`

2. **Populate Dataset Once**:
   ```bash
   python scripts/oil_gas_data_generator.py
   ```

3. **Configure Flow to Mount Dataset**:
   - Flow Settings → Datasets → Add "Oil-and-Gas-Demo"
   - All tasks will now share the same data

4. **Benefit**:
   - Data generated once, shared across all tasks
   - Faster execution (~20-25 minutes total)
   - More efficient resource usage

---

## Path Configuration Reference

### Correct Paths (After Fix)
```
Data files (.parquet, .csv):
  ├─ /mnt/data/Oil-and-Gas-Demo/production_timeseries.parquet ✓
  ├─ /mnt/data/Oil-and-Gas-Demo/prices_timeseries.parquet ✓
  ├─ /mnt/data/Oil-and-Gas-Demo/demand_timeseries.parquet ✓
  └─ /mnt/data/Oil-and-Gas-Demo/maintenance_timeseries.parquet ✓

Artifacts (.pkl, models, reports):
  ├─ /mnt/artifacts/models/*.pkl ✓
  ├─ /mnt/artifacts/reports/*.json ✓
  └─ /mnt/artifacts/visualizations/*.png ✓
```

### All Scripts Use Dynamic Paths
```python
from scripts.data_config import get_data_paths

paths = get_data_paths('Oil-and-Gas-Demo')
data_dir = paths['base_data_path']        # /mnt/data/Oil-and-Gas-Demo
artifacts_dir = paths['artifacts_path']    # /mnt/artifacts
```

---

## Troubleshooting

### "Still getting FileNotFoundError"
**Check**:
1. Is `/mnt/data` directory writable?
   ```bash
   ls -la /mnt/data
   ```
2. Is auto-generation running?
   - Look for "Checking data availability..." in logs
   - Should see "Generating data..." if missing

### "Workflow is slow"
**Reason**: Auto-generation running in each task (4 times)
**Solution**: Set up Domino Dataset (see Optimized Production Use above)

### "Permission denied"
**Likely Cause**: `/mnt/data` not configured as Domino Dataset
**Solution**: Create and mount Domino Dataset

---

## Success Criteria

Your workflow is working correctly if:

✅ Workflow completes without FileNotFoundError
✅ All 4 model training tasks complete successfully
✅ Model comparison runs and selects champion
✅ MLflow experiments show all runs
✅ Models saved to `/mnt/artifacts/models/`

---

## Summary

**The workflow now works in both scenarios:**
1. **With Domino Dataset**: Efficient, fast, recommended for production
2. **Without Domino Dataset**: Auto-generates data, slower but functional

**No more manual data preparation required!** The scripts handle it automatically.

**For questions or issues**, refer to:
- `/mnt/code/docs/DOMINO_FLOWS_DATA_SETUP.md` - Detailed setup guide
- `/mnt/code/docs/DATA_PATH_TROUBLESHOOTING.md` - Troubleshooting steps
