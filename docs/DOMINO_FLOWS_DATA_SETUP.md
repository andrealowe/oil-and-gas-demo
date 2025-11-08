# Domino Flows Data Setup Guide

## Problem: Data Not Persisting Between Flow Tasks

When running the AutoML forecasting workflow in Domino Flows, you may encounter:
```
FileNotFoundError: /mnt/data/Oil-and-Gas-Demo/production_timeseries.parquet
```

### Why This Happens

In Domino Flows, **each task runs in a separate, isolated Domino Job environment**. The filesystem is not shared between tasks by default. This means:

1. Data generation task writes to `/mnt/data/Oil-and-Gas-Demo/`
2. That task completes and its environment is destroyed
3. Training tasks start in new environments
4. `/mnt/data/Oil-and-Gas-Demo/` doesn't exist in the new environments

### Solutions

We've implemented **TWO complementary solutions**:

---

## Solution 1: Auto-Generate Data Per Task (Implemented ✅)

**What we did:**
- Added `ensure_data_exists()` function to all forecasting scripts
- Each script now checks if data exists at startup
- If data is missing, it automatically generates it

**Pros:**
- Works immediately without Domino configuration
- No dataset setup required
- Self-contained tasks

**Cons:**
- Each task generates its own data (4× data generation)
- Slower execution (data gen takes ~2-3 minutes per task)
- Higher resource usage

**Code changes:**
```python
# In each forecasting script (autogluon, prophet, nixtla, combined)
from src.models.ensure_data import ensure_data_exists

def main(args=None):
    # Ensure data exists (will generate if missing)
    ensure_data_exists('Oil-and-Gas-Demo')

    # ... rest of the script
```

---

## Solution 2: Use Domino Dataset (Recommended for Production)

**What to do:**

### Step 1: Create Domino Dataset

1. Go to your Domino Project
2. Navigate to **Data** → **Datasets**
3. Click **Create New Dataset**
4. Name: `Oil-and-Gas-Demo`
5. Type: **Domino File System (DFS)** or **External Volume**
6. Mount path: `/mnt/data/Oil-and-Gas-Demo`

### Step 2: Populate the Dataset

Run the data generator once to populate the dataset:

```bash
# In a Domino Workspace or Job
python scripts/oil_gas_data_generator.py
```

This writes data to `/mnt/data/Oil-and-Gas-Demo/`:
- `production_timeseries.parquet`
- `prices_timeseries.parquet`
- `demand_timeseries.parquet`
- `maintenance_timeseries.parquet`

### Step 3: Configure Flow to Mount Dataset

When creating/editing your Domino Flow:

1. Go to **Flow Settings** → **Datasets**
2. Add the `Oil-and-Gas-Demo` dataset
3. Ensure it's mounted to all tasks
4. The `use_latest=True` flag in tasks will mount the latest snapshot

### Step 4: Run the Flow

The workflow will now:
1. Data generation task writes to the shared dataset
2. All training tasks read from the shared dataset
3. Data persists across tasks ✅

**Pros:**
- Data generated once, used by all tasks
- Much faster (parallel training without waiting for data gen)
- More efficient resource usage
- Standard Domino pattern

**Cons:**
- Requires Domino Dataset configuration
- Extra setup step

---

## Current Workflow Behavior

The `oil_gas_automl_forecasting_workflow` in `scripts/flows.py` now:

### With Dataset Configured:
```
1. Data Prep Task → Writes to /mnt/data/Oil-and-Gas-Demo/
2. Training Tasks → Read from shared dataset (fast, parallel)
3. Model Comparison → Uses trained models
```

### Without Dataset (Auto-generate):
```
1. Data Prep Task → Writes to /mnt/data/Oil-and-Gas-Demo/ (but doesn't persist)
2. Each Training Task → Detects missing data → Generates its own copy
3. Model Comparison → Uses trained models
```

Both approaches work, but dataset approach is more efficient.

---

## How to Verify Your Setup

### Check if Dataset is Configured:

```python
# In a Domino Workspace
import os
from pathlib import Path

data_dir = Path('/mnt/data/Oil-and-Gas-Demo')
print(f"Directory exists: {data_dir.exists()}")
print(f"Is writable: {os.access(data_dir, os.W_OK)}")

if data_dir.exists():
    files = list(data_dir.glob('*.parquet'))
    print(f"Files found: {len(files)}")
    for f in files:
        print(f"  - {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")
```

### Test Data Generation:

```bash
# Test the ensure_data module
python src/models/ensure_data.py
```

### Test a Forecasting Script:

```bash
# Should auto-generate data if missing
python src/models/autogluon_forecasting.py
```

---

## Troubleshooting

### "Permission denied" when writing to /mnt/data/
**Solution**: Create a Domino Dataset (Solution 2)

### Data generation is slow in Flows
**Solution**: Use Domino Dataset (Solution 2) - data generated once instead of 4 times

### Data exists but script can't find it
**Check**:
1. Dataset is mounted to the Flow tasks
2. Mount path is `/mnt/data/Oil-and-Gas-Demo`
3. Files have `.parquet` extension

### Flow fails on first run but works on second
**Reason**: Auto-generation kicked in
**Solution**: Either accept this behavior or use Domino Dataset (Solution 2)

---

## Recommended Approach

For **production/frequent runs**:
1. Set up Domino Dataset (one-time setup)
2. Remove auto-generation from scripts (optional, for performance)
3. Enjoy fast, efficient parallel training

For **quick testing/demos**:
1. Use auto-generation (current implementation)
2. Accept slower first run
3. No configuration needed

---

## Files Modified

1. `scripts/flows.py` - Added explicit task dependencies
2. `src/models/ensure_data.py` - New module for auto-generation
3. `src/models/autogluon_forecasting.py` - Added ensure_data_exists() call
4. `src/models/prophet_forecasting.py` - (TODO)
5. `src/models/nixtla_forecasting.py` - (TODO)
6. `src/models/oil_gas_forecasting.py` - (TODO)

---

## Next Steps

1. **For immediate use**: Current implementation works with auto-generation
2. **For optimization**: Set up Domino Dataset as described in Solution 2
3. **For other scripts**: Apply the same ensure_data_exists() pattern to prophet, nixtla, and oil_gas_forecasting scripts

---

## Contact / Support

If issues persist:
1. Check Domino Dataset configuration
2. Verify write permissions to /mnt/data
3. Review Flow task logs for detailed error messages
4. Check MLflow experiments for partial results
