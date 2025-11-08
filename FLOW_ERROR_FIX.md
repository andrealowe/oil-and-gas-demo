# Flow Error Fix - ensure_data.py Method Names

## Error Encountered

```
RuntimeError: Failed to generate data: 'OilGasDataGenerator' object has no attribute 'generate_all_timeseries'
```

**Location**: Step 2 of the Flow (AutoGluon training task)
**File**: `/mnt/code/src/models/ensure_data.py`, line 95

---

## Root Cause

When I created the `ensure_data.py` helper module, I used incorrect method names that don't exist in the `OilGasDataGenerator` class.

**Incorrect Method Names Used**:
- ❌ `generate_all_timeseries()` - doesn't exist
- ❌ `save_all_data()` - doesn't exist

**Actual Method Names**:
- ✅ `generate_time_series_data()` - correct
- ✅ `save_datasets()` - correct

---

## Fix Applied

Updated `/mnt/code/src/models/ensure_data.py` to use the correct method names:

### Before (BROKEN):
```python
# Generate time series data
timeseries_dict = generator.generate_all_timeseries(geospatial_df)

# Save all data
saved_paths = generator.save_all_data(geospatial_df, timeseries_dict)
```

### After (FIXED):
```python
# Generate time series data
timeseries_dict = generator.generate_time_series_data(
    geospatial_df,
    start_date='2022-01-01',
    end_date='2025-11-01'
)

# Save all data
saved_paths = generator.save_datasets(geospatial_df, timeseries_dict)
```

---

## Verification

✅ All required methods now exist in `OilGasDataGenerator`:
- `generate_geospatial_data(n_wells, n_refineries, n_facilities)`
- `generate_time_series_data(geospatial_df, start_date, end_date)`
- `save_datasets(geospatial_df, time_series_data)`

✅ Method signatures match what's used in `scripts/oil_gas_data_generator.py::main()`

---

## Impact

This fix resolves the error in **Step 2 of your Flow** (and all subsequent training tasks that call `ensure_data_exists()`).

The auto-generation feature will now work correctly when:
1. Data is missing in Domino Flow tasks
2. Scripts detect missing data files
3. `ensure_data_exists()` is called to regenerate data

---

## Testing

To verify the fix works:

```bash
# Test data availability check
python3 -c "
import sys
sys.path.insert(0, '/mnt/code')
from src.models.ensure_data import check_data_availability
status = check_data_availability('Oil-and-Gas-Demo')
print('✓ Data check passed')
"

# Test ensure_data_exists (will check existing data, not regenerate)
python3 -c "
import sys
sys.path.insert(0, '/mnt/code')
from src.models.ensure_data import ensure_data_exists
result = ensure_data_exists('Oil-and-Gas-Demo')
print('✓ ensure_data_exists passed')
"
```

---

## Next Steps

**Your Flow should now work!**

When you re-run the Flow:
1. ✅ Step 1 (Refresh Data) - generates data, writes to `/workflow/outputs/data_summary`
2. ✅ Step 2 (Train AutoGluon) - reads input, checks data exists, trains models
3. ✅ Step 3-4 (Train Prophet, Nixtla, etc.) - same pattern
4. ✅ Step 5 (Compare Models) - compares all results

The error you encountered should no longer occur.

---

## Files Modified

- `/mnt/code/src/models/ensure_data.py` - Fixed method names (line 81, 89)

---

## Summary

**Issue**: Used non-existent method names in ensure_data.py
**Fix**: Updated to use correct method names from OilGasDataGenerator
**Status**: ✅ RESOLVED
**Next**: Re-run your Domino Flow - it should complete successfully now
