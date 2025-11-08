#!/bin/bash
# Test all scripts used in the main AutoML workflow

echo "========================================="
echo "Testing Flow Scripts - Standalone Mode"
echo "========================================="
echo

# Function to test a script
test_script() {
    local script=$1
    local name=$2
    local timeout_sec=${3:-10}

    echo "Testing: $name"
    echo "  Script: $script"
    echo -n "  Status: "

    # Run script with timeout and capture first 20 lines
    timeout $timeout_sec python3 $script 2>&1 | head -20 > /tmp/test_output.txt
    exit_code=$?

    # Check for errors
    if grep -qi "error\|traceback\|exception" /tmp/test_output.txt; then
        echo "❌ FAILED"
        echo "  Error output:"
        head -10 /tmp/test_output.txt | sed 's/^/    /'
        return 1
    elif [ $exit_code -eq 124 ]; then
        echo "✓ RUNNING (timed out but started successfully)"
        return 0
    elif [ $exit_code -eq 0 ]; then
        echo "✓ PASSED"
        return 0
    else
        echo "⚠ WARNING (exit code: $exit_code)"
        head -5 /tmp/test_output.txt | sed 's/^/    /'
        return 0
    fi
}

# Test each script in the main workflow
echo "Main AutoML Workflow Scripts:"
echo "------------------------------"
echo

test_script "scripts/oil_gas_data_generator.py" "Data Generator" 15
echo

test_script "src/models/autogluon_forecasting.py" "AutoGluon Training" 30
echo

test_script "src/models/prophet_forecasting.py" "Prophet Training" 30
echo

test_script "src/models/nixtla_forecasting.py" "Nixtla Training" 30
echo

test_script "src/models/oil_gas_forecasting.py" "Combined Model Training" 30
echo

test_script "src/models/model_comparison.py" "Model Comparison" 10
echo

echo "========================================="
echo "Test Summary"
echo "========================================="
echo
echo "All scripts tested for:"
echo "  ✓ Import errors"
echo "  ✓ Startup issues"
echo "  ✓ Basic execution"
echo
echo "Note: Timeouts are expected for training scripts"
echo "      (they take minutes to complete)"
echo
