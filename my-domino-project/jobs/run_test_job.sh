#!/bin/bash

# Activate the virtual environment if needed
# source /path/to/your/venv/bin/activate

# Navigate to the source directory
cd ../src

# Run the test script
python test_script.py

# Check the exit status of the test script
if [ $? -eq 0 ]; then
    echo "Test script executed successfully."
else
    echo "Test script failed."
    exit 1
fi