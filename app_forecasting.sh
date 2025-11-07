#!/bin/bash

# Oil & Gas Forecasting Dashboard Launcher for Domino
# This script launches the Streamlit forecasting dashboard

echo "Starting Oil & Gas Forecasting Dashboard..."
echo "Dashboard will be available shortly..."

# Set environment variables for better performance
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_PORT=${STREAMLIT_PORT:-8502}
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Navigate to the correct directory
cd /mnt/code

# Check if required files exist
if [ ! -f "scripts/forecasting_dashboard.py" ]; then
    echo "Error: Forecasting dashboard script not found!"
    exit 1
fi

if [ ! -f "scripts/data_config.py" ]; then
    echo "Error: Data config script not found!"
    exit 1
fi

echo "Files verified. Starting Streamlit server..."

# Start Streamlit with optimized settings
streamlit run scripts/forecasting_dashboard.py \
    --server.headless true \
    --server.port $STREAMLIT_SERVER_PORT \
    --server.address 0.0.0.0 \
    --server.baseUrlPath "" \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --server.maxUploadSize 50 \
    --server.maxMessageSize 50 \
    --browser.gatherUsageStats false