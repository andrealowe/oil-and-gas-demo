#!/bin/bash

# Oil & Gas Dashboard Launcher Script
# Ensures light mode theme is enforced

# Set environment variables to force light theme
export STREAMLIT_THEME_BACKGROUND_COLOR="#ffffff"
export STREAMLIT_THEME_TEXT_COLOR="#000000"
export STREAMLIT_THEME_PRIMARY_COLOR="#3b82f6"

# Make sure we're in the right directory
cd /mnt/code

# Check which dashboard to run based on argument
if [ "$1" = "forecasting" ]; then
    echo "🚀 Starting Oil & Gas Forecasting Dashboard in Light Mode..."
    streamlit run scripts/forecasting_dashboard.py --server.port=8501 --browser.gatherUsageStats=false --theme.backgroundColor="#ffffff" --theme.textColor="#000000"
elif [ "$1" = "geospatial" ]; then
    echo "🚀 Starting Oil & Gas Geospatial Dashboard in Light Mode..."
    streamlit run scripts/geospatial_dashboard.py --server.port=8502 --browser.gatherUsageStats=false --theme.backgroundColor="#ffffff" --theme.textColor="#000000"
else
    echo "🚀 Starting Oil & Gas Forecasting Dashboard (default) in Light Mode..."
    streamlit run scripts/forecasting_dashboard.py --server.port=8501 --browser.gatherUsageStats=false --theme.backgroundColor="#ffffff" --theme.textColor="#000000"
fi