#!/bin/bash

mkdir -p ~/.streamlit

# Get Domino's base path for proper routing
DOMINO_BASE_PATH="${DOMINO_RUN_HOST_PATH:-}"

cat << EOF > ~/.streamlit/config.toml
[browser]
gatherUsageStats = false

[server]
port = 8888
enableCORS = false
enableXsrfProtection = false
address = "0.0.0.0"
baseUrlPath = "$DOMINO_BASE_PATH"
EOF

echo "Starting Streamlit app with base path: $DOMINO_BASE_PATH"
streamlit run scripts/forecasting_dashboard.py