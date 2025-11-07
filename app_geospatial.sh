#!/bin/bash

# Oil & Gas Geospatial Dashboard Launcher for Domino
# This script launches the Streamlit geospatial dashboard

#!/bin/bash

mkdir -p ~/.streamlit

cat << EOF > ~/.streamlit/config.toml
[browser]
gatherUsageStats = true

[server]
port = 8888
enableCORS = false
enableXsrfProtection = false
address = "0.0.0.0"
EOF

streamlit run scripts/geospatial_dashboard.py