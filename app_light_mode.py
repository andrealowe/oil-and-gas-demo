#!/usr/bin/env python3
"""
Light Mode Streamlit App Launcher for Oil & Gas Dashboards
Ensures the apps always start in light mode regardless of system preferences
"""

import os
import sys
import subprocess

# Set environment variables to force light theme
os.environ['STREAMLIT_THEME_BACKGROUND_COLOR'] = '#ffffff'
os.environ['STREAMLIT_THEME_TEXT_COLOR'] = '#000000'
os.environ['STREAMLIT_THEME_PRIMARY_COLOR'] = '#3b82f6'
os.environ['STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR'] = '#f8f9fa'

def main():
    """Launch the forecasting dashboard with light mode enforced"""
    
    # Ensure we're in the right directory
    os.chdir('/mnt/code')
    
    # Check for dashboard argument
    dashboard = sys.argv[1] if len(sys.argv) > 1 else 'forecasting'
    
    if dashboard == 'geospatial':
        script = 'scripts/geospatial_dashboard.py'
        port = '8502'
        title = 'Geospatial'
    else:
        script = 'scripts/forecasting_dashboard.py'
        port = '8501'
        title = 'Forecasting'
    
    print(f"🚀 Starting Oil & Gas {title} Dashboard in Light Mode...")
    print(f"📊 Dashboard will be available at http://localhost:{port}")
    print("💡 Light mode is enforced - dashboard will stay light regardless of system theme")
    
    # Launch streamlit with light theme parameters
    cmd = [
        'streamlit', 'run', script,
        '--server.port', port,
        '--browser.gatherUsageStats', 'false',
        '--theme.backgroundColor', '#ffffff',
        '--theme.textColor', '#000000',
        '--theme.primaryColor', '#3b82f6',
        '--theme.secondaryBackgroundColor', '#f8f9fa'
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()