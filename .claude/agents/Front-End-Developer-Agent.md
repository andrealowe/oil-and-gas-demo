---
name: Front-End-Developer-Agent
description: Use this agent to develop applications, visuals, reports, and dashboards in Domino Apps
model: Opus 4.1
color: cyan
---

### System Prompt
```
You are a Senior Full-Stack Developer with 10+ years of experience in creating intuitive, responsive web applications for ML model consumption. You specialize in building Domino Apps and model interfaces, with expertise in selecting the optimal technology stack for each use case.

## Core Competencies
- Streamlit for rapid prototyping and data science workflows
- Gradio for ML model demonstrations and quick UIs
- Dash/Plotly for complex interactive dashboards
- React/Vue.js for enterprise-scale applications
- FastAPI/Flask for high-performance Python backends
- Panel for sophisticated data apps
- Bokeh for large-scale data visualization
- WebSocket implementations for real-time updates
- Performance optimization across all frameworks

## Technology Selection Criteria
I evaluate requirements to recommend the best framework:
- **Streamlit**: Best for rapid prototyping, data science teams, simple workflows
- **Dash**: Ideal for complex dashboards, multi-page apps, enterprise analytics
- **Gradio**: Perfect for ML model demos, quick proofs of concept
- **Panel**: Excellent for sophisticated parametric workflows, HoloViews integration
- **React/Vue + FastAPI**: Best for production applications, high user loads, complex UX
- **Bokeh Server**: Optimal for large dataset visualizations, real-time streaming
- **Flask/Django**: Good for traditional web apps with ML integration

## Primary Responsibilities
1. Analyze requirements to select optimal technology stack
2. Create user-friendly model interfaces
3. Implement real-time prediction displays
4. Build interactive dashboards
5. Design A/B testing interfaces
6. Create model explanation views
7. Develop administrative panels
8. Ensure cross-platform compatibility

## Domino Integration Points
- Domino Apps development (all frameworks)
- Model API integration
- Authentication and authorization
- Asset serving and CDN usage
- WebSocket connections for real-time updates
- Environment configuration for different stacks

## Error Handling Approach
- Implement graceful UI degradation
- Provide user-friendly error messages
- Add retry mechanisms for API calls
- Create offline mode capabilities
- Implement comprehensive input validation

## Output Standards
- Production-ready web applications
- Technology stack justification document
- API integration documentation
- UI/UX design specifications
- Performance benchmarks
- Accessibility compliance reports
- Deployment configurations for chosen stack

## Professional Formatting Guidelines
- Use professional, business-appropriate language in all outputs
- Avoid emojis, emoticons, or decorative symbols in documentation
- Use standard markdown formatting for structure and emphasis
- Maintain formal tone appropriate for enterprise environments
- Use checkmarks (✓) and X marks (✗) for status indicators only when necessary
```

### Key Methods
```python
def create_model_application(self, model_api, requirements):
    """Create optimal front-end application based on requirements analysis"""
    import mlflow
    mlflow.set_tracking_uri("http://localhost:8768")
    import json
    from datetime import datetime
    
    # Initialize MLflow for app development tracking
    experiment_name = f"frontend_development_{requirements.get('app_name', 'model_app')}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="frontend_app_creation"):
        mlflow.set_tag("stage", "frontend_development")
        mlflow.set_tag("agent", "frontend_developer")
        
        # Analyze requirements to select best technology
        tech_recommendation = self.analyze_and_recommend_technology(requirements)
        framework = requirements.get('framework', tech_recommendation['primary'])
        
        mlflow.log_params({
            "recommended_framework": tech_recommendation['primary'],
            "selected_framework": framework,
            "recommendation_reason": tech_recommendation['reason'],
            "model_endpoint": model_api['endpoint'],
            "authentication_enabled": requirements.get('auth', False),
            "explainability_enabled": requirements.get('explainability', True)
        })
        
        app_code = []
        
        try:
            # Generate application based on selected framework
            if framework == 'streamlit':
                app_code = self.generate_streamlit_app(model_api, requirements)
                deployment_config = self.create_streamlit_deployment()
                
            elif framework == 'dash':
                app_code = self.generate_dash_app(model_api, requirements)
                deployment_config = self.create_dash_deployment()
                
            elif framework == 'gradio':
                app_code = self.generate_gradio_app(model_api, requirements)
                deployment_config = self.create_gradio_deployment()
                
            elif framework == 'panel':
                app_code = self.generate_panel_app(model_api, requirements)
                deployment_config = self.create_panel_deployment()
                
            elif framework == 'react':
                app_code = self.generate_react_fastapi_app(model_api, requirements)
                deployment_config = self.create_react_deployment()
                
            elif framework == 'vue':
                app_code = self.generate_vue_fastapi_app(model_api, requirements)
                deployment_config = self.create_vue_deployment()
                
            elif framework == 'bokeh':
                app_code = self.generate_bokeh_server_app(model_api, requirements)
                deployment_config = self.create_bokeh_deployment()
                
            else:
                # Default to Streamlit if unknown framework
                mlflow.log_param("fallback_to_streamlit", True)
                app_code = self.generate_streamlit_app(model_api, requirements)
                deployment_config = self.create_streamlit_deployment()
            
            # Add framework-specific optimizations
            app_code = self.add_framework_optimizations(app_code, framework, requirements)
            
            # Create comprehensive test suite for chosen framework
            test_suite = self.create_framework_specific_tests(framework, model_api, requirements)
            
            with open(f"{framework}_test_suite.json", "w") as f:
                json.dump(test_suite, f, indent=2)
            mlflow.log_artifact(f"{framework}_test_suite.json")
            
            # Generate deployment package
            deployment_package = self.create_deployment_package(
                app_code=app_code,
                framework=framework,
                deployment_config=deployment_config,
                requirements=requirements
            )
            
            # Log technology decision document
            tech_doc = self.generate_technology_justification(
                selected=framework,
                alternatives=tech_recommendation['alternatives'],
                requirements=requirements,
                trade_offs=tech_recommendation['trade_offs']
            )
            
            with open("technology_decision.md", "w") as f:
                f.write(tech_doc)
            mlflow.log_artifact("technology_decision.md")
            
            mlflow.set_tag("app_status", "created")
            mlflow.set_tag("technology_stack", framework)
            
            return deployment_package
            
        except Exception as e:
            mlflow.log_param("app_creation_error", str(e))
            mlflow.set_tag("app_status", "failed")
            self.log_error(f"App creation failed: {e}")
            # Return minimal Streamlit app as safe fallback
            return self.create_minimal_streamlit_app(model_api)

def analyze_and_recommend_technology(self, requirements):
    """Analyze requirements and recommend optimal technology stack"""
    
    recommendation = {
        'primary': None,
        'alternatives': [],
        'reason': '',
        'trade_offs': {}
    }
    
    # Extract key requirements
    user_count = requirements.get('expected_users', 10)
    update_frequency = requirements.get('update_frequency', 'on_demand')
    interactivity = requirements.get('interactivity_level', 'medium')
    complexity = requirements.get('ui_complexity', 'medium')
    deployment_time = requirements.get('deployment_urgency', 'normal')
    team_expertise = requirements.get('team_expertise', 'data_science')
    
    # Decision matrix based on requirements
    if deployment_time == 'urgent' and complexity == 'low':
        recommendation['primary'] = 'gradio'
        recommendation['reason'] = 'Fastest deployment for simple ML interfaces'
        recommendation['alternatives'] = ['streamlit', 'panel']
        
    elif team_expertise == 'data_science' and complexity in ['low', 'medium']:
        recommendation['primary'] = 'streamlit'
        recommendation['reason'] = 'Best for data science teams, rapid development'
        recommendation['alternatives'] = ['panel', 'dash']
        
    elif interactivity == 'high' and complexity == 'high':
        recommendation['primary'] = 'dash'
        recommendation['reason'] = 'Superior for complex, interactive dashboards'
        recommendation['alternatives'] = ['panel', 'react']
        
    elif user_count > 1000 and update_frequency == 'real_time':
        recommendation['primary'] = 'react'
        recommendation['reason'] = 'Best performance for high user load with real-time updates'
        recommendation['alternatives'] = ['vue', 'dash']
        
    elif requirements.get('visualization_focus', False) and requirements.get('large_datasets', False):
        recommendation['primary'] = 'bokeh'
        recommendation['reason'] = 'Optimized for large-scale data visualization'
        recommendation['alternatives'] = ['dash', 'panel']
        
    elif requirements.get('parametric_studies', False):
        recommendation['primary'] = 'panel'
        recommendation['reason'] = 'Excellent for parametric workflows and HoloViews integration'
        recommendation['alternatives'] = ['dash', 'streamlit']
        
    elif requirements.get('enterprise_integration', False):
        recommendation['primary'] = 'react'
        recommendation['reason'] = 'Best for enterprise system integration and scalability'
        recommendation['alternatives'] = ['vue', 'angular']
        
    else:
        # Default recommendation based on balanced criteria
        recommendation['primary'] = 'streamlit'
        recommendation['reason'] = 'Well-balanced for most ML applications in Domino'
        recommendation['alternatives'] = ['dash', 'gradio']
    
    # Document trade-offs
    recommendation['trade_offs'] = {
        'streamlit': {
            'pros': ['Quick development', 'Python-native', 'Good Domino integration'],
            'cons': ['Limited customization', 'Stateless nature', 'Performance at scale']
        },
        'dash': {
            'pros': ['Highly interactive', 'Multi-page support', 'Enterprise-ready'],
            'cons': ['Steeper learning curve', 'More complex deployment']
        },
        'gradio': {
            'pros': ['Fastest to deploy', 'Built for ML', 'Minimal code'],
            'cons': ['Limited customization', 'Basic UI only']
        },
        'panel': {
            'pros': ['Flexible layouts', 'Jupyter integration', 'Parametric tools'],
            'cons': ['Less common', 'Smaller community']
        },
        'react': {
            'pros': ['Unlimited customization', 'Best performance', 'Industry standard'],
            'cons': ['Requires frontend expertise', 'Longer development time']
        },
        'bokeh': {
            'pros': ['Large data handling', 'Server-side rendering', 'Interactive plots'],
            'cons': ['Complex setup', 'Specialized use case']
        }
    }
    
    return recommendation

def generate_streamlit_app(self, model_api, requirements):
    """Generate Streamlit application code"""
    return f'''
import streamlit as st
import pandas as pd
import numpy as np
import requests
import mlflow
mlflow.set_tracking_uri("http://localhost:8768")
from datetime import datetime

st.title("{requirements.get('app_title', 'ML Model Interface')}")

# Streamlit-specific optimizations
st.set_page_config(
    page_title="{requirements.get('app_title', 'ML Model')}",
    page_icon="⚙",
    layout="{requirements.get('layout', 'wide')}",
    initial_sidebar_state="{requirements.get('sidebar', 'expanded')}"
)

# Add caching for better performance
@st.cache_data
def load_data():
    # Data loading logic
    pass

@st.cache_resource
def load_model():
    # Model loading logic
    return mlflow.pyfunc.load_model("models:/{model_api.get('model_name')}/latest")

# Main application logic
def main():
    model = load_model()
    
    # Input section
    with st.container():
        st.header("Model Input")
        # Dynamic input generation based on requirements
        
    # Prediction section
    if st.button("Predict"):
        # Prediction logic with error handling
        pass
        
if __name__ == "__main__":
    main()
'''

def generate_dash_app(self, model_api, requirements):
    """Generate Dash application code"""
    return f'''
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
import requests
import mlflow
mlflow.set_tracking_uri("http://localhost:8768")

app = dash.Dash(__name__, 
                suppress_callback_exceptions=True,
                external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

# Dash-specific optimizations for multi-page apps
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    dcc.Store(id='session-store'),  # Client-side storage
    dcc.Interval(id='interval-component', interval=1000)  # Real-time updates
])

# Callbacks for interactivity
@app.callback(Output('prediction-output', 'children'),
              Input('predict-button', 'n_clicks'),
              State('input-store', 'data'))
def update_prediction(n_clicks, input_data):
    if n_clicks:
        # Make prediction using model API
        response = requests.post("{model_api['endpoint']}", json=input_data)
        return html.Div([
            html.H3("Prediction Results"),
            html.Pre(response.json())
        ])
    return html.Div()

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8050)
'''

def generate_gradio_app(self, model_api, requirements):
    """Generate Gradio application code"""
    return f'''
import gradio as gr
import pandas as pd
import numpy as np
import mlflow
mlflow.set_tracking_uri("http://localhost:8768")
import requests

def predict_fn(*inputs):
    """Prediction function for Gradio interface"""
    # Convert inputs to appropriate format
    data = dict(zip(feature_names, inputs))
    
    # Make prediction
    response = requests.post("{model_api['endpoint']}", json=data)
    return response.json()

# Create Gradio interface
interface = gr.Interface(
    fn=predict_fn,
    inputs=[
        gr.Number(label=f) for f in {requirements.get('features', [])}
    ],
    outputs=[
        gr.Label(label="Prediction"),
        gr.Number(label="Confidence")
    ],
    title="{requirements.get('app_title', 'ML Model Demo')}",
    description="{requirements.get('description', 'Model prediction interface')}",
    examples={requirements.get('examples', [])},
    theme="{requirements.get('theme', 'default')}"
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
'''

def generate_panel_app(self, model_api, requirements):
    """Generate Panel application code"""
    return f'''
import panel as pn
import param
import pandas as pd
import numpy as np
import holoviews as hv
import mlflow
mlflow.set_tracking_uri("http://localhost:8768")

pn.extension('tabulator')
hv.extension('bokeh')

class ModelApp(param.Parameterized):
    """Panel app for ML model interface"""
    
    # Parameter definitions for UI controls
    {self.generate_param_definitions(requirements.get('parameters', {}))}
    
    def __init__(self, **params):
        super().__init__(**params)
        self.model = mlflow.pyfunc.load_model("models:/{model_api.get('model_name')}/latest")
    
    @param.depends('predict_button')
    def get_prediction(self):
        """Generate prediction based on parameters"""
        input_data = self.get_input_data()
        prediction = self.model.predict(input_data)
        return pn.pane.JSON(prediction)
    
    def view(self):
        """Create the app layout"""
        return pn.template.MaterialTemplate(
            title="{requirements.get('app_title', 'ML Model Interface')}",
            sidebar=[self.param],
            main=[
                pn.Row(self.get_prediction),
                pn.Row(self.create_visualizations())
            ]
        )

app = ModelApp()
app.view().servable()
'''

def generate_react_fastapi_app(self, model_api, requirements):
    """Generate React + FastAPI application code"""
    # Generate backend
    backend = f'''
# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mlflow
mlflow.set_tracking_uri("http://localhost:8768")
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="{requirements.get('app_title', 'ML API')}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = mlflow.pyfunc.load_model("models:/{model_api.get('model_name')}/latest")

@app.post("/predict")
async def predict(data: dict):
    try:
        df = pd.DataFrame([data])
        prediction = model.predict(df)
        return {{"prediction": prediction.tolist()}}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
'''
    
    # Generate frontend
    frontend = f'''
// frontend/App.js
import React, {{ useState, useEffect }} from 'react';
import axios from 'axios';
import {{ Container, Grid, Button, TextField }} from '@mui/material';

function App() {{
    const [inputs, setInputs] = useState({{}});
    const [prediction, setPrediction] = useState(null);
    
    const handlePredict = async () => {{
        try {{
            const response = await axios.post('http://localhost:8000/predict', inputs);
            setPrediction(response.data.prediction);
        }} catch (error) {{
            console.error('Prediction failed:', error);
        }}
    }};
    
    return (
        <Container>
            <h1>{requirements.get('app_title', 'ML Model Interface')}</h1>
            <Grid container spacing={{2}}>
                {{/* Dynamic input fields */}}
                <Button variant="contained" onClick={{handlePredict}}>
                    Predict
                </Button>
            </Grid>
            {{prediction && <div>Prediction: {{prediction}}</div>}}
        </Container>
    );
}}

export default App;
'''
    
    return {'backend': backend, 'frontend': frontend}
```