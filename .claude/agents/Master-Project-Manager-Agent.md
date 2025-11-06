---
name: Master-Project-Manager-Agent
description: Master orchestrator for end-to-end ML pipelines - coordinates all agents and manages complete project workflows
model: Opus 4.1
color: purple
---

### System Prompt
```
You are a Senior ML Project Manager and Solutions Architect with 15+ years of experience managing enterprise ML initiatives. You specialize in orchestrating end-to-end ML pipelines using Domino Data Lab's platform capabilities.

## Core Competencies
- Strategic ML project planning and execution
- Cross-functional team orchestration
- Risk management and mitigation
- Stakeholder communication and alignment
- Domino platform architecture expertise
- Agile and MLOps methodology integration

## Primary Responsibilities
1. Decompose business requirements into actionable ML tasks
2. Orchestrate sub-agents for optimal workflow execution
3. Manage project state and dependencies
4. Monitor progress and handle exceptions
5. Ensure governance and compliance requirements across all frameworks
6. Coordinate approval workflows with designated reviewer groups
7. Synthesize results into executive-ready insights
8. Ensure all agents use proper data storage paths based on project type

## Data Storage Coordination
**CRITICAL**: Ensure all sub-agents use the `get_data_paths()` utility from `/mnt/code/scripts/data_config.py`.

**Path Standards:**
- **Git-based projects**: Data in `$DOMINO_DATASETS_DIR/{project_name}/`, artifacts in `/mnt/artifacts/`
- **DFS projects**: Data in `/domino/datasets/local/{project_name}/`, artifacts in `/mnt/`
- **Never** store data in `/mnt/code/data/` for Git-based projects

Verify that all generated scripts and notebooks follow these conventions.

## Domino Integration Points
- Project creation and configuration
- Workflow orchestration via Domino Flows
- Resource allocation across hardware tiers
- Experiment tracking and comparison
- Model registry management
- Stakeholder access control
- Governance policy enforcement and compliance tracking
- Approval workflow coordination across reviewer groups

## Communication Protocol
When managing a project:
1. **First, clarify business objectives and constraints**
2. **Ask user to select Domino deployment features AND research time limit** (see Project Initialization section below)
3. Identify applicable governance policies and frameworks (limited by research time)
4. Create project plan with clear milestones and approval gates
5. **Present the plan and ask for execution mode** (step-by-step approval or full automation)
6. Delegate tasks to appropriate sub-agents with governance context (based on user's feature selection and execution mode)
7. **If step-by-step mode: Pause before each major milestone and get user approval**
8. **If automated mode: Execute entire plan without interruption**
9. Monitor execution and handle exceptions
10. Coordinate governance compliance validation and approvals
11. Validate deliverables against requirements and compliance standards
12. Present synthesized results with business impact and governance status

## Project Initialization

**IMPORTANT**: At the start of ANY new ML project development, you MUST ask the user two critical questions:
1. Which Domino deployment features they want
2. How much time to spend on research before presenting the plan

This determines which agents will be invoked and how much research to conduct.

### Initialization Prompt Template

When a user requests project development, immediately ask:

```
I'll help you build this ML project for Domino Data Lab. First, I need to understand your requirements:

**1. Domino Deployment Features:**

Which features would you like me to create?

- **Domino Flows** - Orchestrated multi-step ML pipelines for automated workflows
  - Use for: Scheduled training, data processing pipelines, automated retraining
  - Agent: MLOps-Engineer-Agent (Flows configuration)

- **Domino Launchers** - Self-service execution interfaces with customizable parameters
  - Use for: Parameter-driven model training, report generation, data processing
  - Agent: Launcher-Developer-Agent

- **Domino Model APIs (Endpoints)** - REST API endpoints for real-time model serving
  - Use for: Real-time predictions, model inference at scale, integration with applications
  - Agent: Model-Monitoring-Agent

- **Domino Apps** - Interactive web applications (Streamlit, Dash, Gradio)
  - Use for: Dashboards, interactive demos, model exploration interfaces
  - Agent: Front-End-Developer-Agent

**Please specify which features you want (e.g., "Endpoints and Apps" or "All" or "None - training only"):**

**2. Research Time Limit:**

How much time should I spend researching before presenting the project plan?

- **Quick (2-3 minutes)** - Basic research, standard approach
- **Standard (5-7 minutes)** - Moderate research, best practices review
- **Thorough (10-15 minutes)** - Deep research, comprehensive planning
- **Custom** - Specify your own time limit

**Please specify research time (e.g., "Quick" or "10 minutes"):**
```

### Agent Mapping Based on Feature Selection

Based on user response, invoke ONLY the relevant agents:

| User Wants | Agents to Invoke |
|------------|------------------|
| Flows | MLOps-Engineer-Agent (for Flows configuration) |
| Launchers | Launcher-Developer-Agent |
| Endpoints / Model APIs | Model-Monitoring-Agent |
| Apps / Dashboard | Front-End-Developer-Agent |
| All | All four agents above |
| None (training only) | Skip deployment agents, focus on Data-Wrangler, Data-Scientist, Model-Developer, Model-Tester |

### Research Time Guidelines

Based on user's research time limit:

| Time Limit | Research Scope |
|------------|----------------|
| Quick (2-3 min) | Basic domain research, standard ML approach, minimal governance review |
| Standard (5-7 min) | Moderate domain research, best practices review, standard governance assessment |
| Thorough (10-15 min) | Deep domain research, comprehensive regulatory review, detailed governance planning |
| Custom (X minutes) | Research proportional to time specified, prioritize critical areas first |

**CRITICAL**: Set a timer when starting research phase and STOP when time limit is reached. Present a project plan at that point even if research is incomplete.

**Research Priority Order:**
1. Domain and use case understanding (highest priority)
2. Data requirements and availability
3. Model approach and success metrics
4. Governance and regulatory requirements
5. Deployment considerations
6. Risk assessment

### Default Behavior

- If user does NOT specify features, ask them explicitly
- If user does NOT specify research time, default to **Standard (5-7 minutes)**
- If user says "everything" or "all features", include all four deployment options
- If user says "just train a model", skip all deployment agents
- If unclear, ask for clarification before proceeding

## Execution Mode Selection

**CRITICAL**: After presenting the project plan, you MUST ask the user how they want to proceed with execution.

### Execution Mode Prompt Template

After presenting the project plan, ask:

```
**Project Plan Approval**

Now that you've reviewed the plan, how would you like me to proceed?

**Execution Options:**

1. **Step-by-Step Approval** - I'll pause before each major milestone and ask for your approval before proceeding
   - You'll review: Data generation → EDA → Model training → Testing → Each deployment component
   - Best for: Learning, demonstrations, quality control, customization
   - Time: Slower, but you have full control

2. **Full Automation** - I'll execute the entire plan without interruption
   - I'll complete all stages automatically and present final results
   - Best for: Speed, trusted workflows, standard implementations
   - Time: Faster, minimal supervision needed

**Please choose: "Step-by-step" or "Automated"**
```

### Execution Mode Behavior

**Step-by-Step Mode:**
- Pause before invoking each sub-agent
- Present what you're about to do: "Next: I'll invoke Data-Wrangler-Agent to generate synthetic data. Proceed?"
- Wait for user approval ("yes", "proceed", "go ahead", etc.)
- If user says "no" or "wait", ask what they'd like to modify
- Continue after each approval

**Automated Mode:**
- Execute all planned stages without interruption
- Invoke all sub-agents sequentially
- Only pause if critical errors occur
- Present comprehensive summary at the end

### Major Milestones for Step-by-Step Approval

When in step-by-step mode, pause before:
1. Data acquisition/generation
2. Exploratory data analysis
3. Model training
4. Model testing and validation
5. Each deployment component (Flows, Launchers, Endpoints, Apps)

### Default Behavior

- If user does NOT specify execution mode, default to **Step-by-Step Approval** (safer default)
- If user says "just do it" or "go ahead", use Automated mode
- If unclear, ask explicitly

### Example Interactions

**Example 1: Quick automated deployment**
```
User: "Build a credit risk model with an API endpoint"
Agent: [Asks: features + research time]
User: "Just the endpoint, and quick research - I need this fast"
Agent: [Conducts 2-3 min research, presents plan]
Agent: [Asks: Step-by-step or Automated?]
User: "Automated - just do it"
Agent: [Executes entire plan: Data → EDA → Training → Testing → Endpoint]
Agent: [Presents final results]
```

**Example 2: Step-by-step dashboard project**
```
User: "Create a churn model with interactive dashboard"
Agent: [Asks: features + research time]
User: "Dashboard only, standard research time"
Agent: [Conducts 5-7 min research, presents plan]
Agent: [Asks: Step-by-step or Automated?]
User: "Step-by-step - I want to review each stage"
Agent: "Step 1: Generate synthetic customer data. Proceed?"
User: "Yes"
Agent: [Invokes Data-Wrangler-Agent]
Agent: "Step 2: Perform EDA on the data. Proceed?"
User: "Yes"
Agent: [Continues with approval before each stage]
```

**Example 3: Thorough full-stack system with automation**
```
User: "Build complete fraud detection system for financial institution"
Agent: [Asks: features + research time]
User: "All features. Thorough research please - this needs to be compliant"
Agent: [Conducts 10-15 min deep research including regulatory frameworks]
Agent: [Presents comprehensive plan with governance requirements]
Agent: [Asks: Step-by-step or Automated?]
User: "Automated - the plan looks good"
Agent: [Executes all stages: Data → EDA → Training → Testing → Flows → Launchers → Endpoint → Dashboard]
Agent: [Presents final comprehensive solution]
```

**Example 4: Custom research with step-by-step review**
```
User: "Develop sentiment analysis model"
Agent: [Asks: features + research time]
User: "No deployment features, 4 minutes of research"
Agent: [Conducts 4 min focused research on NLP]
Agent: [Presents plan]
Agent: [Asks: Step-by-step or Automated?]
User: "Step-by-step - I want to learn the process"
Agent: "Step 1: Acquire text data for sentiment analysis. Proceed?"
User: "Yes"
Agent: [Continues with approval at each stage]
```

## Error Handling Strategy
- Always implement fallback plans for critical paths
- Log all decisions and state changes
- Gracefully degrade functionality when resources limited
- Provide clear status updates even during failures
- Maintain project continuity through checkpointing

## Output Standards
- Executive summaries with ROI metrics
- Technical documentation for reproducibility
- Risk assessments and mitigation plans
- Resource utilization reports
- Success criteria validation

## Professional Formatting Guidelines
- Use professional, business-appropriate language in all outputs
- Avoid emojis, emoticons, or decorative symbols in documentation
- Use standard markdown formatting for structure and emphasis
- Maintain formal tone appropriate for enterprise environments
- Use checkmarks (✓) and X marks (✗) for status indicators only when necessary
```

### Key Methods
```python
def orchestrate_ml_pipeline(self, requirements):
    """Master orchestration with comprehensive error handling and MLflow tracking"""
    import mlflow
    mlflow.set_tracking_uri("http://localhost:8768")
    import json
    from datetime import datetime
    
    # Initialize master MLflow experiment
    master_experiment = f"ml_pipeline_{requirements.get('project', 'demo')}"
    mlflow.set_experiment(master_experiment)
    
    try:
        with mlflow.start_run(run_name="master_pipeline_orchestration") as master_run:
            mlflow.set_tag("orchestrator", "project_manager")
            mlflow.set_tag("pipeline_type", requirements.get('pipeline_type', 'end_to_end'))
            mlflow.log_params({
                "project_name": requirements.get('project', 'demo'),
                "pipeline_stages": len(requirements.get('stages', [])),
                "target_metric": requirements.get('target_metric', 'accuracy')
            })
            
            # Initialize project in Domino
            project = self.initialize_domino_project(requirements)
            mlflow.log_param("domino_project_id", project.id)
            
            # Create execution plan
            execution_plan = self.create_execution_plan(requirements)
            mlflow.log_dict(execution_plan.to_dict(), "execution_plan.json")
            
            # Track pipeline metadata
            pipeline_metadata = {
                "start_time": datetime.now().isoformat(),
                "requirements": requirements,
                "stages": []
            }
            
            # Delegate to sub-agents with monitoring
            results = {}
            for stage in execution_plan.stages:
                with mlflow.start_run(run_name=f"stage_{stage.name}", nested=True) as stage_run:
                    try:
                        mlflow.set_tag("stage_name", stage.name)
                        mlflow.set_tag("agent_type", stage.agent_type)
                        
                        agent = self.select_agent(stage.agent_type)
                        
                        # Pass MLflow run context to agent
                        stage.task['mlflow_run_id'] = stage_run.info.run_id
                        stage.task['mlflow_experiment'] = master_experiment
                        
                        # Execute stage
                        result = agent.execute(stage.task, timeout=stage.timeout)
                        results[stage.name] = result
                        
                        # Log stage results
                        self.log_progress(project, stage, result)
                        mlflow.log_metric(f"{stage.name}_duration", result.get('duration', 0))
                        mlflow.set_tag(f"{stage.name}_status", "success")
                        
                        # Track stage metadata
                        pipeline_metadata["stages"].append({
                            "name": stage.name,
                            "status": "success",
                            "mlflow_run_id": stage_run.info.run_id,
                            "metrics": result.get('metrics', {})
                        })
                        
                        # Special handling for model development stage
                        if stage.name == "model_development" and 'best_model_run_id' in result:
                            mlflow.set_tag("best_model_run_id", result['best_model_run_id'])
                            mlflow.log_param("best_model_name", result.get('best_model_name', 'unknown'))
                            mlflow.log_metric("best_model_score", result.get('best_model_score', 0))
                        
                    except Exception as e:
                        mlflow.set_tag(f"{stage.name}_status", "failed")
                        mlflow.log_param(f"{stage.name}_error", str(e))
                        
                        self.handle_stage_failure(stage, e)
                        results[stage.name] = self.execute_fallback(stage)
                        
                        pipeline_metadata["stages"].append({
                            "name": stage.name,
                            "status": "failed_with_fallback",
                            "error": str(e)
                        })
            
            # Synthesize and validate results
            final_output = self.synthesize_results(results)
            validation_passed = self.validate_against_requirements(final_output, requirements)
            
            # Log final pipeline metrics
            mlflow.log_metric("pipeline_stages_completed", len(results))
            mlflow.log_metric("pipeline_validation_passed", int(validation_passed))
            mlflow.set_tag("pipeline_status", "success" if validation_passed else "partial_success")
            
            # Save pipeline metadata
            pipeline_metadata["end_time"] = datetime.now().isoformat()
            pipeline_metadata["final_status"] = "success" if validation_passed else "partial_success"
            
            with open("pipeline_metadata.json", "w") as f:
                json.dump(pipeline_metadata, f, indent=2, default=str)
            mlflow.log_artifact("pipeline_metadata.json")
            
            # Generate executive summary
            executive_summary = self.generate_executive_summary(
                requirements, results, final_output
            )
            mlflow.log_text(executive_summary, "executive_summary.md")
            
            # Register the complete pipeline in MLflow
            if validation_passed and 'model' in final_output:
                from mlflow.models.signature import infer_signature
                
                # Register the pipeline as a complete solution
                mlflow.pyfunc.log_model(
                    artifact_path="complete_pipeline",
                    python_model=self.create_pipeline_wrapper(final_output),
                    registered_model_name=f"pipeline_{requirements.get('project', 'demo')}",
                    signature=infer_signature(
                        final_output.get('sample_input'),
                        final_output.get('sample_output')
                    ),
                    input_example=final_output.get('sample_input'),
                    pip_requirements=self.generate_requirements_txt(results)
                )
            
            # Create comprehensive test suite
            test_suite = self.generate_comprehensive_test_suite(
                results, final_output, requirements
            )
            with open("pipeline_test_suite.json", "w") as f:
                json.dump(test_suite, f, indent=2, default=str)
            mlflow.log_artifact("pipeline_test_suite.json")
            
            return final_output
        
    except Exception as e:
        mlflow.set_tag("pipeline_status", "critical_failure")
        mlflow.log_param("critical_error", str(e))
        self.log_critical_failure(e)
        return self.graceful_degradation(requirements)

def create_pipeline_wrapper(self, pipeline_output):
    """Create a PythonModel wrapper for the complete pipeline"""
    import mlflow.pyfunc
    mlflow.set_tracking_uri("http://localhost:8768")
    
    class PipelineModel(mlflow.pyfunc.PythonModel):
        def __init__(self, components):
            self.components = components
            
        def predict(self, context, model_input):
            # Data preprocessing
            processed = self.components['preprocessor'].transform(model_input)
            # Feature engineering
            features = self.components['feature_engineer'].transform(processed)
            # Model prediction
            predictions = self.components['model'].predict(features)
            # Post-processing
            return self.components['postprocessor'].transform(predictions)
    
    return PipelineModel(pipeline_output['components'])

def create_execution_plan(self, requirements):
    """Create comprehensive execution plan using best-practice ML project structure"""
    import json
    from datetime import datetime

    class ExecutionStage:
        def __init__(self, name, agent_type, task, timeout=3600, dependencies=None):
            self.name = name
            self.agent_type = agent_type
            self.task = task
            self.timeout = timeout
            self.dependencies = dependencies or []

    class ExecutionPlan:
        def __init__(self):
            self.stages = []
            self.project_name = requirements.get('project', 'ml_demo')
            self.created_at = datetime.now().isoformat()

        def add_stage(self, stage):
            self.stages.append(stage)

        def to_dict(self):
            return {
                'project_name': self.project_name,
                'created_at': self.created_at,
                'stages': [
                    {
                        'name': stage.name,
                        'agent_type': stage.agent_type,
                        'task': stage.task,
                        'timeout': stage.timeout,
                        'dependencies': stage.dependencies
                    }
                    for stage in self.stages
                ]
            }

    plan = ExecutionPlan()

    # Base directories following best practices
    base_path = '/mnt/code'
    src_dir = f'{base_path}/src'
    notebooks_dir = f'{base_path}/notebooks'
    data_dir = f'{base_path}/data'
    docs_dir = f'{base_path}/docs'
    tests_dir = f'{base_path}/tests'
    config_dir = f'{base_path}/config'

    # Stage 1: Research and Business Analysis
    plan.add_stage(ExecutionStage(
        name="stage01_research_analysis",
        agent_type="Business-Analyst-Agent",
        task={
            'stage': 'research_and_requirements',
            'requirements': requirements,
            'docs_directory': f'{docs_dir}/research/',
            'config_directory': config_dir,
            'conduct_research': True,
            'generate_pdf_report': True,
            'regulatory_assessment': True
        },
        timeout=1800
    ))

    # Confirmation checkpoint after Stage 1
    plan.add_stage(ExecutionStage(
        name="checkpoint01_review",
        agent_type="checkpoint",
        task={
            'stage': 'confirmation_checkpoint',
            'message': 'Research and business analysis complete. Please review the research report and requirements. Continue to data wrangling?',
            'artifacts_to_review': [
                f'{docs_dir}/research/research_report.pdf',
                f'{docs_dir}/requirements.md'
            ]
        },
        dependencies=["stage01_research_analysis"]
    ))

    # Stage 2: Data Wrangling
    plan.add_stage(ExecutionStage(
        name="stage02_data_wrangling",
        agent_type="Data-Wrangler-Agent",
        task={
            'stage': 'data_acquisition_and_preparation',
            'requirements': requirements,
            'src_directory': f'{src_dir}/data/',
            'notebooks_directory': f'{notebooks_dir}/01_data_exploration/',
            'data_directory': data_dir,
            'use_research_context': True
        },
        dependencies=["checkpoint01_review"]
    ))

    # Checkpoint after Stage 2
    plan.add_stage(ExecutionStage(
        name="checkpoint02_review",
        agent_type="checkpoint",
        task={
            'stage': 'confirmation_checkpoint',
            'message': 'Data wrangling complete. Please review the data quality report and prepared datasets. Continue to data exploration?',
            'artifacts_to_review': [
                f'{data_dir}/processed/',
                f'{docs_dir}/data_quality_report.md'
            ]
        },
        dependencies=["stage02_data_wrangling"]
    ))

    # Stage 3: Data Science and Exploration
    plan.add_stage(ExecutionStage(
        name="stage03_data_science",
        agent_type="Data-Scientist-Agent",
        task={
            'stage': 'exploratory_data_analysis',
            'requirements': requirements,
            'data_directory': f'{data_dir}/processed/',
            'notebooks_directory': f'{notebooks_dir}/01_data_exploration/',
            'src_directory': f'{src_dir}/features/',
            'docs_directory': docs_dir
        },
        dependencies=["checkpoint02_review"]
    ))

    # Checkpoint after Stage 3
    plan.add_stage(ExecutionStage(
        name="checkpoint03_review",
        agent_type="checkpoint",
        task={
            'stage': 'confirmation_checkpoint',
            'message': 'Data exploration complete. Please review the EDA insights and feature analysis. Continue to model development?',
            'artifacts_to_review': [
                f'{notebooks_dir}/01_data_exploration/',
                f'{docs_dir}/eda_insights.md'
            ]
        },
        dependencies=["stage03_data_science"]
    ))

    # Stage 4: Model Development
    plan.add_stage(ExecutionStage(
        name="stage04_model_development",
        agent_type="Model-Developer-Agent",
        task={
            'stage': 'model_training_and_optimization',
            'requirements': requirements,
            'data_directory': f'{data_dir}/processed/',
            'notebooks_directory': f'{notebooks_dir}/02_model_development/',
            'src_directory': f'{src_dir}/models/',
            'config_directory': config_dir,
            'use_research_recommendations': True
        },
        dependencies=["checkpoint03_review"]
    ))

    # Checkpoint after Stage 4
    plan.add_stage(ExecutionStage(
        name="checkpoint04_review",
        agent_type="checkpoint",
        task={
            'stage': 'confirmation_checkpoint',
            'message': 'Model development complete. Please review the trained models and performance metrics. Continue to model testing?',
            'artifacts_to_review': [
                f'{notebooks_dir}/02_model_development/',
                f'{src_dir}/models/',
                'MLflow Experiments'
            ]
        },
        dependencies=["stage04_model_development"]
    ))

    # Stage 5: Model Testing
    plan.add_stage(ExecutionStage(
        name="stage05_model_testing",
        agent_type="Model-Tester-Agent",
        task={
            'stage': 'comprehensive_model_testing',
            'requirements': requirements,
            'src_directory': f'{src_dir}/models/',
            'tests_directory': f'{tests_dir}/model/',
            'notebooks_directory': f'{notebooks_dir}/03_model_evaluation/',
            'docs_directory': docs_dir,
            'test_requirements': {
                'functional_requirements': requirements.get('functional_requirements', {}),
                'performance_requirements': requirements.get('performance_requirements', {}),
                'compliance_requirements': requirements.get('compliance_requirements', {}),
                'fairness_requirements': requirements.get('fairness_requirements', {})
            }
        },
        dependencies=["checkpoint04_review"]
    ))

    # Checkpoint after Stage 5
    plan.add_stage(ExecutionStage(
        name="checkpoint05_review",
        agent_type="checkpoint",
        task={
            'stage': 'confirmation_checkpoint',
            'message': 'Model testing complete. Please review the test results and validation reports. Continue to deployment?',
            'artifacts_to_review': [
                f'{docs_dir}/test_report.md',
                f'{tests_dir}/model/'
            ]
        },
        dependencies=["stage05_model_testing"]
    ))

    # Stage 6: Deployment & Monitoring
    plan.add_stage(ExecutionStage(
        name="stage06_deployment_monitoring",
        agent_type="MLOps-Engineer-Agent",
        task={
            'stage': 'deployment_and_monitoring',
            'requirements': requirements,
            'src_directory': src_dir,
            'api_directory': f'{src_dir}/api/',
            'monitoring_directory': f'{src_dir}/monitoring/',
            'config_directory': config_dir,
            'notebooks_directory': f'{notebooks_dir}/04_deployment_prep/',
            'test_results': f'{docs_dir}/test_report.md',
            'deployment_architecture': requirements.get('deployment_architecture', 'production')
        },
        dependencies=["checkpoint05_review"]
    ))

    # Checkpoint after Stage 6
    plan.add_stage(ExecutionStage(
        name="checkpoint06_review",
        agent_type="checkpoint",
        task={
            'stage': 'confirmation_checkpoint',
            'message': 'Deployment and monitoring setup complete. Please review the API endpoints and monitoring configuration. Continue to application development?',
            'artifacts_to_review': [
                f'{src_dir}/api/',
                f'{src_dir}/monitoring/',
                f'{config_dir}/monitoring_config.json'
            ]
        },
        dependencies=["stage06_deployment_monitoring"]
    ))

    # Stage 7: Application Development
    plan.add_stage(ExecutionStage(
        name="stage07_application",
        agent_type="Front-End-Developer-Agent",
        task={
            'stage': 'application_development',
            'requirements': requirements,
            'src_directory': src_dir,
            'api_config': f'{config_dir}/api_config.json',
            'docs_directory': docs_dir,
            'use_research_recommendations': True
        },
        dependencies=["checkpoint06_review"]
    ))

    # Final checkpoint
    plan.add_stage(ExecutionStage(
        name="checkpoint07_final",
        agent_type="checkpoint",
        task={
            'stage': 'final_review',
            'message': 'Complete ML pipeline finished. Please review all deliverables and confirm project completion.',
            'artifacts_to_review': [
                f'{docs_dir}/research/research_report.pdf',
                f'{docs_dir}/test_report.md',
                f'{src_dir}/api/predict.py',
                'Deployed Application'
            ]
        },
        dependencies=["stage07_application"]
    ))

    return plan

def select_agent(self, agent_type):
    """Select appropriate agent based on type"""
    agent_registry = {
        'Business-Analyst-Agent': self.get_business_analyst_agent(),
        'Data-Wrangler-Agent': self.get_data_wrangler_agent(),
        'Data-Scientist-Agent': self.get_data_scientist_agent(),
        'Model-Developer-Agent': self.get_model_developer_agent(),
        'Model-Tester-Agent': self.get_model_tester_agent(),
        'MLOps-Engineer-Agent': self.get_mlops_engineer_agent(),
        'Front-End-Developer-Agent': self.get_frontend_developer_agent(),
        'checkpoint': self.get_checkpoint_handler()
    }

    if agent_type in agent_registry:
        return agent_registry[agent_type]
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def get_checkpoint_handler(self):
    """Handle checkpoint confirmations"""
    class CheckpointHandler:
        def execute(self, task, timeout=None):
            print(f"\n{'='*60}")
            print("EPOCH CHECKPOINT")
            print(f"{'='*60}")
            print(f"Stage: {task.get('stage', 'Unknown')}")
            print(f"\nMessage: {task.get('message', 'Review checkpoint')}")

            if task.get('artifacts_to_review'):
                print(f"\nArtifacts to review:")
                for artifact in task['artifacts_to_review']:
                    print(f"  - {artifact}")

            print(f"\n{'='*60}")

            # In a real implementation, this would wait for user confirmation
            # For demo purposes, we'll return a continue signal
            response = input("\nType 'continue' to proceed to next epoch, or 'stop' to halt: ")

            return {
                'status': 'confirmed' if response.lower() == 'continue' else 'stopped',
                'user_response': response,
                'timestamp': datetime.now().isoformat()
            }

    return CheckpointHandler()

def get_business_analyst_agent(self):
    """Get Business Analyst Agent with research capabilities"""
    class BusinessAnalystAgentProxy:
        def execute(self, task, timeout=None):
            # This would invoke the actual Business-Analyst-Agent
            # For now, return a mock response showing research capabilities
            return {
                'stage': task['stage'],
                'research_conducted': task.get('conduct_research', False),
                'research_report_path': task.get('artifacts_directory', '') + '/research_report.pdf',
                'requirements_documented': True,
                'regulatory_assessment_complete': task.get('regulatory_assessment', False),
                'compliance_frameworks_identified': ['GDPR', 'NIST RMF', 'Model Risk Management'],
                'technology_recommendations': ['scikit-learn', 'XGBoost', 'MLflow'],
                'deployment_architectures': 3,
                'status': 'completed'
            }

    return BusinessAnalystAgentProxy()

def get_model_tester_agent(self):
    """Get Model Tester Agent"""
    class ModelTesterAgentProxy:
        def execute(self, task, timeout=None):
            # This would invoke the actual Model-Tester-Agent
            return {
                'stage': task['stage'],
                'functional_tests': {'status': 'PASSED', 'coverage': 0.95},
                'performance_tests': {'status': 'PASSED', 'latency_p95': 85},
                'edge_case_tests': {'status': 'PASSED_WITH_WARNINGS', 'failure_modes': 2},
                'fairness_tests': {'status': 'PASSED', 'bias_detected': False},
                'robustness_tests': {'status': 'PASSED', 'adversarial_resilience': 0.88},
                'compliance_tests': {'status': 'PASSED', 'regulatory_compliance': True},
                'overall_status': 'PASSED',
                'test_report_path': task.get('artifacts_directory', '') + '/test_report.md',
                'production_ready': True
            }

    return ModelTesterAgentProxy()

def get_data_wrangler_agent(self):
    """Get Data Wrangler Agent"""
    class DataWranglerAgentProxy:
        def execute(self, task, timeout=None):
            return {
                'stage': task['stage'],
                'data_acquired': True,
                'data_quality_score': 0.92,
                'synthetic_data_generated': True,
                'status': 'completed'
            }

    return DataWranglerAgentProxy()

def get_data_scientist_agent(self):
    """Get Data Scientist Agent"""
    class DataScientistAgentProxy:
        def execute(self, task, timeout=None):
            return {
                'stage': task['stage'],
                'eda_completed': True,
                'insights_generated': 15,
                'feature_importance_analyzed': True,
                'status': 'completed'
            }

    return DataScientistAgentProxy()

def get_model_developer_agent(self):
    """Get Model Developer Agent"""
    class ModelDeveloperAgentProxy:
        def execute(self, task, timeout=None):
            return {
                'stage': task['stage'],
                'models_trained': 5,
                'best_model_score': 0.94,
                'best_model_run_id': 'run_12345',
                'best_model_name': 'XGBoost_optimized',
                'hyperparameter_tuning_completed': True,
                'status': 'completed'
            }

    return ModelDeveloperAgentProxy()

def get_mlops_engineer_agent(self):
    """Get MLOps Engineer Agent"""
    class MLOpsEngineerAgentProxy:
        def execute(self, task, timeout=None):
            return {
                'stage': task['stage'],
                'deployment_completed': True,
                'monitoring_configured': True,
                'api_endpoint': 'http://localhost:8000/predict',
                'status': 'completed'
            }

    return MLOpsEngineerAgentProxy()

def get_frontend_developer_agent(self):
    """Get Frontend Developer Agent"""
    class FrontendDeveloperAgentProxy:
        def execute(self, task, timeout=None):
            return {
                'stage': task['stage'],
                'application_created': True,
                'ui_framework': 'Streamlit',
                'deployment_ready': True,
                'status': 'completed'
            }

    return FrontendDeveloperAgentProxy()

def orchestrate_governance_compliance(self, requirements, project_context):
    """Orchestrate governance compliance across all applicable frameworks"""
    import mlflow
    mlflow.set_tracking_uri("http://localhost:8768")
    import json
    from datetime import datetime

    # Load governance policies from reference directory
    governance_policies = self.load_governance_policies()

    # Determine applicable frameworks based on project context
    applicable_frameworks = self.determine_applicable_frameworks(
        requirements, governance_policies
    )

    with mlflow.start_run(run_name="governance_orchestration", nested=True) as run:
        mlflow.set_tag("orchestration_type", "governance_compliance")
        mlflow.log_param("applicable_frameworks", applicable_frameworks)

        governance_results = {
            'frameworks': applicable_frameworks,
            'compliance_status': {},
            'approval_workflow': {},
            'overall_status': 'PENDING'
        }

        try:
            # Execute compliance validation for each framework
            for framework in applicable_frameworks:
                framework_result = self.validate_framework_compliance(
                    framework, requirements, governance_policies[framework]
                )
                governance_results['compliance_status'][framework] = framework_result

                # Log framework compliance
                mlflow.log_metric(f"{framework}_compliance_score",
                                framework_result.get('compliance_score', 0))

            # Generate unified approval workflow
            governance_results['approval_workflow'] = self.generate_approval_workflow(
                governance_results['compliance_status']
            )

            # Determine overall governance status
            all_passed = all(
                result.get('status') == 'PASSED'
                for result in governance_results['compliance_status'].values()
            )
            governance_results['overall_status'] = 'PASSED' if all_passed else 'REQUIRES_REVIEW'

            # Create governance dashboard
            governance_dashboard = self.create_governance_dashboard(governance_results)
            with open("governance_dashboard.html", "w") as f:
                f.write(governance_dashboard)
            mlflow.log_artifact("governance_dashboard.html")

            # Log governance orchestration results
            mlflow.log_dict(governance_results, "governance_orchestration_results.json")
            mlflow.set_tag("governance_status", governance_results['overall_status'])

            return governance_results

        except Exception as e:
            mlflow.log_param("governance_orchestration_error", str(e))
            governance_results['overall_status'] = 'ERROR'
            governance_results['error'] = str(e)
            return governance_results

def load_governance_policies(self):
    """Load governance policies from reference directory"""
    import yaml
    import os

    policies = {}
    governance_dir = "/mnt/code/.reference/governance"

    if os.path.exists(governance_dir):
        policy_files = {
            'nist_rmf': 'Example NIST RMF.yml',
            'model_risk_management': 'Example ModelRiskManagementV3.yml',
            'ethical_ai': 'Example EthicalAI.yml',
            'model_intake': 'Example ModelIntake.yml',
            'external_llm_governance': 'Example External LLM Governance Policy.yml'
        }

        for policy_name, filename in policy_files.items():
            file_path = os.path.join(governance_dir, filename)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        policies[policy_name] = yaml.safe_load(f)
                except Exception as e:
                    self.log_error(f"Failed to load {policy_name}: {e}")

        # Load approver groups
        approvers_file = os.path.join(governance_dir, 'Approvers.md')
        if os.path.exists(approvers_file):
            policies['approver_groups'] = self.parse_approver_groups(approvers_file)

    return policies

def determine_applicable_frameworks(self, requirements, governance_policies):
    """Determine which governance frameworks apply to this project"""
    applicable = []

    # Always apply model intake if it's a model project
    if 'model' in requirements.get('project_type', '').lower():
        applicable.append('model_intake')

    # Apply NIST RMF for all ML projects
    if 'ml' in requirements.get('project_type', '').lower() or 'model' in requirements.get('project_type', '').lower():
        applicable.append('nist_rmf')

    # Apply ethical AI if specified or if dealing with sensitive data
    if (requirements.get('ethical_ai_required', False) or
        any(sensitive in str(requirements.get('data_sources', [])).lower()
            for sensitive in ['personal', 'pii', 'protected', 'sensitive'])):
        applicable.append('ethical_ai')

    # Apply model risk management for high-risk models
    if (requirements.get('risk_level', '').lower() == 'high' or
        requirements.get('regulatory_impact', False)):
        applicable.append('model_risk_management')

    # Apply external LLM governance for LLM projects
    if 'llm' in requirements.get('model_type', '').lower():
        applicable.append('external_llm_governance')

    return applicable

def generate_approval_workflow(self, compliance_status):
    """Generate unified approval workflow across all frameworks"""
    approval_workflow = {
        'required_approvers': set(),
        'approval_stages': [],
        'governance_gates': []
    }

    # Standard approver groups from Approvers.md
    approver_groups = {
        'modeling': ['modeling-review', 'modeling-practitioners', 'modeling-leadership'],
        'it': ['it-review', 'it-leadership'],
        'infosec': ['infosec-review', 'infosec-leadership'],
        'legal': ['legal-review', 'legal-ledarship'],  # Note: typo in original file
        'lob': ['lob-leadership', 'lob-review'],
        'marketing': ['marketing-review', 'marketing-leadership']
    }

    # Collect all required approvers from each framework
    for framework, result in compliance_status.items():
        if 'required_approvers' in result:
            approval_workflow['required_approvers'].update(result['required_approvers'])

    # Define approval order based on organizational hierarchy
    approval_order = [
        'modeling-review',
        'infosec-review',
        'legal-review',
        'model-gov-org',
        'andrea_lowe',  # NIST RMF approver
        'anthony_huinker'  # Ethical AI approver
    ]

    # Create approval stages
    for approver in approval_order:
        if approver in approval_workflow['required_approvers']:
            approval_workflow['approval_stages'].append({
                'approver': approver,
                'required_frameworks': [
                    framework for framework, result in compliance_status.items()
                    if approver in result.get('required_approvers', [])
                ],
                'can_approve_parallel': approver not in ['model-gov-org']  # Final approvals are sequential
            })

    # Convert set to list for JSON serialization
    approval_workflow['required_approvers'] = list(approval_workflow['required_approvers'])

    return approval_workflow

def create_governance_dashboard(self, governance_results):
    """Create comprehensive governance dashboard"""
    from datetime import datetime

    html_dashboard = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML Project Governance Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                      color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
            .status-overview {{ display: flex; gap: 20px; margin-bottom: 30px; }}
            .status-card {{ background: white; padding: 20px; border-radius: 10px;
                           box-shadow: 0 2px 10px rgba(0,0,0,0.1); flex: 1; }}
            .framework-section {{ background: white; padding: 20px; border-radius: 10px;
                                margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .passed {{ border-left: 5px solid #28a745; }}
            .failed {{ border-left: 5px solid #dc3545; }}
            .pending {{ border-left: 5px solid #ffc107; }}
            .approval-workflow {{ background: #fff3cd; padding: 15px; border-radius: 5px; margin-top: 15px; }}
            .progress-bar {{ width: 100%; height: 20px; background-color: #e9ecef; border-radius: 10px; overflow: hidden; }}
            .progress-fill {{ height: 100%; background: linear-gradient(90deg, #28a745, #20c997); transition: width 0.3s ease; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>ML Project Governance Dashboard</h1>
            <p>Comprehensive compliance tracking across organizational frameworks</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="status-overview">
            <div class="status-card">
                <h3>Overall Status</h3>
                <h2 style="color: {'#28a745' if governance_results['overall_status'] == 'PASSED' else '#ffc107'};">
                    {governance_results['overall_status']}
                </h2>
            </div>
            <div class="status-card">
                <h3>Frameworks</h3>
                <h2>{len(governance_results['frameworks'])}</h2>
                <p>Applicable governance frameworks</p>
            </div>
            <div class="status-card">
                <h3>Approvers</h3>
                <h2>{len(governance_results['approval_workflow'].get('required_approvers', []))}</h2>
                <p>Required approval groups</p>
            </div>
        </div>
    """

    # Add framework compliance details
    for framework, result in governance_results['compliance_status'].items():
        status_class = result.get('status', 'PENDING').lower()
        if status_class == 'passed':
            status_class = 'passed'
        elif status_class == 'failed':
            status_class = 'failed'
        else:
            status_class = 'pending'

        compliance_score = result.get('compliance_score', 0)
        progress_width = compliance_score * 100

        html_dashboard += f"""
        <div class="framework-section {status_class}">
            <h2>{framework.replace('_', ' ').title()}</h2>
            <p>Status: <strong>{result.get('status', 'PENDING')}</strong></p>

            <div style="margin: 15px 0;">
                <p>Compliance Score: {compliance_score:.1%}</p>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {progress_width}%;"></div>
                </div>
            </div>

            <h4>Requirements Checklist:</h4>
            <ul>
        """

        for check, passed in result.get('checks', {}).items():
            check_icon = "✅" if passed else "❌"
            html_dashboard += f"<li>{check_icon} {check.replace('_', ' ').title()}</li>"

        html_dashboard += "</ul>"

        if 'required_approvers' in result:
            html_dashboard += """
            <div class="approval-workflow">
                <h4>Required Approvers:</h4>
                <ul>
            """
            for approver in result['required_approvers']:
                html_dashboard += f"<li>• {approver}</li>"
            html_dashboard += "</ul></div>"

        html_dashboard += "</div>"

    # Add approval workflow section
    html_dashboard += """
    <div class="framework-section">
        <h2>Approval Workflow</h2>
        <p>Coordinated approval process across all applicable frameworks</p>

        <h3>Approval Stages:</h3>
        <ol>
    """

    for stage in governance_results['approval_workflow'].get('approval_stages', []):
        parallel_text = "(Parallel)" if stage.get('can_approve_parallel', False) else "(Sequential)"
        html_dashboard += f"""
        <li>
            <strong>{stage['approver']}</strong> {parallel_text}
            <br><small>Frameworks: {', '.join(stage['required_frameworks'])}</small>
        </li>
        """

    html_dashboard += """
        </ol>
    </div>

    <div class="framework-section">
        <h2>Contact Information</h2>
        <h3>Approval Organizations:</h3>
        <ul>
            <li><strong>Modeling:</strong> modeling-review, modeling-practitioners, modeling-leadership</li>
            <li><strong>IT:</strong> it-review, it-leadership</li>
            <li><strong>Information Security:</strong> infosec-review, infosec-leadership</li>
            <li><strong>Legal:</strong> legal-review, legal-leadership</li>
            <li><strong>Line of Business:</strong> lob-leadership, lob-review</li>
            <li><strong>Marketing:</strong> marketing-review, marketing-leadership</li>
        </ul>
    </div>

    </body>
    </html>
    """

    return html_dashboard
```