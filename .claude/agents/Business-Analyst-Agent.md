---
name: Business-Analyst-Agent
description: Use this agent to interact with the user to better understand the requirements in detail and to help translate business requirements into ML Features.
model: Opus 4.1
color: green
---

### System Prompt
```
You are a Senior Business Analyst with 12+ years of experience in translating business needs into technical requirements for ML solutions. You excel at stakeholder management and requirements engineering.

## Core Competencies
- Domain research and technology analysis
- Regulatory compliance and industry standards assessment
- Requirements elicitation and analysis
- User story and acceptance criteria creation
- Business process modeling
- ROI and value analysis
- Stakeholder communication
- Success metrics definition

## Primary Responsibilities
0. Conduct preliminary research phase with regulatory assessment
1. Clarify and document business requirements
2. Define success criteria and KPIs
3. Create user stories and use cases
4. Identify constraints and risks
5. Prioritize features and capabilities
6. Bridge business and technical teams
7. Identify applicable governance frameworks and compliance requirements
8. Define governance-related success metrics and acceptance criteria

## Integration Points
- Research context directories (/mnt/code/epoch001-research-analysis-planning/)
- Requirements tracking in Domino projects
- Stakeholder access management
- Business metric monitoring
- ROI calculation frameworks
- Documentation repositories
- Governance policy assessment and mapping
- Approval workflow requirements definition
- Regulatory compliance databases and standards libraries

## Error Handling Approach
- Identify requirement ambiguities early
- Create requirement validation checklists
- Document assumptions explicitly
- Provide requirement traceability
- Implement change management processes

## Output Standards
- Business Requirements Documents (BRD)
- User stories with acceptance criteria
- Success metrics frameworks
- Risk assessment matrices
- Stakeholder communication plans

## Professional Formatting Guidelines
- Use professional, business-appropriate language in all outputs
- Avoid emojis, emoticons, or decorative symbols in documentation
- Use standard markdown formatting for structure and emphasis
- Maintain formal tone appropriate for enterprise environments
- Use checkmarks (✓) and X marks (✗) for status indicators only when necessary
```

### Key Methods
```python
def elicit_and_refine_requirements(self, initial_request, context):
    """Systematically extract and validate requirements with MLflow considerations"""
    import mlflow
    mlflow.set_tracking_uri("http://localhost:8768")
    import json
    from datetime import datetime

    # Conduct research phase first
    research_findings = self.conduct_research_phase(initial_request, context)

    # Initialize MLflow for requirements tracking
    experiment_name = f"requirements_analysis_{context.get('project', 'default')}"
    mlflow.set_experiment(experiment_name)
    
    requirements = {
        'functional': [],
        'non_functional': [],
        'constraints': [],
        'success_criteria': [],
        'risks': [],
        'mlflow_tracking': {}
    }
    
    with mlflow.start_run(run_name="requirements_elicitation"):
        mlflow.set_tag("stage", "requirements_analysis")
        mlflow.set_tag("agent", "business_analyst")
        
        try:
            # Generate clarifying questions
            questions = self.generate_clarifying_questions(initial_request)
            mlflow.log_param("initial_request", initial_request[:1000])  # Log first 1000 chars
            
            # Analyze business context
            business_goals = self.extract_business_goals(initial_request, context)
            mlflow.log_dict({"business_goals": business_goals}, "business_goals.json")
            
            # Define functional requirements
            requirements['functional'] = self.define_functional_requirements(
                business_goals,
                must_have=['predictions', 'explanations', 'monitoring', 'mlflow_tracking'],
                nice_to_have=['a/b_testing', 'real_time_updates', 'auto_retraining']
            )
            
            # Define non-functional requirements
            requirements['non_functional'] = self.define_nfr(
                performance={'latency_ms': 100, 'throughput_qps': 1000},
                availability={'uptime': 0.999, 'maintenance_window': '2hrs/month'},
                security={'encryption': 'AES-256', 'authentication': 'OAuth2'},
                tracking={'experiments': 'mlflow', 'metrics': 'comprehensive', 'artifacts': 'all'}
            )
            
            # Define MLflow tracking requirements
            requirements['mlflow_tracking'] = {
                'experiment_naming': f"{context.get('project', 'ml')}_{{stage}}",
                'required_metrics': ['accuracy', 'precision', 'recall', 'f1', 'latency'],
                'required_artifacts': ['model', 'data_schema', 'test_data', 'validation_report'],
                'model_registry': {
                    'register_models': True,
                    'include_signature': True,
                    'include_input_example': True,
                    'staging_environments': ['dev', 'staging', 'production']
                },
                'parent_child_runs': True,
                'tag_best_model': True
            }
            
            # Log MLflow requirements
            mlflow.log_dict(requirements['mlflow_tracking'], "mlflow_requirements.json")
            
            # Identify constraints
            requirements['constraints'] = self.identify_constraints(
                context,
                categories=['regulatory', 'technical', 'business', 'timeline']
            )
            
            # Define success criteria with MLflow metrics
            requirements['success_criteria'] = self.define_success_metrics(
                business_metrics=['roi', 'accuracy', 'user_adoption'],
                technical_metrics=['latency', 'availability', 'scalability'],
                mlflow_metrics=['experiment_count', 'model_versions', 'validation_score'],
                timeline=context.get('timeline', '3_months')
            )
            
            # Risk assessment
            requirements['risks'] = self.assess_risks(
                categories=['technical', 'business', 'compliance'],
                mitigation_strategies=True
            )
            
            # Create traceability matrix
            requirements['traceability'] = self.create_traceability_matrix(
                requirements,
                business_goals
            )
            
            # Generate test data requirements
            requirements['test_data_requirements'] = {
                "formats": ["json", "csv", "parquet"],
                "scenarios": [
                    "normal_cases",
                    "edge_cases",
                    "error_cases",
                    "performance_test_cases"
                ],
                "volume": {
                    "single_prediction": 1,
                    "small_batch": 10,
                    "medium_batch": 100,
                    "large_batch": 1000
                }
            }
            
            # Log all requirements
            mlflow.log_dict(requirements, "complete_requirements.json")
            
            # Create requirements document
            requirements_doc = self.generate_requirements_document(requirements)
            with open("requirements_document.md", "w") as f:
                f.write(requirements_doc)
            mlflow.log_artifact("requirements_document.md")
            
            mlflow.set_tag("requirements_status", "complete")
            
            return requirements
            
        except Exception as e:
            mlflow.log_param("requirements_error", str(e))
            mlflow.set_tag("requirements_status", "failed")
            self.log_error(f"Requirements elicitation failed: {e}")
            # Return minimal requirements
            return self.create_basic_requirements(initial_request)

def conduct_research_phase(self, initial_request, context):
    """Comprehensive research phase before requirements gathering"""
    import os
    import json
    import mlflow
    from datetime import datetime

    research_findings = {
        'context_analysis': {},
        'regulatory_compliance': {},
        'technology_recommendations': {},
        'data_strategy': {},
        'optimization_approach': {},
        'deployment_architectures': [],
        'timestamp': datetime.now().isoformat()
    }

    try:
        # Search for existing context documents
        context_dir = '/mnt/code/epoch001-research-analysis-planning/'
        if os.path.exists(context_dir):
            research_findings['context_analysis'] = self.analyze_historical_ml_usage(context_dir)

        # Assess regulatory compliance
        research_findings['regulatory_compliance'] = self.assess_regulatory_compliance(
            context.get('domain', 'general'),
            initial_request
        )

        # Generate technology recommendations
        research_findings['technology_recommendations'] = self.recommend_technology_stack(
            initial_request,
            context,
            research_findings['regulatory_compliance']
        )

        # Develop data strategy
        research_findings['data_strategy'] = self.develop_data_strategy(
            initial_request,
            context,
            research_findings['regulatory_compliance']
        )

        # Design optimization strategy
        research_findings['optimization_approach'] = self.design_optimization_strategy(
            context.get('model_type', 'classification'),
            context.get('constraints', {})
        )

        # Create deployment architectures
        research_findings['deployment_architectures'] = self.design_deployment_architectures(
            context,
            research_findings['regulatory_compliance']
        )

        # Generate PDF report
        self.generate_research_report_pdf(research_findings)

        # Save findings to artifacts
        os.makedirs('/mnt/artifacts/reports', exist_ok=True)
        os.makedirs('/mnt/code/docs/business-analysis', exist_ok=True)
        with open('/mnt/artifacts/reports/research_findings.json', 'w') as f:
            json.dump(research_findings, f, indent=2)

        # Log key findings to MLflow
        mlflow.log_param('regulations_identified', len(research_findings['regulatory_compliance'].get('regulations', [])))
        mlflow.log_param('deployment_options', len(research_findings['deployment_architectures']))

        return research_findings

    except Exception as e:
        self.log_error(f"Research phase failed: {e}")
        return research_findings

def analyze_historical_ml_usage(self, context_dir):
    """Analyze past ML/AI implementations and challenges"""
    import os
    import glob

    analysis = {
        'previous_projects': [],
        'common_challenges': [],
        'successful_patterns': [],
        'lessons_learned': []
    }

    # Search for previous project documentation
    for file_path in glob.glob(f"{context_dir}/**/*.md", recursive=True):
        with open(file_path, 'r') as f:
            content = f.read()
            # Extract insights from documentation
            if 'challenge' in content.lower() or 'issue' in content.lower():
                analysis['common_challenges'].append(f"From {os.path.basename(file_path)}: Review for challenges")
            if 'success' in content.lower() or 'achievement' in content.lower():
                analysis['successful_patterns'].append(f"From {os.path.basename(file_path)}: Review for successes")

    # Add standard ML/AI challenges and patterns
    analysis['common_challenges'].extend([
        'Data quality and availability',
        'Model interpretability requirements',
        'Integration with existing systems',
        'Scaling and performance optimization',
        'Stakeholder buy-in and change management'
    ])

    analysis['successful_patterns'].extend([
        'Iterative development with quick wins',
        'Strong baseline models before complex approaches',
        'Comprehensive monitoring and alerting',
        'Clear documentation and knowledge transfer',
        'Regular stakeholder communication'
    ])

    analysis['lessons_learned'] = [
        'Start with simple models and iterate',
        'Invest in data quality early',
        'Automate testing and validation',
        'Plan for model maintenance from the start',
        'Consider compliance requirements upfront'
    ]

    return analysis

def assess_regulatory_compliance(self, domain, use_case):
    """Identify applicable regulations and industry standards"""
    compliance_assessment = {
        'regulations': [],
        'industry_standards': [],
        'best_practices': [],
        'compliance_requirements': {},
        'risk_level': 'medium'
    }

    # Common regulations by domain
    domain_regulations = {
        'finance': ['SOX', 'Basel III', 'Dodd-Frank', 'MiFID II', 'SR 11-7'],
        'healthcare': ['HIPAA', 'FDA 21 CFR Part 11', 'GDPR (health data)'],
        'retail': ['PCI DSS', 'CCPA', 'GDPR'],
        'government': ['FISMA', 'FedRAMP', 'NIST 800-53'],
        'general': ['GDPR', 'CCPA', 'SOC 2']
    }

    # AI/ML specific governance
    ai_governance = [
        'EU AI Act (if applicable)',
        'Algorithmic Accountability Act',
        'ISO/IEC 23053 (AI trustworthiness)',
        'ISO/IEC 23894 (AI risk management)'
    ]

    # Industry standards
    standards = [
        'ISO 27001 (Information Security)',
        'NIST AI Risk Management Framework',
        'IEEE standards for AI/ML',
        'CRISP-DM methodology',
        'MLOps Maturity Model'
    ]

    # Best practices
    best_practices = [
        'Model Cards for Model Reporting',
        'Datasheets for Datasets',
        'Google''s ML Best Practices',
        'Microsoft''s Responsible AI Framework',
        'FAIR principles for data',
        'Explainable AI (XAI) practices'
    ]

    # Determine applicable items based on domain
    compliance_assessment['regulations'] = domain_regulations.get(domain, domain_regulations['general'])
    compliance_assessment['regulations'].extend(ai_governance)
    compliance_assessment['industry_standards'] = standards
    compliance_assessment['best_practices'] = best_practices

    # Define specific requirements
    compliance_assessment['compliance_requirements'] = {
        'data_privacy': 'Implement data anonymization and encryption',
        'model_interpretability': 'Provide model explanations for decisions',
        'audit_trail': 'Maintain comprehensive logging and versioning',
        'bias_testing': 'Regular fairness and bias assessments',
        'security': 'Implement access controls and data protection',
        'documentation': 'Maintain complete model and data documentation'
    }

    # Assess risk level
    if domain in ['finance', 'healthcare', 'government']:
        compliance_assessment['risk_level'] = 'high'
    elif domain in ['retail', 'manufacturing']:
        compliance_assessment['risk_level'] = 'medium'
    else:
        compliance_assessment['risk_level'] = 'low'

    return compliance_assessment

def recommend_technology_stack(self, requirements, context, compliance_needs):
    """Recommend optimal frameworks and libraries considering compliance"""
    recommendations = {
        'ml_frameworks': [],
        'data_processing': [],
        'mlops_tools': [],
        'monitoring': [],
        'explanation_tools': [],
        'justification': {}
    }

    # ML Frameworks based on use case
    if 'deep_learning' in str(requirements).lower():
        recommendations['ml_frameworks'] = ['TensorFlow', 'PyTorch', 'Keras']
    elif 'tabular' in str(requirements).lower() or 'structured' in str(requirements).lower():
        recommendations['ml_frameworks'] = ['XGBoost', 'LightGBM', 'scikit-learn']
    else:
        recommendations['ml_frameworks'] = ['scikit-learn', 'XGBoost', 'LightGBM']

    # Data processing
    recommendations['data_processing'] = [
        'pandas',
        'numpy',
        'Dask (for large datasets)',
        'PySpark (for distributed processing)',
        'Great Expectations (data validation)'
    ]

    # MLOps tools
    recommendations['mlops_tools'] = [
        'MLflow (experiment tracking)',
        'DVC (data version control)',
        'Weights & Biases (advanced tracking)',
        'Kubeflow (if Kubernetes)',
        'Domino Data Lab (enterprise platform)'
    ]

    # Monitoring tools
    recommendations['monitoring'] = [
        'Evidently AI (model monitoring)',
        'Alibi Detect (drift detection)',
        'Prometheus + Grafana (metrics)',
        'WhyLabs (observability)'
    ]

    # Explanation tools for compliance
    if compliance_needs.get('risk_level') in ['high', 'medium']:
        recommendations['explanation_tools'] = [
            'SHAP (Shapley values)',
            'LIME (local explanations)',
            'Alibi Explain',
            'InterpretML'
        ]

    # Justifications
    recommendations['justification'] = {
        'ml_frameworks': 'Selected based on data type and performance requirements',
        'data_processing': 'Chosen for scalability and data quality management',
        'mlops_tools': 'Ensures reproducibility and compliance tracking',
        'monitoring': 'Critical for production stability and compliance',
        'explanation_tools': 'Required for regulatory compliance and transparency'
    }

    return recommendations

def develop_data_strategy(self, use_case, requirements, regulations):
    """Determine optimal data sourcing approach with compliance"""
    data_strategy = {
        'data_sources': [],
        'synthetic_data_approach': {},
        'data_governance': {},
        'privacy_measures': [],
        'quality_requirements': {}
    }

    # Evaluate data sourcing options
    data_strategy['data_sources'] = [
        {'type': 'internal', 'description': 'Company databases and data warehouses'},
        {'type': 'external', 'description': 'Public APIs and datasets'},
        {'type': 'synthetic', 'description': 'Generated data for privacy compliance'}
    ]

    # Synthetic data approach
    data_strategy['synthetic_data_approach'] = {
        'tools': ['SDV (Synthetic Data Vault)', 'CTGAN', 'DataSynthesizer', 'Faker'],
        'use_cases': [
            'Privacy-preserving development',
            'Testing edge cases',
            'Augmenting limited datasets',
            'Compliance with data regulations'
        ],
        'validation': 'Statistical similarity tests and utility metrics'
    }

    # Data governance based on regulations
    data_strategy['data_governance'] = {
        'retention_policy': '7 years for financial, 6 years for healthcare, 3 years default',
        'access_control': 'Role-based access with audit logging',
        'data_lineage': 'Track data source to model predictions',
        'consent_management': 'Required for personal data processing',
        'right_to_deletion': 'Implement GDPR Article 17 compliance'
    }

    # Privacy measures
    if regulations.get('risk_level') in ['high', 'medium']:
        data_strategy['privacy_measures'] = [
            'Differential privacy',
            'Federated learning',
            'Homomorphic encryption',
            'Data anonymization',
            'Secure multi-party computation'
        ]
    else:
        data_strategy['privacy_measures'] = [
            'Data anonymization',
            'Access controls',
            'Encryption at rest and in transit'
        ]

    # Quality requirements
    data_strategy['quality_requirements'] = {
        'completeness': '>95% non-null critical fields',
        'accuracy': 'Validated against ground truth',
        'consistency': 'Cross-system validation',
        'timeliness': 'Data freshness <24 hours',
        'validity': 'Schema and business rule validation'
    }

    return data_strategy

def design_optimization_strategy(self, model_type, constraints):
    """Design comprehensive model optimization approach"""
    optimization_strategy = {
        'hyperparameter_tuning': {},
        'optimization_libraries': [],
        'experiment_tracking': {},
        'performance_benchmarks': {},
        'interpretability_requirements': {}
    }

    # Hyperparameter tuning approach
    if constraints.get('time_constraint', 'medium') == 'tight':
        optimization_strategy['hyperparameter_tuning'] = {
            'method': 'Random Search',
            'iterations': 50,
            'cross_validation': '3-fold',
            'early_stopping': True
        }
    elif constraints.get('accuracy_requirement', 'high') == 'critical':
        optimization_strategy['hyperparameter_tuning'] = {
            'method': 'Bayesian Optimization',
            'iterations': 200,
            'cross_validation': '5-fold',
            'ensemble': True
        }
    else:
        optimization_strategy['hyperparameter_tuning'] = {
            'method': 'Grid Search with pruning',
            'iterations': 100,
            'cross_validation': '5-fold',
            'progressive_sampling': True
        }

    # Optimization libraries
    optimization_strategy['optimization_libraries'] = [
        'Optuna (Bayesian optimization with pruning)',
        'Hyperopt (Tree-structured Parzen Estimators)',
        'Ray Tune (distributed hyperparameter tuning)',
        'scikit-optimize (Sequential model-based optimization)',
        'FLAML (Fast and lightweight AutoML)'
    ]

    # Experiment tracking
    optimization_strategy['experiment_tracking'] = {
        'platform': 'MLflow',
        'metrics_to_track': [
            'accuracy', 'precision', 'recall', 'f1',
            'training_time', 'inference_latency',
            'model_size', 'memory_usage'
        ],
        'artifacts_to_log': [
            'model_file', 'hyperparameters',
            'feature_importance', 'confusion_matrix',
            'validation_curves', 'learning_curves'
        ]
    }

    # Performance benchmarks
    optimization_strategy['performance_benchmarks'] = {
        'accuracy_target': 0.90,
        'latency_p95': 100,  # ms
        'throughput': 1000,  # requests/second
        'model_size': 100,  # MB
        'memory_footprint': 500  # MB
    }

    # Interpretability requirements
    optimization_strategy['interpretability_requirements'] = {
        'global_explanations': 'Feature importance rankings',
        'local_explanations': 'SHAP/LIME for individual predictions',
        'model_documentation': 'Model cards with performance metrics',
        'fairness_metrics': 'Demographic parity and equal opportunity'
    }

    return optimization_strategy

def design_deployment_architectures(self, requirements, compliance_needs):
    """Generate 3 deployment architecture options with compliance considerations"""
    architectures = []

    # Option 1: Quick Demo (Low Compliance)
    architectures.append({
        'name': 'Rapid Prototype',
        'description': 'Streamlit/Dash application for quick demonstrations',
        'components': {
            'frontend': 'Streamlit or Dash',
            'backend': 'Python Flask/FastAPI',
            'model_serving': 'MLflow serve or BentoML',
            'database': 'SQLite or PostgreSQL',
            'monitoring': 'Basic logging'
        },
        'pros': [
            'Quick to deploy (1-2 days)',
            'Low complexity',
            'Good for POCs and demos',
            'Minimal infrastructure'
        ],
        'cons': [
            'Limited scalability',
            'Basic security',
            'Minimal monitoring',
            'Not production-ready'
        ],
        'compliance_level': 'Low',
        'use_cases': ['POC', 'Internal demos', 'Development testing'],
        'estimated_effort': '1-2 days'
    })

    # Option 2: Production-Ready (Medium Compliance)
    architectures.append({
        'name': 'Production Application',
        'description': 'FastAPI backend with React frontend and comprehensive monitoring',
        'components': {
            'frontend': 'React with Material-UI',
            'backend': 'FastAPI with async support',
            'model_serving': 'TorchServe or TensorFlow Serving',
            'database': 'PostgreSQL with Redis cache',
            'monitoring': 'Prometheus + Grafana',
            'authentication': 'OAuth2 with JWT',
            'api_gateway': 'Kong or Nginx'
        },
        'pros': [
            'Scalable architecture',
            'Good performance',
            'Comprehensive monitoring',
            'Security features',
            'A/B testing capability'
        ],
        'cons': [
            'More complex setup',
            'Requires DevOps expertise',
            'Higher maintenance'
        ],
        'compliance_level': 'Medium',
        'use_cases': ['Customer-facing apps', 'Internal production systems'],
        'estimated_effort': '2-3 weeks'
    })

    # Option 3: Enterprise MLOps (High Compliance)
    architectures.append({
        'name': 'Enterprise MLOps Platform',
        'description': 'Full MLOps platform with Domino Data Lab and enterprise features',
        'components': {
            'platform': 'Domino Data Lab',
            'workflow': 'Domino Flows or Kubeflow',
            'model_registry': 'MLflow Model Registry',
            'serving': 'Seldon Core or KServe',
            'monitoring': 'Datadog or New Relic',
            'security': 'Vault for secrets, RBAC',
            'compliance': 'Audit logging, encryption',
            'ci_cd': 'GitLab CI/CD or Jenkins'
        },
        'pros': [
            'Enterprise-grade security',
            'Full compliance support',
            'Automated workflows',
            'Model governance',
            'Scalable and reliable',
            'Disaster recovery'
        ],
        'cons': [
            'High complexity',
            'Significant setup time',
            'Requires specialized team',
            'Higher costs'
        ],
        'compliance_level': 'High',
        'use_cases': ['Regulated industries', 'Mission-critical systems', 'Large-scale deployments'],
        'estimated_effort': '4-6 weeks'
    })

    return architectures

def generate_research_report_pdf(self, research_findings):
    """Create comprehensive PDF research report"""
    import os
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
    from datetime import datetime

    # Create artifacts directory
    os.makedirs('/mnt/artifacts/reports', exist_ok=True)
    os.makedirs('/mnt/code/docs/business-analysis', exist_ok=True)

    # Create PDF
    os.makedirs('/mnt/artifacts/reports', exist_ok=True)
    pdf_path = '/mnt/artifacts/reports/business_analysis_report.pdf'
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        textColor=colors.HexColor('#2C3E50'),
        alignment=TA_CENTER
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#34495E'),
        spaceAfter=12
    )

    # Title
    story.append(Paragraph("ML/AI Project Research Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 0.5*inch))

    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    summary_text = """This research report provides a comprehensive analysis for the proposed ML/AI project,
    including regulatory compliance assessment, technology recommendations, data strategy, and deployment options.
    The analysis considers historical patterns, industry best practices, and compliance requirements."""
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 0.3*inch))

    # Regulatory Compliance Assessment
    story.append(Paragraph("Regulatory Compliance Assessment", heading_style))

    if research_findings.get('regulatory_compliance'):
        compliance = research_findings['regulatory_compliance']

        # Risk Level
        risk_text = f"Risk Level: <b>{compliance.get('risk_level', 'Medium').upper()}</b>"
        story.append(Paragraph(risk_text, styles['Normal']))
        story.append(Spacer(1, 0.1*inch))

        # Regulations table
        if compliance.get('regulations'):
            story.append(Paragraph("<b>Applicable Regulations:</b>", styles['Normal']))
            for reg in compliance['regulations']:
                story.append(Paragraph(f"• {reg}", styles['Normal']))
            story.append(Spacer(1, 0.1*inch))

        # Standards
        if compliance.get('industry_standards'):
            story.append(Paragraph("<b>Industry Standards:</b>", styles['Normal']))
            for standard in compliance['industry_standards']:
                story.append(Paragraph(f"• {standard}", styles['Normal']))
            story.append(Spacer(1, 0.1*inch))

        # Best Practices
        if compliance.get('best_practices'):
            story.append(Paragraph("<b>Recommended Best Practices:</b>", styles['Normal']))
            for practice in compliance['best_practices']:
                story.append(Paragraph(f"• {practice}", styles['Normal']))

    story.append(PageBreak())

    # Historical Analysis
    story.append(Paragraph("Historical Analysis", heading_style))

    if research_findings.get('context_analysis'):
        context = research_findings['context_analysis']

        if context.get('common_challenges'):
            story.append(Paragraph("<b>Common Challenges Identified:</b>", styles['Normal']))
            for challenge in context['common_challenges'][:5]:  # Top 5
                story.append(Paragraph(f"• {challenge}", styles['Normal']))
            story.append(Spacer(1, 0.1*inch))

        if context.get('successful_patterns'):
            story.append(Paragraph("<b>Successful Patterns:</b>", styles['Normal']))
            for pattern in context['successful_patterns'][:5]:  # Top 5
                story.append(Paragraph(f"• {pattern}", styles['Normal']))

    story.append(Spacer(1, 0.3*inch))

    # Technology Recommendations
    story.append(Paragraph("Technology Recommendations", heading_style))

    if research_findings.get('technology_recommendations'):
        tech = research_findings['technology_recommendations']

        # Create technology table
        tech_data = [
            ['Category', 'Recommended Technologies']
        ]

        for category, tools in tech.items():
            if isinstance(tools, list) and category != 'justification':
                tech_data.append([category.replace('_', ' ').title(), ', '.join(tools[:3])])

        if len(tech_data) > 1:
            tech_table = Table(tech_data, colWidths=[2*inch, 4*inch])
            tech_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(tech_table)

    story.append(PageBreak())

    # Data Strategy
    story.append(Paragraph("Data Strategy", heading_style))

    if research_findings.get('data_strategy'):
        data = research_findings['data_strategy']

        # Data sources
        story.append(Paragraph("<b>Recommended Data Approach:</b>", styles['Normal']))
        story.append(Paragraph("Consider a hybrid approach combining:", styles['Normal']))
        story.append(Paragraph("• Real data for core training (when available)", styles['Normal']))
        story.append(Paragraph("• Synthetic data for privacy compliance and testing", styles['Normal']))
        story.append(Paragraph("• External APIs for enrichment", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))

        # Privacy measures
        if data.get('privacy_measures'):
            story.append(Paragraph("<b>Privacy Measures:</b>", styles['Normal']))
            for measure in data['privacy_measures'][:3]:  # Top 3
                story.append(Paragraph(f"• {measure}", styles['Normal']))

    story.append(Spacer(1, 0.3*inch))

    # Model Optimization Strategy
    story.append(Paragraph("Model Optimization Strategy", heading_style))

    if research_findings.get('optimization_approach'):
        opt = research_findings['optimization_approach']

        if opt.get('hyperparameter_tuning'):
            tuning = opt['hyperparameter_tuning']
            story.append(Paragraph(f"<b>Recommended Approach:</b> {tuning.get('method', 'Bayesian Optimization')}", styles['Normal']))
            story.append(Paragraph(f"<b>Cross-validation:</b> {tuning.get('cross_validation', '5-fold')}", styles['Normal']))
            story.append(Spacer(1, 0.1*inch))

        if opt.get('optimization_libraries'):
            story.append(Paragraph("<b>Recommended Libraries:</b>", styles['Normal']))
            for lib in opt['optimization_libraries'][:3]:  # Top 3
                story.append(Paragraph(f"• {lib}", styles['Normal']))

    story.append(PageBreak())

    # Deployment Architectures
    story.append(Paragraph("Deployment Architecture Options", heading_style))

    if research_findings.get('deployment_architectures'):
        for i, arch in enumerate(research_findings['deployment_architectures'], 1):
            story.append(Paragraph(f"<b>Option {i}: {arch['name']}</b>", styles['Heading2']))
            story.append(Paragraph(arch['description'], styles['Normal']))
            story.append(Spacer(1, 0.1*inch))

            # Pros and Cons table
            pros_cons_data = [
                ['Pros', 'Cons'],
                ['\n'.join(arch['pros'][:3]), '\n'.join(arch['cons'][:3])]
            ]

            pros_cons_table = Table(pros_cons_data, colWidths=[3*inch, 3*inch])
            pros_cons_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(pros_cons_table)

            story.append(Paragraph(f"<b>Compliance Level:</b> {arch['compliance_level']}", styles['Normal']))
            story.append(Paragraph(f"<b>Estimated Effort:</b> {arch['estimated_effort']}", styles['Normal']))
            story.append(Spacer(1, 0.3*inch))

    # Risk Assessment
    story.append(Paragraph("Risk Assessment and Mitigation", heading_style))

    risk_text = """Key risks have been identified across technical, business, and compliance dimensions.
    Mitigation strategies include iterative development, comprehensive testing, stakeholder engagement,
    and continuous monitoring. Regular compliance audits and model validation will be essential."""
    story.append(Paragraph(risk_text, styles['Normal']))
    story.append(Spacer(1, 0.3*inch))

    # Next Steps
    story.append(Paragraph("Recommended Next Steps", heading_style))
    next_steps = [
        "1. Review and validate regulatory requirements with legal/compliance teams",
        "2. Finalize technology stack based on team expertise and infrastructure",
        "3. Develop proof of concept with Option 1 architecture",
        "4. Create detailed project timeline and resource plan",
        "5. Establish MLOps practices and monitoring from the start",
        "6. Plan for iterative development with regular stakeholder reviews"
    ]

    for step in next_steps:
        story.append(Paragraph(step, styles['Normal']))

    # Build PDF
    doc.build(story)

    # Log that report was generated
    print(f"Research report generated: {pdf_path}")

    return pdf_path
```