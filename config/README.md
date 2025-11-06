# Configuration

Configuration files for different environments and components.

## Files

```
config/
├── model_config.yaml           # Model hyperparameters and settings
├── data_config.yaml            # Data sources and processing params
├── monitoring_config.json      # Model monitoring configuration
└── deployment_config.yaml      # Deployment settings (staging, prod)
```

## Best Practices

- Separate configs by environment (dev, staging, prod)
- Never commit secrets or credentials
- Use environment variables for sensitive values
- Document all configuration parameters
