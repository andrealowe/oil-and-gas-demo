# Source Code

Production-ready source code for the ML project.

## Structure

```
src/
├── data/              # Data processing and ETL pipelines
├── models/            # Model training and inference code
├── features/          # Feature engineering modules
├── utils/             # Shared utilities and helpers
├── api/               # API endpoints (predict.py, etc.)
└── monitoring/        # Model monitoring and ground truth scripts
```

## Best Practices

- Keep exploratory code in `notebooks/`, production code in `src/`
- All modules should have docstrings and type hints
- Write unit tests in `tests/` for all production code
- Use configuration files from `config/` directory
