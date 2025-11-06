# Tests

Automated tests for production code quality and reliability.

## Test Types

```
tests/
├── unit/              # Unit tests for individual functions
├── integration/       # Integration tests for workflows
├── model/             # Model validation and fairness tests
└── conftest.py        # Pytest configuration and fixtures
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/unit/test_features.py

# Run with coverage
pytest --cov=src tests/
```

## Guidelines

- Aim for >80% code coverage
- Test edge cases and error handling
- Include model performance regression tests
- Test fairness and bias metrics for production models
