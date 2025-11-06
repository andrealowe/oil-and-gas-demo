# Data

Data storage directory (contents excluded from git via .gitignore).

## Structure

```
data/
├── raw/               # Original, immutable data
├── processed/         # Cleaned and transformed data
├── features/          # Generated feature sets
└── external/          # External data sources
```

## Notes

- Never commit large datasets to git
- Use Domino Datasets for data storage and versioning
- Document data sources and transformations in `docs/data_dictionary.md`
- Add data validation checks to ensure quality
