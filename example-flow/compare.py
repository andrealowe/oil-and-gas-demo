# compare_training_results.py
import json
import os
import sys
from pathlib import Path

# Add project root to path for data_config import
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from data_config import get_config

# Get configuration
config = get_config()
artifacts_dir = config.get_artifacts_dir()

print(f"Reading results from: {artifacts_dir}")

# Read result files directly from artifacts directory
def read_result_file(model_name: str) -> dict:
    """Read model results from artifacts directory"""
    result_file = artifacts_dir / f"{model_name.lower()}_result.json"

    if result_file.exists():
        with open(result_file, 'r') as f:
            return json.load(f)
    else:
        print(f"⚠️  Result file not found: {result_file}")
        return None

# Read the three result blobs
ada_blob = read_result_file("adaboost")
gnb_blob = read_result_file("gaussiannb")
xgb_blob = read_result_file("xgboost")

# Filter out None results
results = {
    "AdaBoost": ada_blob,
    "GaussianNB": gnb_blob,
    "XGBoost": xgb_blob
}
results = {k: v for k, v in results.items() if v is not None}

if not results:
    print("❌ No model results found!")
    print("   Make sure training tasks completed successfully.")
    sys.exit(1)

print(f"\n{'=' * 60}")
print("MODEL COMPARISON RESULTS")
print(f"{'=' * 60}\n")

# Find best model by ROC AUC
best_model, best_metric = '', 0
for name, blob in results.items():
    roc_auc = blob.get('roc_auc', 0)
    print(f"{name:20s} ROC AUC: {roc_auc:.4f}")

    if roc_auc > best_metric:
        best_model = name
        best_metric = roc_auc

print(f"\n{'=' * 60}")
print(f"🏆 BEST MODEL: {best_model}")
print(f"   ROC AUC: {best_metric:.4f}")
print(f"{'=' * 60}\n")

# Prepare output for workflow
OUT_DIR = Path("/workflow/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "consolidated"

output_message = f"Best model: {best_model} with ROC AUC: {best_metric:.4f}"
OUT_FILE.write_text(output_message)

# Also save detailed comparison to artifacts
comparison_file = artifacts_dir / "model_comparison.json"
comparison_data = {
    "best_model": best_model,
    "best_roc_auc": float(best_metric),
    "all_results": {k: {"roc_auc": v.get("roc_auc")} for k, v in results.items()}
}

with open(comparison_file, 'w') as f:
    json.dump(comparison_data, f, indent=2)

print(f"✅ Comparison saved to: {comparison_file}")
