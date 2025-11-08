#!/usr/bin/env python3
"""
Domino Flows Input/Output Helper

This module provides utilities for scripts to work both standalone and in Domino Flows.

According to Domino documentation:
- Workflow inputs appear at: /workflow/inputs/<INPUT_NAME>
- Workflow outputs must be written to: /workflow/outputs/<OUTPUT_NAME>
- Tasks MUST write all declared outputs or the task fails

Usage:
    from src.models.workflow_io import WorkflowIO

    wf = WorkflowIO()

    # Check if running in a flow
    if wf.is_workflow_job():
        # Read flow input
        data = wf.read_input("data_prep")  # Reads from /workflow/inputs/data_prep

        # Write flow output
        wf.write_output("training_summary", summary_data)  # Writes to /workflow/outputs/training_summary
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class WorkflowIO:
    """Helper class for handling Domino Flows inputs and outputs"""

    WORKFLOW_INPUTS_DIR = Path("/workflow/inputs")
    WORKFLOW_OUTPUTS_DIR = Path("/workflow/outputs")

    def __init__(self):
        """Initialize WorkflowIO helper"""
        self._is_workflow = None

    def is_workflow_job(self) -> bool:
        """
        Check if running in a Domino Flow.

        Returns:
            bool: True if running in a Domino Flow, False if standalone
        """
        if self._is_workflow is None:
            # Check environment variable first
            env_check = os.getenv("DOMINO_IS_WORKFLOW_JOB", "").lower() == "true"

            # Fallback: check if workflow directories exist
            dir_check = self.WORKFLOW_INPUTS_DIR.exists() or self.WORKFLOW_OUTPUTS_DIR.exists()

            self._is_workflow = env_check or dir_check

            if self._is_workflow:
                logger.info("Running in Domino Flow mode")
            else:
                logger.info("Running in standalone mode")

        return self._is_workflow

    def read_input(self, name: str) -> Optional[Any]:
        """
        Read an input from the workflow inputs directory.

        According to Domino docs, inputs appear at /workflow/inputs/<INPUT_NAME>:
        - For files (FlyteFile), the path itself is the file
        - For primitives, read the value from the file

        Args:
            name: Name of the input (matches the key in inputs dict)

        Returns:
            Input value (loaded from JSON if applicable) or None if not found
        """
        if not self.is_workflow_job():
            logger.warning(f"Not in workflow mode, cannot read input '{name}'")
            return None

        input_path = self.WORKFLOW_INPUTS_DIR / name

        if not input_path.exists():
            logger.warning(f"Workflow input '{name}' not found at {input_path}")
            return None

        try:
            # Try to read as JSON first (most common for our use case)
            with open(input_path, 'r') as f:
                content = f.read().strip()

                # Try to parse as JSON
                try:
                    data = json.loads(content)
                    logger.info(f"Read workflow input '{name}' as JSON from {input_path}")
                    return data
                except json.JSONDecodeError:
                    # Not JSON, return as string
                    logger.info(f"Read workflow input '{name}' as string from {input_path}")
                    return content

        except Exception as e:
            logger.error(f"Error reading workflow input '{name}': {e}")
            return None

    def write_output(self, name: str, data: Any) -> bool:
        """
        Write an output to the workflow outputs directory.

        According to Domino docs, outputs MUST be written to /workflow/outputs/<OUTPUT_NAME>.
        If outputs don't exist at job completion, the task fails.

        Args:
            name: Name of the output (matches the key in outputs dict)
            data: Data to write (will be JSON-serialized for dicts/lists)

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_workflow_job():
            logger.info(f"Not in workflow mode, skipping output write for '{name}'")
            return False

        # Ensure outputs directory exists
        self.WORKFLOW_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

        output_path = self.WORKFLOW_OUTPUTS_DIR / name

        try:
            # Write as JSON if dict or list
            if isinstance(data, (dict, list)):
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                logger.info(f"Wrote workflow output '{name}' as JSON to {output_path}")
            else:
                # Write as string for primitives
                with open(output_path, 'w') as f:
                    f.write(str(data))
                logger.info(f"Wrote workflow output '{name}' as string to {output_path}")

            return True

        except Exception as e:
            logger.error(f"Error writing workflow output '{name}': {e}")
            return False

    def ensure_output_written(self, name: str, default_data: Optional[Dict] = None) -> None:
        """
        Ensure an output file exists, writing default data if needed.

        This is critical for Domino Flows - tasks fail if declared outputs don't exist.

        Args:
            name: Output name
            default_data: Default data to write if output doesn't exist
        """
        if not self.is_workflow_job():
            return

        output_path = self.WORKFLOW_OUTPUTS_DIR / name

        if not output_path.exists():
            logger.warning(f"Output '{name}' doesn't exist, writing default data")
            if default_data is None:
                default_data = {"status": "completed", "message": "No data generated"}
            self.write_output(name, default_data)

    def write_error_output(self, name: str, error: Exception, framework: str = "unknown") -> None:
        """
        Write an error output when script fails.

        This ensures the sidecar uploader has an output file even if the script crashes.

        Args:
            name: Output name (e.g., "training_summary")
            error: The exception that occurred
            framework: Framework name (e.g., "autogluon", "prophet")
        """
        from datetime import datetime

        error_data = {
            "timestamp": datetime.now().isoformat(),
            "framework": framework,
            "status": "error",
            "error_message": str(error),
            "error_type": type(error).__name__,
            "total_configs": 0,
            "successful_configs": 0,
            "best_config": None,
            "best_mae": None
        }

        logger.error(f"Writing error output for '{name}': {error}")
        self.write_output(name, error_data)

    def get_model_save_path(self, base_path: str = "/mnt/data/Oil-and-Gas-Demo") -> Path:
        """
        Get appropriate model save path based on execution mode
        
        Args:
            base_path: Default path for non-Flow execution
            
        Returns:
            Path object for saving models
        """
        if self.is_workflow_job():
            # In Domino Flows, models should be saved to workflow outputs
            # They will be handled as artifacts by the Flow system
            return self.WORKFLOW_OUTPUTS_DIR / "models"
        else:
            # Normal execution - use artifacts directory
            return Path(base_path)
            
    def get_artifact_save_path(self, artifact_type: str = "models") -> Path:
        """
        Get appropriate artifact save path based on execution mode
        
        Args:
            artifact_type: Type of artifact (models, reports, etc.)
            
        Returns:
            Path object for saving artifacts
        """
        if self.is_workflow_job():
            # In Domino Flows, save to workflow outputs
            return self.WORKFLOW_OUTPUTS_DIR / artifact_type
        else:
            # Normal execution - use artifacts directory
            return Path(f"/mnt/data/Oil-and-Gas-Demo/{artifact_type}")

    def ensure_model_directory(self, subdirectory: str = "models") -> Path:
        """
        Ensure model directory exists and return path
        
        Args:
            subdirectory: Subdirectory within the model save path
            
        Returns:
            Path to the model directory
        """
        base_path = self.get_model_save_path()
        model_dir = base_path / subdirectory
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir


# Convenience functions for quick use
def is_workflow_job() -> bool:
    """Check if running in a Domino Flow"""
    return WorkflowIO().is_workflow_job()


def read_workflow_input(name: str) -> Optional[Any]:
    """Read a workflow input"""
    return WorkflowIO().read_input(name)


def write_workflow_output(name: str, data: Any) -> bool:
    """Write a workflow output"""
    return WorkflowIO().write_output(name, data)


if __name__ == "__main__":
    # Test the module
    logging.basicConfig(level=logging.INFO)

    wf = WorkflowIO()

    print(f"Is workflow job: {wf.is_workflow_job()}")
    print(f"Workflow inputs dir exists: {wf.WORKFLOW_INPUTS_DIR.exists()}")
    print(f"Workflow outputs dir exists: {wf.WORKFLOW_OUTPUTS_DIR.exists()}")

    if wf.is_workflow_job():
        # List available inputs
        if wf.WORKFLOW_INPUTS_DIR.exists():
            inputs = list(wf.WORKFLOW_INPUTS_DIR.iterdir())
            print(f"Available inputs: {[i.name for i in inputs]}")

        # Test writing an output
        test_data = {"test": "data", "timestamp": "2025-01-01"}
        success = wf.write_output("test_output", test_data)
        print(f"Test output write: {'SUCCESS' if success else 'FAILED'}")
