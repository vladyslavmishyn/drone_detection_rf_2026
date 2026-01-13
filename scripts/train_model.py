"""Train ML models for drone detection."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Train drone detection model.
    
    TODO: Implement model training pipeline
    - Load processed dataset from /data
    - Extract features using src.core.utils
    - Train random forest classifier
    - Save model to /models
    """
    print("Model training not yet implemented")
    print("TODO: Implement using /src/models and /data")


if __name__ == "__main__":
    main()
