"""Deploy trained models to hardware."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Deploy drone detection system to receiver hardware.
    
    TODO: Implement hardware deployment
    - Load trained model from /models
    - Configure receiver nodes (/src/hardware)
    - Push code to hardware devices
    """
    print("Hardware deployment not yet implemented")
    print("TODO: Implement using /src/hardware")


if __name__ == "__main__":
    main()
