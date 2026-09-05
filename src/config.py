import os
from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Detect Streamlit Cloud (read-only source, writable home)
def _get_data_dir() -> Path:
    """Return writable data directory. On Streamlit Cloud, use ~/.streamlit/data."""
    # Check if running on Streamlit Cloud
    if os.path.exists("/mount/src"):
        home = Path.home()
        data_dir = home / ".streamlit" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "raw").mkdir(exist_ok=True)
        (data_dir / "processed").mkdir(exist_ok=True)
        return data_dir
    return PROJECT_ROOT / "data"


DATA_DIR = _get_data_dir()
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


# Dataset configuration
NUM_TRANSACTIONS = 10_000
NUM_USERS = 1_000
NUM_MERCHANTS = 200
NUM_DEVICES = 1_200
NUM_IPS = 800


# Reproducibility
RANDOM_SEED = 42