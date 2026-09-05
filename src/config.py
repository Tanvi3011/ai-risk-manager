from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Data directories
DATA_DIR = PROJECT_ROOT / "data"
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