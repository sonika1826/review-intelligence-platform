from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Source
SRC_DIR = PROJECT_ROOT / "src"

# Models
MODELS_DIR = PROJECT_ROOT / "models"

# Artifacts
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Logs
LOGS_DIR = PROJECT_ROOT / "logs"

# Reports
REPORTS_DIR = PROJECT_ROOT / "reports"

# Configs
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Tests
TESTS_DIR = PROJECT_ROOT / "tests"

# App
APP_DIR = PROJECT_ROOT / "app"
