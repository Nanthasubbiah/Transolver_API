"""
Application configuration via environment variables.
"""
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./elasticity.db")
DEVICE = os.getenv("DEVICE", "cpu")
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")

MODELS_JSON = os.getenv("MODELS_JSON", "./models.json")
CHECKPOINTS_DIR = os.getenv("CHECKPOINTS_DIR", "./checkpoints")
TRAIN_DATA_DIR = os.getenv("TRAIN_DATA_DIR", "./train_data")
UPLOADS_DIR = os.getenv("UPLOADS_DIR", "./uploads")
SCREENSHOTS_DIR = os.getenv("SCREENSHOTS_DIR", "./screenshots")
