"""
app/config.py — Application Configuration
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/airsense_db"

    # APIs
    OPENAQ_API_KEY: str = ""
    IQAIR_API_KEY:  str = ""
    WAQI_API_TOKEN: str = ""

    # Anthropic AI
    ANTHROPIC_API_KEY: str = ""

    # Auth
    SECRET_KEY:         str = "your-super-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE: int = 1440  # minutes

    # ML Model
    MODEL_PATH: str   = "models/lstm_aqi_model.h5"
    SCALER_PATH: str  = "models/scaler.pkl"
    SEQUENCE_LEN: int = 24

    # App
    DEBUG: bool  = True
    CITIES: list = [
        "Kathmandu", "Pokhara", "Lalitpur",
        "Bhaktapur", "Biratnagar", "Bharatpur", "Birgunj", "Dharan"
    ]

    class Config:
        env_file = ".env"


settings = Settings()