"""
app/database.py — SQLAlchemy Database Setup
"""
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

from app.config import settings

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ──────────────────────────────────────────────
# ORM Models
# ──────────────────────────────────────────────

class AQIRecord(Base):
    __tablename__ = "aqi_records"

    id          = Column(Integer, primary_key=True, index=True)
    city        = Column(String(100), index=True, nullable=False)
    timestamp   = Column(DateTime, default=datetime.utcnow, index=True)

    # Core AQI
    aqi         = Column(Float, nullable=False)
    category    = Column(String(50))

    # Pollutants
    pm25        = Column(Float)
    pm10        = Column(Float)
    co          = Column(Float)
    no2         = Column(Float)
    so2         = Column(Float)
    o3          = Column(Float)

    # Meteorological
    temperature = Column(Float)
    humidity    = Column(Float)
    wind_speed  = Column(Float)
    pressure    = Column(Float)

    # Metadata
    source      = Column(String(50), default="OpenAQ")
    is_verified = Column(Boolean, default=True)


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id            = Column(Integer, primary_key=True, index=True)
    city          = Column(String(100), index=True)
    created_at    = Column(DateTime, default=datetime.utcnow)
    horizon       = Column(String(10))  # 24h, 3d, 7d
    model_version = Column(String(20), default="2.1")

    # Predictions stored as JSON string
    predictions   = Column(Text)
    rmse          = Column(Float)
    mae           = Column(Float)

    # Confidence bands
    upper_bound   = Column(Text)
    lower_bound   = Column(Text)


class Alert(Base):
    __tablename__ = "alerts"

    id         = Column(Integer, primary_key=True, index=True)
    city       = Column(String(100))
    timestamp  = Column(DateTime, default=datetime.utcnow)
    aqi_value  = Column(Float)
    threshold  = Column(Float)
    category   = Column(String(50))
    message    = Column(Text)
    is_active  = Column(Boolean, default=True)