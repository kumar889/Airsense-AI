"""
AirSense AI - FastAPI Backend
Air Quality Monitoring & LSTM Prediction System
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import uvicorn
import logging

from app.routers import aqi, predictions, admin, alerts
from app.database import engine, Base
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Starting AirSense AI Backend...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created.")
    yield
    logger.info("Shutting down AirSense AI Backend...")


app = FastAPI(
    title="AirSense AI API",
    description="Air Quality Monitoring & LSTM Prediction System for Nepal/Kathmandu",
    version="2.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: restrict to your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(aqi.router,         prefix="/api/v1/aqi",         tags=["AQI Data"])
app.include_router(predictions.router, prefix="/api/v1/predictions", tags=["LSTM Predictions"])
app.include_router(admin.router,       prefix="/api/v1/admin",       tags=["Admin"])
app.include_router(alerts.router,      prefix="/api/v1/alerts",      tags=["Alerts"])


@app.get("/")
async def root():
    return {
        "service": "AirSense AI API",
        "version": "2.1.0",
        "status": "online",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "database": "connected", "model": "loaded"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)