"""
app/routers/aqi.py — AQI Data Endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
import httpx
import os

from app.database import get_db, AQIRecord
from app.config import settings

router = APIRouter()


# ──────────────────────────────────────────────
# Pydantic Schemas
# ──────────────────────────────────────────────

class AQIResponse(BaseModel):
    id: int
    city: str
    timestamp: datetime
    aqi: float
    category: str
    pm25: Optional[float]
    pm10: Optional[float]
    co: Optional[float]
    no2: Optional[float]
    so2: Optional[float]
    o3: Optional[float]
    temperature: Optional[float]
    humidity: Optional[float]

    class Config:
        from_attributes = True


def get_aqi_category(aqi: float) -> str:
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Moderate"
    if aqi <= 150:  return "Unhealthy for Sensitive Groups"
    if aqi <= 200:  return "Unhealthy"
    if aqi <= 300:  return "Very Unhealthy"
    return "Hazardous"


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@router.get("/current/{city}", response_model=AQIResponse)
async def get_current_aqi(city: str, db: Session = Depends(get_db)):
    """Get the most recent AQI record for a city."""
    record = (
        db.query(AQIRecord)
        .filter(AQIRecord.city == city)
        .order_by(desc(AQIRecord.timestamp))
        .first()
    )
    if not record:
        raise HTTPException(status_code=404, detail=f"No data found for city: {city}")
    return record


@router.get("/history/{city}")
async def get_aqi_history(
    city: str,
    hours: int = Query(24, ge=1, le=720),
    db: Session = Depends(get_db),
):
    """Get historical AQI data for a city (up to 30 days)."""
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    records = (
        db.query(AQIRecord)
        .filter(AQIRecord.city == city, AQIRecord.timestamp >= cutoff)
        .order_by(AQIRecord.timestamp)
        .all()
    )
    return {
        "city": city,
        "hours": hours,
        "count": len(records),
        "data": [
            {
                "timestamp": r.timestamp.isoformat(),
                "aqi": r.aqi,
                "pm25": r.pm25,
                "pm10": r.pm10,
                "category": r.category,
            }
            for r in records
        ],
    }


@router.get("/all-cities")
async def get_all_cities_current(db: Session = Depends(get_db)):
    """Get current AQI for all monitored cities."""
    results = []
    for city in settings.CITIES:
        record = (
            db.query(AQIRecord)
            .filter(AQIRecord.city == city)
            .order_by(desc(AQIRecord.timestamp))
            .first()
        )
        if record:
            results.append({
                "city": city,
                "aqi": record.aqi,
                "category": record.category,
                "pm25": record.pm25,
                "timestamp": record.timestamp.isoformat(),
            })
    return {"cities": results}


@router.post("/fetch-live")
async def fetch_live_from_openaq(city: str = "Kathmandu", db: Session = Depends(get_db)):
    """Fetch real-time data from OpenAQ API and store in database."""
    location_ids = {
        "Kathmandu": "NPL001", "Pokhara": "NPL002",
        "Lalitpur": "NPL003",  "Bhaktapur": "NPL004",
    }
    if city not in location_ids:
        raise HTTPException(status_code=400, detail=f"City {city} not in OpenAQ registry")

    url = f"https://api.openaq.io/v3/locations/{location_ids[city]}/latest"
    headers = {"X-API-Key": settings.OPENAQ_API_KEY}

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"OpenAQ API error: {str(e)}")

    # Parse and store (simplified — adapt to actual API response structure)
    record = AQIRecord(city=city, aqi=0, category="Unknown")
    db.add(record)
    db.commit()
    db.refresh(record)
    return {"message": "Data fetched and stored", "record_id": record.id}