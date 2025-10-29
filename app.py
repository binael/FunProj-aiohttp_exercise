"""
RESTful Countries Cache API (FastAPI + PostgreSQL + SQLAlchemy + Async)

This API:
- Fetches country data from https://restcountries.com/v2/all
- Fetches exchange rates from https://open.er-api.com/v6/latest/USD
- Computes estimated GDP = population × random(1000–2000) ÷ exchange_rate
- Caches everything in PostgreSQL
- Generates a summary image after each refresh
- Provides CRUD and status endpoints

Run setup:
    pip install fastapi[all] sqlalchemy[asyncio] asyncpg httpx pillow python-multipart
    export DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/countries_db
    uvicorn main:app --reload
"""

from __future__ import annotations
import os
import random
import datetime
from typing import List, Optional, Dict, Any, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy import (
    Column,
    Integer,
    String,
    BigInteger,
    Float,
    DateTime,
    func,
    select,
)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import SQLAlchemyError
from PIL import Image, ImageDraw, ImageFont

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
# Load environment variables
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL"
)
COUNTRIES_API = "https://restcountries.com/v2/all?fields=name,capital,region,population,flag,currencies"
EXCHANGE_API = "https://open.er-api.com/v6/latest/USD"
CACHE_IMAGE_PATH = os.getenv("CACHE_IMAGE_PATH", "cache/summary.png")
os.makedirs(os.path.dirname(CACHE_IMAGE_PATH) or ".", exist_ok=True)

# -------------------------------------------------------
# Database Setup
# -------------------------------------------------------
Base = declarative_base()


class Country(Base):
    """Country model representing cached API data."""
    __tablename__ = "countries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True, index=True)
    capital: Mapped[Optional[str]] = mapped_column(String(200))
    region: Mapped[Optional[str]] = mapped_column(String(100))
    population: Mapped[int] = mapped_column(BigInteger, nullable=False)
    currency_code: Mapped[Optional[str]] = mapped_column(String(10), index=True)
    exchange_rate: Mapped[Optional[float]] = mapped_column(Float)
    estimated_gdp: Mapped[Optional[float]] = mapped_column(Float)
    flag_url: Mapped[Optional[str]] = mapped_column(String(500))
    last_refreshed_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


engine = create_async_engine(DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


# -------------------------------------------------------
# FastAPI App
# -------------------------------------------------------
app = FastAPI(title="Countries Cache API", version="1.0")


# -------------------------------------------------------
# Schemas
# -------------------------------------------------------
class CountryOut(BaseModel):
    id: int
    name: str
    capital: Optional[str]
    region: Optional[str]
    population: int
    currency_code: Optional[str]
    exchange_rate: Optional[float]
    estimated_gdp: Optional[float]
    flag_url: Optional[str]
    last_refreshed_at: datetime.datetime

    class Config:
        orm_mode = True


class StatusOut(BaseModel):
    total_countries: int
    last_refreshed_at: Optional[datetime.datetime]


class CountryCreateIn(BaseModel):
    name: str = Field(..., min_length=1)
    population: int = Field(..., ge=0)
    currency_code: str = Field(..., min_length=1)
    capital: Optional[str] = None
    region: Optional[str] = None
    flag_url: Optional[str] = None

    @validator("name")
    def strip_name(cls, v: str) -> str:
        return v.strip()


# -------------------------------------------------------
# DB Dependency
# -------------------------------------------------------
async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


async def ensure_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@app.on_event("startup")
async def startup_event():
    await ensure_tables()


# -------------------------------------------------------
# Utility functions
# -------------------------------------------------------
async def fetch_external_json(url: str) -> Any:
    """Fetch JSON from external API or raise 503 error."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError:
        raise HTTPException(
            status_code=503,
            detail={"error": "External data source unavailable", "details": f"Could not fetch data from {url}"},
        )


def extract_currency_code(item: dict) -> Optional[str]:
    currencies = item.get("currencies") or []
    if not currencies:
        return None
    first = currencies[0]
    return first.get("code")


def compute_estimated_gdp(population: int, exchange_rate: Optional[float]) -> Optional[float]:
    if not population or not exchange_rate:
        return None
    multiplier = random.uniform(1000, 2000)
    return (population * multiplier) / exchange_rate


def generate_summary_image(total: int, top5: List[Tuple[str, Optional[float]]], refreshed_at: datetime.datetime):
    """Generate and save summary image."""
    width, height = 800, 300
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = bold = ImageFont.load_default()

    y = 30
    draw.text((20, y), "Countries Summary", font=bold, fill="black")
    y += 40
    draw.text((20, y), f"Total Countries: {total}", font=font, fill="black")
    y += 30
    draw.text((20, y), "Top 5 by Estimated GDP:", font=font, fill="black")
    y += 30

    for i, (name, gdp) in enumerate(top5[:5], start=1):
        draw.text((40, y), f"{i}. {name} - {gdp:,.2f}" if gdp else f"{i}. {name} - N/A", font=font, fill="black")
        y += 25

    y += 10
    draw.text((20, y), f"Last Refreshed: {refreshed_at.isoformat()}", font=font, fill="black")
    img.save(CACHE_IMAGE_PATH)


# -------------------------------------------------------
# Core Logic
# -------------------------------------------------------
@app.post("/countries/refresh")
async def refresh_countries(db: AsyncSession = Depends(get_db)):
    """Fetch countries & exchange rates, then cache them."""
    countries_data = await fetch_external_json(COUNTRIES_API)
    exchange_data = await fetch_external_json(EXCHANGE_API)
    rates = exchange_data.get("rates", {})

    refreshed_at = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)

    try:
        async with db.begin():
            for c in countries_data:
                name = c.get("name")
                if not name:
                    continue
                currency_code = extract_currency_code(c)
                population = c.get("population", 0)
                region = c.get("region")
                capital = c.get("capital")
                flag_url = c.get("flag")
                exchange_rate = rates.get(currency_code) if currency_code else None
                estimated_gdp = (
                    compute_estimated_gdp(population, exchange_rate)
                    if exchange_rate
                    else (0.0 if not currency_code else None)
                )

                stmt = select(Country).where(func.lower(Country.name) == name.lower())
                existing = (await db.execute(stmt)).scalars().first()
                if existing:
                    existing.capital = capital
                    existing.region = region
                    existing.population = population
                    existing.currency_code = currency_code
                    existing.exchange_rate = exchange_rate
                    existing.estimated_gdp = estimated_gdp
                    existing.flag_url = flag_url
                    existing.last_refreshed_at = refreshed_at
                    db.add(existing)
                else:
                    db.add(
                        Country(
                            name=name,
                            capital=capital,
                            region=region,
                            population=population,
                            currency_code=currency_code,
                            exchange_rate=exchange_rate,
                            estimated_gdp=estimated_gdp,
                            flag_url=flag_url,
                            last_refreshed_at=refreshed_at,
                        )
                    )
    except SQLAlchemyError:
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})

    total = (await db.execute(select(func.count()).select_from(Country))).scalar_one()
    top5 = (
        (await db.execute(select(Country.name, Country.estimated_gdp)
                          .order_by(Country.estimated_gdp.desc().nullslast()).limit(5)))
        .all()
    )

    try:
        generate_summary_image(total, top5, refreshed_at)
    except Exception:
        print("⚠️ Failed to generate summary image")

    return {"message": "Refresh successful", "total_countries": total, "last_refreshed_at": refreshed_at.isoformat()}


@app.get("/countries", response_model=List[CountryOut])
async def list_countries(
    region: Optional[str] = Query(None),
    currency: Optional[str] = Query(None),
    sort: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """List countries with optional filters and sorting."""
    stmt = select(Country)
    if region:
        stmt = stmt.where(Country.region == region)
    if currency:
        stmt = stmt.where(Country.currency_code == currency)
    if sort == "gdp_desc":
        stmt = stmt.order_by(Country.estimated_gdp.desc().nullslast())
    else:
        stmt = stmt.order_by(Country.name.asc())
    stmt = stmt.limit(limit).offset(offset)
    rows = (await db.execute(stmt)).scalars().all()
    return rows


@app.get("/countries/{name}", response_model=CountryOut)
async def get_country(name: str, db: AsyncSession = Depends(get_db)):
    stmt = select(Country).where(func.lower(Country.name) == name.lower())
    country = (await db.execute(stmt)).scalars().first()
    if not country:
        raise HTTPException(status_code=404, detail={"error": "Country not found"})
    return country


@app.delete("/countries/{name}")
async def delete_country(name: str, db: AsyncSession = Depends(get_db)):
    stmt = select(Country).where(func.lower(Country.name) == name.lower())
    country = (await db.execute(stmt)).scalars().first()
    if not country:
        raise HTTPException(status_code=404, detail={"error": "Country not found"})
    async with db.begin():
        await db.delete(country)
    return {"message": "Deleted"}


@app.get("/status", response_model=StatusOut)
async def get_status(db: AsyncSession = Depends(get_db)):
    total = (await db.execute(select(func.count()).select_from(Country))).scalar_one()
    last = (await db.execute(select(func.max(Country.last_refreshed_at)))).scalar_one()
    return {"total_countries": total, "last_refreshed_at": last}


@app.get("/countries/image")
async def get_summary_image():
    if not os.path.exists(CACHE_IMAGE_PATH):
        return JSONResponse(status_code=404, content={"error": "Summary image not found"})
    return FileResponse(CACHE_IMAGE_PATH, media_type="image/png")


# -------------------------------------------------------
# Error Handlers
# -------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content=exc.detail)


@app.exception_handler(Exception)
async def global_exception_handler(_, __):
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


# -------------------------------------------------------
# End of File
# -------------------------------------------------------
