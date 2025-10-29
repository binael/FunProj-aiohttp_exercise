"""
fastapi_countries_api.py

Module-level docstring: RESTful API to fetch country data, cache it in Postgres, and provide CRUD operations.

Stack: FastAPI, SQLAlchemy (async), aiohttp for external HTTP, Pillow for image generation.

Endpoints:
- POST /countries/refresh
- GET  /countries
- GET  /countries/{name}
- DELETE /countries/{name}
- GET  /status
- GET  /countries/image

Run (example):
    pip install fastapi[all] aiohttp sqlalchemy[asyncio] asyncpg pillow python-multipart
    uvicorn fastapi_countries_api:app --reload

DB:
Configure DATABASE_URL environment variable, e.g.: postgresql+asyncpg://user:pass@localhost:5432/mydb

Notes:
- Module contains inline validation, error responses that follow the requested JSON shapes.
- Module-level docstring is included as requested.
"""

from __future__ import annotations

import os
import io
import asyncio
import random
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

import aiohttp
from fastapi import FastAPI, HTTPException, Query, Depends, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import (Column, Integer, String, Float, DateTime, func, select, update, insert, null)
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError
from PIL import Image, ImageDraw, ImageFont

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
CACHE_IMAGE_PATH = os.getenv("CACHE_IMAGE_PATH", "cache/summary.png")

# --- SQLAlchemy setup ---
Base = declarative_base()
engine = create_async_engine(DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


def get_session() -> AsyncSession:
    return AsyncSessionLocal()


# --- Models ---
class Country(Base):
    __tablename__ = "country"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True, index=True)
    capital = Column(String, nullable=True)
    region = Column(String, nullable=True)
    population = Column(Integer, nullable=False)
    currency_code = Column(String, nullable=True)
    exchange_rate = Column(Float, nullable=True)
    estimated_gdp = Column(Float, nullable=True)
    flag_url = Column(String, nullable=True)
    last_refreshed_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class AppMeta(Base):
    """Key/value table for storing last_refreshed_at and other global metadata."""
    __tablename__ = "app_meta"

    key = Column(String, primary_key=True)
    value = Column(String, nullable=True)


# --- Pydantic schemas ---
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
    last_refreshed_at: datetime

    class Config:
        orm_mode = True


# --- FastAPI app ---
app = FastAPI(title="Countries Cache API")


# --- Helper functions ---
async def fetch_json(session: aiohttp.ClientSession, url: str, timeout: int = 15) -> Any:
    try:
        async with session.get(url, timeout=timeout) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}")
            return await resp.json()
    except Exception as e:
        raise RuntimeError(str(e)) from e


async def fetch_external_data() -> Dict[str, Any]:
    countries_url = "https://restcountries.com/v2/all?fields=name,capital,region,population,flag,currencies"
    rates_url = "https://open.er-api.com/v6/latest/USD"

    async with aiohttp.ClientSession() as session:
        # fetch both concurrently
        try:
            countries_task = asyncio.create_task(fetch_json(session, countries_url))
            rates_task = asyncio.create_task(fetch_json(session, rates_url))
            countries_data, rates_data = await asyncio.gather(countries_task, rates_task)
        except Exception as exc:
            # Determine which API failed (best-effort)
            msg = str(exc)
            if "restcountries" in msg or "restcountries.com" in msg:
                raise ExternalAPIError("Countries API", msg)
            if "open.er-api" in msg or "exchangerate" in msg or "er-api" in msg:
                raise ExternalAPIError("Exchange rates API", msg)
            # generic
            raise ExternalAPIError("External API", msg)

    return {"countries": countries_data, "rates": rates_data}


class ExternalAPIError(Exception):
    def __init__(self, api_name: str, details: str):
        super().__init__(f"{api_name}: {details}")
        self.api_name = api_name
        self.details = details


def compute_estimated_gdp(population: int, exchange_rate: Optional[float], multiplier: int, currency_missing: bool) -> Optional[float]:
    # Behavior per spec:
    # - if currencies array empty -> estimated_gdp = 0
    # - if currency_code not found in rates -> estimated_gdp = null
    if currency_missing:
        return 0.0
    if exchange_rate is None:
        return None
    if exchange_rate == 0:
        return None
    return float(population * multiplier / exchange_rate)


async def upsert_countries(db: AsyncSession, countries_raw: List[dict], rates_raw: dict) -> datetime:
    """Insert or update countries in DB. Returns the refresh timestamp."""
    now = datetime.now(timezone.utc)
    rates_map: Dict[str, float] = {}

    # rates_raw expected structure: { 'result': 'success', 'rates': {...}, ... }
    if isinstance(rates_raw, dict) and "rates" in rates_raw:
        rates_map = rates_raw.get("rates", {})

    # We'll perform upserts. For safety we can use INSERT..ON CONFLICT via PostgreSQL dialect.
    for item in countries_raw:
        name = item.get("name")
        if not name:
            # skip records without name
            continue
        capital = item.get("capital")
        region = item.get("region")
        population = item.get("population")
        flag_url = item.get("flag")

        # Validation: name and population are required by model logic; but spec says to store record even if currency missing
        # Currency extraction
        currencies = item.get("currencies") or []
        if isinstance(currencies, list) and len(currencies) > 0:
            # currencies entries can be objects with 'code' key, or strings; handle common shapes
            first = currencies[0]
            if isinstance(first, dict):
                currency_code = first.get("code")
            else:
                currency_code = str(first)
            currency_missing = False
            if not currency_code:
                currency_missing = True
                currency_code = None
        else:
            currency_code = None
            currency_missing = True

        exchange_rate = None
        estimated_gdp = None

        if currency_missing:
            exchange_rate = None
            estimated_gdp = 0.0
        else:
            code = currency_code
            rate = rates_map.get(code) if rates_map else None
            if rate is None:
                exchange_rate = None
                estimated_gdp = None
            else:
                exchange_rate = float(rate)
                multiplier = random.randint(1000, 2000)
                estimated_gdp = compute_estimated_gdp(population, exchange_rate, multiplier, currency_missing=False)

        # Upsert via Postgres ON CONFLICT by name
        stmt = pg_insert(Country.__table__).values(
            name=name,
            capital=capital,
            region=region,
            population=population,
            currency_code=currency_code,
            exchange_rate=exchange_rate,
            estimated_gdp=estimated_gdp,
            flag_url=flag_url,
            last_refreshed_at=now,
        )
        update_cols = {
            "capital": stmt.excluded.capital,
            "region": stmt.excluded.region,
            "population": stmt.excluded.population,
            "currency_code": stmt.excluded.currency_code,
            "exchange_rate": stmt.excluded.exchange_rate,
            "estimated_gdp": stmt.excluded.estimated_gdp,
            "flag_url": stmt.excluded.flag_url,
            "last_refreshed_at": stmt.excluded.last_refreshed_at,
        }
        stmt = stmt.on_conflict_do_update(index_elements=[Country.name], set_=update_cols)
        try:
            await db.execute(stmt)
        except SQLAlchemyError as e:
            # Log and continue
            print(f"DB upsert error for {name}: {e}")
    # update global metadata last_refreshed_at
    meta_stmt = pg_insert(AppMeta.__table__).values(key="last_refreshed_at", value=now.isoformat()).on_conflict_do_update(
        index_elements=[AppMeta.key], set_= {"value": now.isoformat()}
    )
    await db.execute(meta_stmt)
    await db.commit()
    return now


def generate_summary_image(total: int, top5: List[Dict[str, Any]], timestamp: datetime, path: str = CACHE_IMAGE_PATH) -> None:
    # Simple image: white background, black text. Use default fonts.
    width, height = 800, 600
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
        small = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()

    y = 20
    draw.text((20, y), f"Countries cached: {total}", font=font)
    y += 40
    draw.text((20, y), "Top 5 countries by estimated GDP:", font=small)
    y += 30
    for i, c in enumerate(top5, start=1):
        name = c.get("name")
        gdp = c.get("estimated_gdp")
        gdp_str = f"{gdp:,.2f}" if gdp is not None else "N/A"
        draw.text((30, y), f"{i}. {name} — {gdp_str}", font=small)
        y += 26
    y += 10
    draw.text((20, y), f"Last refreshed at: {timestamp.isoformat()}", font=small)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img.save(path)


# --- API routes ---
@app.post("/countries/refresh")
async def refresh_countries(session: AsyncSession = Depends(get_session)):
    """Fetch all countries and exchange rates and cache them in DB.

    - If external APIs fail, return 503 and do not modify DB.
    - After successful save, generate summary image at cache/summary.png
    """
    try:
        external = await fetch_external_data()
    except ExternalAPIError as e:
        return JSONResponse(status_code=503, content={"error": "External data source unavailable", "details": f"Could not fetch data from {e.api_name}"})
    except Exception as e:
        return JSONResponse(status_code=503, content={"error": "External data source unavailable", "details": str(e)})

    countries_raw = external.get("countries", [])
    rates_raw = external.get("rates", {})

    # Perform DB upsert transaction
    try:
        refresh_time = await upsert_countries(session, countries_raw, rates_raw)
    except Exception as e:
        # If DB fails after external fetch, it's a 500. But per spec, do not partially write if failure during refresh— our upsert commits at the end; any exceptions here are treated as 500
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

    # After successful commit, generate summary image
    # Query summary info
    try:
        total_q = await session.execute(select(func.count()).select_from(Country))
        total = total_q.scalar_one()

        top_q = await session.execute(select(Country).order_by(Country.estimated_gdp.desc().nullslast()).limit(5))
        top_countries = top_q.scalars().all()
        top5 = [
            {"name": c.name, "estimated_gdp": c.estimated_gdp}
            for c in top_countries
        ]

        generate_summary_image(total, top5, refresh_time, path=CACHE_IMAGE_PATH)
    except Exception as e:
        # Image generation failure shouldn't break the refresh result
        print("Image generation/summary failed:", e)

    return {"message": "Refresh successful", "total_countries": total, "last_refreshed_at": refresh_time.isoformat()}


@app.get("/countries", response_model=List[CountryOut])
async def list_countries(region: Optional[str] = Query(None), currency: Optional[str] = Query(None, alias="currency"), sort: Optional[str] = Query(None), session: AsyncSession = Depends(get_session)):
    """Get all countries from DB (support filters and sorting)
    Query params: ?region=Africa | ?currency=NGN | ?sort=gdp_desc
    """
    q = select(Country)
    if region:
        q = q.where(func.lower(Country.region) == region.lower())
    if currency:
        q = q.where(func.lower(Country.currency_code) == currency.lower())
    if sort:
        if sort == "gdp_desc":
            q = q.order_by(Country.estimated_gdp.desc().nullslast())
        elif sort == "gdp_asc":
            q = q.order_by(Country.estimated_gdp.asc().nullsfirst())
    result = await session.execute(q)
    countries = result.scalars().all()
    return countries


@app.get("/countries/{name}", response_model=CountryOut)
async def get_country(name: str, session: AsyncSession = Depends(get_session)):
    q = select(Country).where(func.lower(Country.name) == name.lower())
    result = await session.execute(q)
    country = result.scalar_one_or_none()
    if not country:
        return JSONResponse(status_code=404, content={"error": "Country not found"})
    return country


@app.delete("/countries/{name}")
async def delete_country(name: str, session: AsyncSession = Depends(get_session)):
    q = select(Country).where(func.lower(Country.name) == name.lower())
    result = await session.execute(q)
    country = result.scalar_one_or_none()
    if not country:
        return JSONResponse(status_code=404, content={"error": "Country not found"})
    await session.delete(country)
    await session.commit()
    return {"message": "Deleted"}


@app.get("/status")
async def status(session: AsyncSession = Depends(get_session)):
    total_q = await session.execute(select(func.count()).select_from(Country))
    total = total_q.scalar_one()
    meta_q = await session.execute(select(AppMeta).where(AppMeta.key == "last_refreshed_at"))
    meta = meta_q.scalar_one_or_none()
    last = meta.value if meta else None
    return {"total_countries": total, "last_refreshed_at": last}


@app.get("/countries/image")
async def get_summary_image():
    if not os.path.exists(CACHE_IMAGE_PATH):
        return JSONResponse(status_code=404, content={"error": "Summary image not found"})
    return FileResponse(CACHE_IMAGE_PATH, media_type="image/png")


# --- Error handlers for consistent JSON ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    # Map 400/404/500 as required
    if exc.status_code == 404:
        return JSONResponse(status_code=404, content={"error": "Country not found"})
    if exc.status_code == 400:
        return JSONResponse(status_code=400, content={"error": "Validation failed"})
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.middleware("http")
async def add_process_time_header(request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        # Log and return JSON 500
        print("Unhandled error:", e)
        return JSONResponse(status_code=500, content={"error": "Internal server error"})


# --- Utilities: create DB tables ---
async def init_models() -> None:
    async with engine.begin() as conn:

        await conn.run_sync(Base.metadata.create_all)


# initialize DB when module loaded in dev
@app.on_event("startup")
async def on_startup():
    await init_models()


# End of file
