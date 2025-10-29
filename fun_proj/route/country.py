# fun_proj/routes/countries.py
"""
Routes for country caching and retrieval.

Endpoints:
- POST   /countries/refresh
- GET    /countries
- GET    /countries/{name}
- DELETE /countries/{name}
- GET    /status
- GET    /countries/image
"""
from __future__ import annotations

import aiohttp
import asyncio
import os
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import select, update, delete, func, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession

from fun_proj.model.country import Country, AssynSessionLocal  # uses your model file names
from fun_proj.utils import hash_string, get_estimated_gdp  # utils you included
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont  # pillow for image creation

# External API endpoints (match your spec)
COUNTRIES_API = "https://restcountries.com/v2/all?fields=name,capital,region,population,flag,currencies"
EXCHANGE_API = "https://open.er-api.com/v6/latest/USD"

router = APIRouter(prefix="/countries", tags=["countries"])

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
SUMMARY_IMAGE_PATH = CACHE_DIR / "summary.png"


# --- Helper: DB dependency (uses your async session maker) ---
async def get_session() -> AsyncSession:
    async with AssynSessionLocal() as session:
        yield session


# --- Helper: consistent JSON HTTPException detail shapes ---
def bad_request(detail: Dict[str, Any]):
    raise HTTPException(status_code=400, detail={"error": "Validation failed", "details": detail})


def not_found():
    raise HTTPException(status_code=404, detail={"error": "Country not found"})


def external_unavailable(api_name: str):
    raise HTTPException(
        status_code=503,
        detail={"error": "External data source unavailable", "details": f"Could not fetch data from {api_name}"},
    )


# --- Helper: fetch JSON with timeout and basic error handling ---
async def _fetch_json(url: str, session: aiohttp.ClientSession, api_name: str, timeout: int = 20) -> Any:
    try:
        async with session.get(url, timeout=timeout) as resp:
            resp.raise_for_status()
            return await resp.json()
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        # bubble as service unavailable with consistent shape
        raise HTTPException(
            status_code=503,
            detail={"error": "External data source unavailable", "details": f"Could not fetch data from {api_name}"},
        ) from e


# --- Helper: build country row dict according to spec ---
def _build_country_record(country_raw: Dict[str, Any], rates: Dict[str, float], refreshed_at: datetime) -> Dict[str, Any]:
    """
    Build dictionary for DB insert/update following the rules:
    - currency_code: first currency code if present else None
    - if currencies empty: exchange_rate=None, estimated_gdp=0
    - if currency not found in rates: exchange_rate=None, estimated_gdp=None
    - otherwise compute estimated_gdp using a fresh random multiplier (1000-2000)
    """
    name = country_raw.get("name")
    capital = country_raw.get("capital")
    region = country_raw.get("region")
    population = country_raw.get("population") or 0
    currencies = country_raw.get("currencies") or []

    currency_code = None
    exchange_rate = None
    estimated_gdp = None

    if currencies and len(currencies) > 0:
        currency_code = currencies[0].get("code")
        # if code present, try lookup in rates
        if currency_code:
            # open.er-api returns rates mapping where USD is base; rates are floats
            exchange_rate = rates.get(currency_code)
            if exchange_rate is None:
                # currency code not found in exchange API per spec:
                exchange_rate = None
                estimated_gdp = None
            else:
                # compute estimated_gdp with new random multiplier between 1000 and 2000
                multiplier = random.randint(1000, 2000)
                # Avoid division by zero defensively
                try:
                    estimated_gdp = multiplier * population / float(exchange_rate)
                except Exception:
                    estimated_gdp = None
        else:
            # currency entry present but no code field
            currency_code = None
            exchange_rate = None
            estimated_gdp = 0
    else:
        # no currencies array or empty per spec
        currency_code = None
        exchange_rate = None
        estimated_gdp = 0

    # compute sha256 hash (use name lowercased to keep match deterministic)
    sha = hash_string((name or "").lower())

    return {
        "name": name,
        "capital": capital,
        "region": region,
        "population": population,
        "currency_code": currency_code,
        "exchange_rate": exchange_rate,
        "estimated_gdp": estimated_gdp,
        "flag_url": country_raw.get("flag"),
        "last_refreshed_at": refreshed_at,
        "sha256_hash": sha,
    }


# --- POST /countries/refresh ---
@router.post("/refresh", status_code=200)
async def refresh_countries(session: AsyncSession = Depends(get_session)):
    """
    Fetch countries and exchange rates from external APIs, then upsert into DB.
    If either external API fails, return 503 and do not modify DB.
    After a successful refresh, generate cache/summary.png with:
      - total number of countries
      - top 5 countries by estimated_gdp
      - timestamp
    """
    # Fetch external APIs concurrently (fail early on any error)
    async with aiohttp.ClientSession() as http:
        try:
            countries_task = _fetch_json(COUNTRIES_API, http, "RestCountries API")
            exchange_task = _fetch_json(EXCHANGE_API, http, "ExchangeRate API")
            countries_data, exchange_data = await asyncio.gather(countries_task, exchange_task)
        except HTTPException as he:
            # forward consistent 503
            raise he

    # pull rates safely
    rates = exchange_data.get("rates") if isinstance(exchange_data, dict) else {}
    if rates is None:
        rates = {}

    # We must guarantee we don't modify DB if anything fails during processing/committing.
    refreshed_at = datetime.now(timezone.utc)

    # Use a transaction; commit only after all upserts succeed
    async with session.begin():  # will rollback on exception
        try:
            # iterate countries and upsert
            for country_raw in countries_data:
                record = _build_country_record(country_raw, rates, refreshed_at)

                # Validate required fields per spec: name, population, currency_code required.
                # BUT spec allows currency_code to be null if currencies empty; for such cases estimated_gdp=0 and currency_code is null.
                # Here we only require 'name' and 'population' non-null
                if not record["name"]:
                    bad_request({"name": "is required"})
                if record["population"] is None:
                    bad_request({"population": "is required"})

                # Upsert logic: match by name case-insensitive
                q = select(Country).where(func.lower(Country.name) == (record["name"] or "").lower()).limit(1)
                existing = (await session.execute(q)).scalars().first()
                if existing:
                    # update all fields including recalculated estimated_gdp and exchange_rate etc.
                    # Map update values explicitly for clarity
                    stmt = (
                        update(Country)
                        .where(Country.id == existing.id)
                        .values(
                            capital=record["capital"],
                            region=record["region"],
                            population=record["population"],
                            currency_code=record["currency_code"],
                            exchange_rate=record["exchange_rate"],
                            estimated_gdp=record["estimated_gdp"],
                            flag_url=record["flag_url"],
                            last_refreshed_at=record["last_refreshed_at"],
                            sha256_hash=record["sha256_hash"],
                        )
                    )
                    await session.execute(stmt)
                else:
                    # insert new record
                    new = Country(
                        name=record["name"],
                        sha256_hash=record["sha256_hash"],
                        capital=record["capital"] or "",
                        population=record["population"],
                        currency_code=record["currency_code"] or "",
                        exchange_rate=record["exchange_rate"] if record["exchange_rate"] is not None else 0.0,
                        estimated_gdp=record["estimated_gdp"] if record["estimated_gdp"] is not None else 0.0,
                        flag_url=record["flag_url"],
                        last_refreshed_at=record["last_refreshed_at"],
                    )
                    session.add(new)
            # commit happens automatically when exiting session.begin() on success
        except HTTPException:
            # re-raise HTTPExceptions (these will trigger rollback)
            raise
        except Exception as e:
            # any other error: rollback and return 500
            raise HTTPException(status_code=500, detail={"error": "Internal server error", "details": str(e)}) from e

    # After DB commit succeeded, generate summary image
    try:
        await _generate_summary_image(session)
    except Exception:
        # image generation should not break the endpoint result; log and continue
        # but return 500 would be too strict; instead return success for refresh and note image failed
        return JSONResponse(
            status_code=200,
            content={"message": "Refresh successful, but summary image generation failed."},
        )

    return {"message": "Refresh successful"}


# --- generate summary image ---
async def _generate_summary_image(session: AsyncSession):
    """
    Generate cache/summary.png containing:
      - total number of countries
      - top 5 countries by estimated_gdp
      - timestamp of last refresh
    Overwrite SUMMARY_IMAGE_PATH.
    """
    # Query total and top 5
    total_q = select(func.count(Country.id))
    total = (await session.execute(total_q)).scalar_one()

    top_q = (
        select(Country.name, Country.estimated_gdp)
        .where(Country.estimated_gdp.isnot(None))
        .order_by(desc(Country.estimated_gdp))
        .limit(5)
    )
    rows = (await session.execute(top_q)).all()
    top_list = [{"name": r[0], "estimated_gdp": float(r[1]) if r[1] is not None else None} for r in rows]

    last_ts_q = select(func.max(Country.last_refreshed_at))
    last_ts = (await session.execute(last_ts_q)).scalar_one()
    last_ts_str = last_ts.isoformat() if last_ts else datetime.now(timezone.utc).isoformat()

    # Compose a simple image
    width, height = 800, 600
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Use a basic font if available; fallback to default
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=18)
        title_font = ImageFont.truetype("DejaVuSans.ttf", size=24)
    except Exception:
        font = ImageFont.load_default()
        title_font = font

    x = 40
    y = 40
    draw.text((x, y), "Countries Summary", font=title_font, fill=(0, 0, 0))
    y += 40
    draw.text((x, y), f"Total countries: {total}", font=font, fill=(0, 0, 0))
    y += 30
    draw.text((x, y), f"Last refreshed at: {last_ts_str}", font=font, fill=(0, 0, 0))
    y += 40
    draw.text((x, y), "Top 5 countries by estimated GDP:", font=font, fill=(0, 0, 0))
    y += 30
    for idx, item in enumerate(top_list, start=1):
        gdp_str = f"{item['estimated_gdp']:.2f}" if item["estimated_gdp"] is not None else "N/A"
        draw.text((x, y), f"{idx}. {item['name']} — {gdp_str}", font=font, fill=(0, 0, 0))
        y += 24

    # Save
    SUMMARY_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    img.save(SUMMARY_IMAGE_PATH)


# --- GET /countries (list with filters + sorting) ---
@router.get("", response_model=List[Dict[str, Any]])
async def list_countries(
    region: Optional[str] = Query(None),
    currency: Optional[str] = Query(None),
    sort: Optional[str] = Query(None, description="gdp_desc or gdp_asc"),
    session: AsyncSession = Depends(get_session),
):
    q = select(
        Country.id,
        Country.name,
        Country.capital,
        Country.region,
        Country.population,
        Country.currency_code,
        Country.exchange_rate,
        Country.estimated_gdp,
        Country.flag_url,
        Country.last_refreshed_at,
    )

    if region:
        q = q.where(func.lower(Country.region) == region.lower())
    if currency:
        q = q.where(func.lower(Country.currency_code) == currency.lower())

    if sort == "gdp_desc":
        q = q.order_by(desc(Country.estimated_gdp))
    elif sort == "gdp_asc":
        q = q.order_by(asc(Country.estimated_gdp))
    else:
        q = q.order_by(Country.name)

    rows = (await session.execute(q)).all()
    result = []
    for row in rows:
        result.append(
            {
                "id": row[0],
                "name": row[1],
                "capital": row[2],
                "region": row[3],
                "population": row[4],
                "currency_code": row[5],
                "exchange_rate": float(row[6]) if row[6] is not None else None,
                "estimated_gdp": float(row[7]) if row[7] is not None else None,
                "flag_url": row[8],
                "last_refreshed_at": row[9].isoformat() if row[9] else None,
            }
        )
    return result


# --- GET /countries/{name} ---
@router.get("/{name}")
async def get_country(name: str, session: AsyncSession = Depends(get_session)):
    q = select(
        Country.id,
        Country.name,
        Country.capital,
        Country.region,
        Country.population,
        Country.currency_code,
        Country.exchange_rate,
        Country.estimated_gdp,
        Country.flag_url,
        Country.last_refreshed_at,
    ).where(func.lower(Country.name) == name.lower())

    row = (await session.execute(q)).first()
    if not row:
        not_found()

    row = row[0]
    return {
        "id": row.id,
        "name": row.name,
        "capital": row.capital,
        "region": row.region,
        "population": row.population,
        "currency_code": row.currency_code if row.currency_code != "" else None,
        "exchange_rate": float(row.exchange_rate) if row.exchange_rate is not None else None,
        "estimated_gdp": float(row.estimated_gdp) if row.estimated_gdp is not None else None,
        "flag_url": row.flag_url,
        "last_refreshed_at": row.last_refreshed_at.isoformat() if row.last_refreshed_at else None,
    }


# --- DELETE /countries/{name} ---
@router.delete("/{name}", status_code=200)
async def delete_country(name: str, session: AsyncSession = Depends(get_session)):
    q = select(Country).where(func.lower(Country.name) == name.lower()).limit(1)
    existing = (await session.execute(q)).scalars().first()
    if not existing:
        not_found()
    stmt = delete(Country).where(Country.id == existing.id)
    await session.execute(stmt)
    await session.commit()
    return {"message": "Country deleted"}


# --- GET /status ---
@router.get("/status")
async def status(session: AsyncSession = Depends(get_session)):
    total_q = select(func.count(Country.id))
    total = (await session.execute(total_q)).scalar_one()
    last_ts_q = select(func.max(Country.last_refreshed_at))
    last_ts = (await session.execute(last_ts_q)).scalar_one()
    return {"total_countries": total, "last_refreshed_at": last_ts.isoformat() if last_ts else None}


# --- GET /countries/image ---
@router.get("/image")
async def get_image():
    if not SUMMARY_IMAGE_PATH.exists():
        return JSONResponse(status_code=404, content={"error": "Summary image not found"})
    return FileResponse(SUMMARY_IMAGE_PATH, media_type="image/png", filename="summary.png")
