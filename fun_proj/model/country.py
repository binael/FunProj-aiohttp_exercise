"""
fun_proj.model.country.

This module defines the SQLAlchemy ORM models, asynchronous database engine,
and initialization utilities for the HNG FastAPI service.

It provides a declarative base class (`Base`), an example ORM model (`Country`),
and an asynchronous database connection setup for efficient non-blocking I/O
operations. The configuration is automatically loaded from environment variables
defined in a `.env` file.

Typical usage example
---------------------
>>> import asyncio
>>> from hng.model.analyser import create_db_and_tables
>>> asyncio.run(create_db_and_tables())

Notes
-----
- This module uses `sqlalchemy.ext.asyncio` for asynchronous database operations.
- Ensure that the environment variable `DATABASE_URL` is correctly set before
  initializing the engine.
- The model `Country` demonstrates how to define unique constraints, timestamp
  columns, and type annotations with SQLAlchemy 2.0.

Attributes
----------
DATABASE_URL : str
    The database connection string loaded from environment variables.
engine : sqlalchemy.ext.asyncio.AsyncEngine
    The asynchronous database engine connected to the specified `DATABASE_URL`.
AssynSessionLocal : sqlalchemy.ext.asyncio.async_sessionmaker
    A factory for creating asynchronous SQLAlchemy sessions.
"""

import os
from dotenv import load_dotenv
from sqlalchemy import DateTime, func, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# Load environment variables
load_dotenv()

# Database Configuration
DATABASE_URL: str = os.environ.get("DATABASE_URL")


class Base(DeclarativeBase):
    """
    Base declarative class for all ORM models.

    This class serves as the base for defining ORM models using SQLAlchemy's
    declarative mapping system. All application models should inherit from
    this class to automatically include metadata and table definitions.

    Notes
    -----
    The `DeclarativeBase` class from SQLAlchemy 2.0 provides native type
    annotations for mapped attributes.
    """
    pass


class Country(Base):
    """
    ORM model representing a country and its related economic data.

    The `Country` class defines a SQLAlchemy model that captures core
    country-level attributes including population, currency, GDP, and
    exchange rate. It enforces a unique SHA-256 hash constraint to ensure
    data integrity and prevent duplicate records.

    Attributes
    ----------
    id : int
        Primary key identifier for the country record.
    name : str
        Official name of the country. Must be unique.
    sha256_hash : str
        Unique SHA-256 hash representing the country record.
    capital : str
        Name of the country's capital city.
    population : int
        Total population count of the country.
    currency_code : str
        ISO 4217 currency code (e.g., 'USD', 'NGN').
    exchange_rate : float
        Exchange rate of the country's currency relative to a base currency.
    estimated_gdp : float
        Estimated Gross Domestic Product (GDP) of the country.
    created_at : datetime
        Timestamp indicating when the record was created.

    Notes
    -----
    The model defines a unique constraint on `sha256_hash` via
    `uq_country_sha256_hash` to prevent duplicate entries.
    """

    __tablename__ = "country"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(nullable=False, unique=True)
    sha256_hash: Mapped[str] = mapped_column(nullable=False, unique=True)
    capital: Mapped[str] = mapped_column(nullable=False)
    population: Mapped[int] = mapped_column(nullable=False)
    currency_code: Mapped[str] = mapped_column(nullable=False)
    exchange_rate: Mapped[float] = mapped_column(nullable=False)
    estimated_gdp: Mapped[float] = mapped_column(nullable=False)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("now()"),
        nullable=False,
    )

    __table_args__ = (UniqueConstraint("sha256_hash", name="uq_country_sha256_hash"),)


# Asynchronous engine and session setup
engine: AsyncEngine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
)

AssynSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def create_db_and_tables() -> None:
    """
    Create all database tables defined in the SQLAlchemy metadata.

    This coroutine initializes the database schema by running
    `Base.metadata.create_all` within an asynchronous database connection.

    Returns
    -------
    None
        This function performs schema creation but does not return a value.

    Raises
    ------
    sqlalchemy.exc.SQLAlchemyError
        If there is an error creating the database schema or connecting
        to the database.

    Examples
    --------
    >>> import asyncio
    >>> from hng.model.analyser import create_db_and_tables
    >>> asyncio.run(create_db_and_tables())
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
