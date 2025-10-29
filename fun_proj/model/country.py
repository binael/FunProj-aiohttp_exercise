"""
fun_proj.model.country.

This module defines the SQLAlchemy ORM models, asynchronous database engine,
and initialization utilities for the FastAPI service.

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
from sqlalchemy import DateTime, UniqueConstraint, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

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
    __tablename__ = "country"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(nullable=False, unique=True)
    sha256_hash: Mapped[str] = mapped_column(nullable=False, unique=True)
    capital: Mapped[str] = mapped_column(nullable=False)
    region: Mapped[str] = mapped_column(nullable=True)
    population: Mapped[int] = mapped_column(nullable=False)
    currency_code: Mapped[str] = mapped_column(nullable=True)
    exchange_rate: Mapped[float] = mapped_column(nullable=True)
    estimated_gdp: Mapped[float] = mapped_column(nullable=True)
    flag_url: Mapped[str] = mapped_column(nullable=True)
    last_refreshed_at: Mapped[DateTime] = mapped_column(
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
    Drop and recreate all database tables defined in the SQLAlchemy metadata.

    This coroutine ensures a clean database schema by first removing all
    existing tables and then reinitializing them. It’s ideal for development
    and testing environments where data persistence is not required.

    ⚠️ Warning
    ----------
    This function permanently deletes all existing data in the database.
    Use only in non-production environments.

    Returns
    -------
    None
        This function performs schema recreation but does not return a value.

    Raises
    ------
    sqlalchemy.exc.SQLAlchemyError
        If there is an error dropping or creating tables, or connecting
        to the database.

    Examples
    --------
    >>> import asyncio
    >>> from fun_proj.model.country import create_db_and_tables
    >>> asyncio.run(create_db_and_tables())
    """
    async with engine.begin() as conn:
        # Drop all tables first
        await conn.run_sync(Base.metadata.drop_all)

        # Then recreate all tables
        await conn.run_sync(Base.metadata.create_all)

