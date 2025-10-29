
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fun_proj.model.country import create_db_and_tables


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """
        Lifespan context manager for startup and shutdown tasks.

        Runs before the app starts serving requests and after it shuts down.

        Parameters
        ----------
        app : FastAPI
            The FastAPI application instance.

        Yields
        ------
        None
            Control back to the FastAPI event loop.
        """
        await create_db_and_tables()
        yield
        # Add cleanup or background shutdown tasks here if needed.

    app = FastAPI(
        title="Fun Project Aiohttp Exercise",
        lifespan=lifespan,
    )

    # Configure CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Consider restricting in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app
