"""
fun_proj.handlers.exceptions.

This module defines a custom exception handler for HTTP exceptions
within a FastAPI application.

The handler standardizes the JSON response structure for all raised
`HTTPException` instances, ensuring consistency across API responses.

If the exception detail is a dictionary, it is returned directly as
the response content. Otherwise, a structured JSON object with
generic error metadata is returned.

Typical usage example
---------------------
>>> from fastapi import FastAPI, HTTPException
>>> from fun_proj.handlers.exceptions import custom_http_exception_handler
>>> app = FastAPI()
>>> app.add_exception_handler(HTTPException, custom_http_exception_handler)
"""

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse


from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

async def custom_http_exception_handler(request: Request, exc: Exception):
    """
    Custom exception handler for FastAPI.

    Handles both FastAPI HTTPException and general Python exceptions.
    Returns structured JSON responses with appropriate status codes.
    """

    # Handle HTTPException (from FastAPI)
    if isinstance(exc, HTTPException):
        if isinstance(exc.detail, dict):
            return JSONResponse(status_code=exc.status_code, content=exc.detail)

        return JSONResponse(
            status_code=exc.status_code,
            content={"error": "HTTP Exception", "details": str(exc.detail)},
        )

    # Handle non-HTTP exceptions
    return JSONResponse(
        status_code=500,
        content={
            "error": exc.__class__.__name__,
            "details": str(exc),
            "path": str(request.url),
        },
    )
