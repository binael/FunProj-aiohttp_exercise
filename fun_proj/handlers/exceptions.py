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


async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """
    Handle `HTTPException` instances with a consistent JSON response format.

    This asynchronous function serves as a FastAPI exception handler that
    captures raised `HTTPException` objects and returns structured JSON
    responses suitable for RESTful APIs. If the exception's `detail` field
    is already a dictionary, it is used directly as the response body;
    otherwise, a standardized format containing the error message is returned.

    Parameters
    ----------
    request : fastapi.Request
        The incoming HTTP request that triggered the exception.
    exc : fastapi.HTTPException
        The HTTP exception instance containing the status code and detail.

    Returns
    -------
    fastapi.responses.JSONResponse
        A JSON response containing the appropriate HTTP status code and
        structured error information.

    Examples
    --------
    >>> from fastapi import FastAPI, HTTPException
    >>> from fun_proj.handlers.exceptions import custom_http_exception_handler
    >>>
    >>> app = FastAPI()
    >>> app.add_exception_handler(HTTPException, custom_http_exception_handler)
    >>>
    >>> @app.get("/items/{item_id}")
    ... async def read_item(item_id: int):
    ...     if item_id < 0:
    ...         raise HTTPException(status_code=400, detail="Invalid ID")
    ...     return {"item_id": item_id}

    Notes
    -----
    - This handler should be registered using `app.add_exception_handler`.
    - It improves API client experience by providing predictable
      error response structures.
    """
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)

    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "HTTP Exception", "details": str(exc.detail)},
    )
