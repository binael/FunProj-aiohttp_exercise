"""
fun_proj.service.

This module provides asynchronous services for fetching and formatting
country and exchange rate data using `aiohttp`.

It integrates data from two public APIs:
    1. RestCountries API - provides country information such as name,
       capital, population, and currency.
    2. ExchangeRate API - provides current exchange rates relative to USD.

The module merges both datasets and formats them into a standardized
structure using the `formatted_country` utility function.

Typical usage example
---------------------
>>> import asyncio
>>> from fun_proj.service import get_formatted_countries_data
>>>
>>> data = asyncio.run(get_formatted_countries_data())
>>> print(data[0]["name"], data[0]["currency"])

Notes
-----
- This module is intended for use within FastAPI or other asynchronous
  Python web services.
- The function `get_formatted_countries_data` orchestrates the entire
  data retrieval and formatting workflow.
"""

from typing import Any

import aiohttp
from fastapi import HTTPException

from fun_proj.utils import formatted_country

# External API endpoints
COUNTRY_API_URL = "https://restcountries.com/v2/all?fields=name,capital,region,population,flag,currencies"
EXCHAGE_RATE_API_URL = "https://api.exchangerate-api.com/v4/latest/USD"


async def fetch_data(url: str, session: aiohttp.ClientSession) -> Any | None:
    """
    Fetch data asynchronously from a given URL using `aiohttp`.

    This coroutine performs a non-blocking HTTP GET request to the specified
    endpoint and returns the parsed JSON response. If the request fails or
    the remote server is unavailable, an appropriate `HTTPException` is raised.

    Parameters
    ----------
    url : str
        The endpoint URL to fetch data from.
    session : aiohttp.ClientSession
        The active aiohttp session used for making HTTP requests.

    Returns
    -------
    Any or None
        The JSON-decoded response from the remote API, or `None` if
        an error occurs during the request.

    Raises
    ------
    fastapi.HTTPException
        If the request fails due to network issues, unavailable services,
        or unexpected exceptions.

    Examples
    --------
    >>> import aiohttp, asyncio
    >>> from fun_proj.service import fetch_data
    >>>
    >>> async def main():
    ...     async with aiohttp.ClientSession() as session:
    ...         data = await fetch_data("https://restcountries.com/v2/all", session)
    ...         print(len(data))
    >>>
    >>> asyncio.run(main())
    """
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.json()
    except aiohttp.ClientError as er:
        raise HTTPException(
            status_code=503,
            detail={
                "error": str(er),
                "details": f"Could not fetch data from {url}",
            },
        ) from er
    except aiohttp.ClientResponse as er:
        raise HTTPException(
            status_code=503,
            detail={
                "error": str(er),
                "details": f"Could not fetch data from {url}",
            },
        ) from er
    except Exception as er:
        raise HTTPException(
            status_code=500,
            detail={"error": "Internal Server Error"},
        ) from er
    return None


async def get_formatted_countries_data() -> list[dict[str, Any]]:
    """
    Fetch and merge country and exchange rate data, then format the results.

    This coroutine retrieves all countries and their respective exchange
    rates from public APIs, merges the datasets, and returns a list of
    structured country dictionaries formatted via `formatted_country`.

    Returns
    -------
    list of dict
        A list of formatted country data objects, each containing country
        details such as name, population, and converted exchange rate.

    Raises
    ------
    fastapi.HTTPException
        If either the country data or the exchange rate data cannot be fetched.

    Examples
    --------
    >>> import asyncio
    >>> from fun_proj.service import get_formatted_countries_data
    >>>
    >>> async def main():
    ...     countries = await get_formatted_countries_data()
    ...     print(countries[0]["name"])
    >>>
    >>> asyncio.run(main())

    Notes
    -----
    - Each country object returned from this function is formatted using
      `fun_proj.utils.formatted_country`.
    - API calls are made concurrently using `aiohttp` for efficiency.
    """
    async with aiohttp.ClientSession() as session:
        countries_data = await fetch_data(COUNTRY_API_URL, session)
        exchange_rate_data = await fetch_data(EXCHAGE_RATE_API_URL, session)

        if not countries_data:
            raise HTTPException(
                status_code=500, detail="Failed to retrieve countries data"
            )
        if not exchange_rate_data:
            raise HTTPException(
                status_code=500, detail="Failed to retrieve exchange rate data"
            )

        exchange_rates = exchange_rate_data.get("rates", {})

        formatted_countries = [
            formatted_country(country, exchange_rates) for country in countries_data
        ]

        return formatted_countries
