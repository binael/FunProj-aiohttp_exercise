from typing import Union
import hashlib
import random
from typing import Any, Dict
from datetime import datetime

def hash_string(input_string: str) -> str:
    """Returns the SHA-256 hash of the given input string."""
    sha256_hash = hashlib.sha256()
    sha256_hash.update(input_string.encode('utf-8'))
    return sha256_hash.hexdigest()

def get_estimated_gdp(population: int, exchange_rate: float) -> float:
    return random.randint(1000, 2000) * population / exchange_rate

def format_country(updated_at: datetime, country: Union[Dict[str, Any], None] = None, exchange_rate: Union[Dict[str, Any], None] = None) -> Dict[str, Any]:
    name = country.get("name")
    capital = country.get("capital")
    region = country.get("region")
    population = country.get("population")
    currency = country.get("currencies", None)
    currency_code = None
    exchange_rate = None
    estimated_gdp = 0
    if currency is not None:
        currency_code = currency[0].get("code", None)
        if currency_code is not None:
            exchange_rate = exchange_rate.get(currency_code)
            estimated_gdp = get_estimated_gdp(population, exchange_rate)
    flag_url = country.get("flag")
    last_refreshed_at = updated_at

    return {
        "name" : name,
        "capital" : capital,
        "region" : region,
        "population" : population,
        "currency_code" : currency_code,
        "exchange_rate" : exchange_rate,
        "estimated_gdp" : estimated_gdp,
        "flag_url" : flag_url,
        "last_refreshed_at" : last_refreshed_at
    }



