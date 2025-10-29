import uvicorn
from fastapi import FastAPI

from fun_proj.handlers.exceptions import custom_http_exception_handler

app = FastAPI()
app.add_exception_handler(Exception, custom_http_exception_handler)

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )
