'''
This file contains the FastAPI module.
'''

from os import getenv
import uvicorn
from fastapi import FastAPI, responses, Response, status

from src.api import router as api_router

app = FastAPI(title="Projeto ML API", version="1")
# Add middleware for safety or monetization

# App routes
app.include_router(api_router)

# Default /docs
@app.get("/", include_in_schema=False)
def home():
    return responses.RedirectResponse("/docs", 301)

# Healthcheck
@app.get("/health", include_in_schema=False)
def healthcheck():
    message = "Estou saudavel"
    return Response(content=message, status_code=status.HTTP_200_OK)

#----------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        app=getenv("APP_MODULE", "main:app"),
        host=getenv("HOST", "0.0.0.0"),
        port=getenv("PORT", "8000"),
        workers=getenv("MAX_WORKERS", "6"),
        timeout_keep_alive=getenv("KEEP_ALIVE", "900")
    )
