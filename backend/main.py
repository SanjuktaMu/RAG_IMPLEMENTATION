from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import API_PREFIX, API_TITLE, API_VERSION, CORS_ORIGINS
from backend.routes.query import router as query_router
from backend.routes.upload import router as upload_router

app = FastAPI(title=API_TITLE, version=API_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["health"])
def health() -> dict:
    return {"status": "ok"}


app.include_router(upload_router, prefix=API_PREFIX)
app.include_router(query_router, prefix=API_PREFIX)
