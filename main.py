from fastapi import FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.security.api_key import APIKeyHeader
from contextlib import asynccontextmanager

from server.utils.database import get_db



@asynccontextmanager
async def lifespan(_):
    yield

app = FastAPI(
    title="Hiago Docs",
    description="This is the documentation for Hiago's API.",
    version="1.0",
    basePath="/v1",
    lifespan=lifespan,
)




app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"],
)
origins=["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

@app.get("/")
async def app_startup():
    return {"status": "elasticsearch engine running"}


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "healthy"}


