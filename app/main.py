import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.conversations import router as conversations_router
from app.core.config import ConfigurationError
from app.db.session import close_db, init_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        await init_db()
    except ConfigurationError:
        logger.warning("Skipping database initialization because DATABASE_URL is not configured")
    try:
        yield
    finally:
        await close_db()


app = FastAPI(
    title="AstralAI API",
    description="基于 FastAPI 与 LangChain 的最基础 AI 聊天后端。",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(chat_router)
app.include_router(conversations_router)


@app.get("/")
async def root():
    return {"message": "AstralAI is running"}
