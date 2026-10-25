
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router
from backend.app.conversation import router as conversation_router
from backend.app.memory_bridge import router as memory_bridge_router
from backend.app.streaming import router as streaming_router
from backend.app.config import Settings, get_settings
from backend.app.context.processor import ContextProcessor
from backend.app.memory.retrieval import MemoryRetriever
from backend.app.reasoning.client import ReasoningClient
from backend.app.storage.database import Database
from backend.app.verification.verifier import EvidenceVerifier


def create_app(settings: Settings | None = None) -> FastAPI:
    active_settings = settings or get_settings()
    database = Database(active_settings)
    processor = ContextProcessor(active_settings, database)
    reasoning = ReasoningClient(active_settings)

    processor_client = processor.client
    reasoning_client = reasoning.client

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        try:
            database.initialize()
            yield
        finally:
            try:
                await processor_client.close()
            finally:
                await reasoning_client.close()

    app = FastAPI(
        title="Contexta API",
        description="Persistent memory and evidence-backed AI context.",
        version="0.2.0",
        lifespan=lifespan,
    )

    app.state.settings = active_settings
    app.state.database = database
    app.state.processor = processor
    app.state.retriever = MemoryRetriever(database)
    app.state.reasoning = reasoning
    app.state.verifier = EvidenceVerifier(database)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=active_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
    )

    app.include_router(router)
    app.include_router(conversation_router)
    app.include_router(streaming_router)
    app.include_router(memory_bridge_router)

    return app


app = create_app()
