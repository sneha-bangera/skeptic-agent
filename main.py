"""
FastAPI application entrypoint for The Skeptic's Brief.

Startup sequence:
  1. Configure structlog JSON logging
  2. Initialise the SQLite audit database (create tables if not exist)
  3. Load persisted credibility adjustments from DB into memory
  4. Compile the LangGraph graph (imports all agents, validates structure)
  5. Start accepting requests

Run locally:
  python main.py
  # or
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router
from api.dependencies import get_compiled_graph


# Logging configuration 

def _configure_logging(log_level: str = "INFO") -> None:
    """Configure structlog with JSON output for production, pretty for dev."""
    is_dev = os.environ.get("ENV", "development").lower() == "development"

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if is_dev:
        processors = shared_processors + [structlog.dev.ConsoleRenderer()]
    else:
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Quieten noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


# Lifespan (startup / shutdown)  

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler — runs startup tasks before first request."""
    from config.settings import get_settings
    from storage.audit_db import init_db, load_adjustments_from_db

    settings = get_settings()
    _configure_logging(settings.log_level)

    log = structlog.get_logger("startup")
    log.info("skeptics_brief_starting", model=settings.model_name, dry_run=settings.dry_run)

    # 1. Init audit DB   
    try:
        await init_db(settings.audit_db_path)
        log.info("audit_db_ready", path=settings.audit_db_path)
    except Exception as exc:
        log.error("audit_db_init_failed", error=str(exc))
        # Non-fatal: the API still works without the DB (audit just fails silently)

    # 2. Load persisted credibility adjustments      
    try:
        loaded = await load_adjustments_from_db(settings.audit_db_path)
        log.info("credibility_adjustments_loaded", count=loaded)
    except Exception as exc:
        log.error("credibility_adjustments_load_failed", error=str(exc))

    # 3. Compile the graph  
    try:
        get_compiled_graph()
        log.info("graph_ready")
    except Exception as exc:
        log.error("graph_compilation_failed", error=str(exc))
        raise  # Fatal: can't serve requests without the graph

    log.info("skeptics_brief_ready", docs="http://localhost:8000/docs")

    yield  # ← application runs here

    log.info("skeptics_brief_shutting_down")


# FastAPI app     

app = FastAPI(
    title="The Skeptic's Brief",
    description=(
        "A production-grade multi-agent AI system that answers research questions "
        "from a corpus of mixed-quality sources by actively challenging its own "
        "conclusions through an adversarial Skeptic agent before finalising them."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS       

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler   

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log = structlog.get_logger("exception_handler")
    log.error(
        "unhandled_exception",
        path=str(request.url),
        method=request.method,
        error=str(exc),
        exc_type=type(exc).__name__,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred. Check logs for details.",
            "error_type": type(exc).__name__,
        },
    )


# Routes      

app.include_router(router, prefix="")


# Dev server entry point   

if __name__ == "__main__":
    from config.settings import get_settings
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower(),
    )