"""
FastAPI dependency injection providers.

Provides:
  - get_compiled_graph()  — cached compiled LangGraph graph (built once at startup)
  - get_settings()        — Settings singleton
  - get_db_path()         — resolved audit DB path

The compiled graph is expensive to build (imports all agents, validates structure).
It is compiled once via a module-level singleton and returned on every request.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

import structlog

logger = structlog.get_logger(__name__)

# Graph singleton             

_compiled_graph = None


def get_compiled_graph():
    """
    Return the cached compiled LangGraph graph.
    Thread-safe: FastAPI runs in a single-threaded async event loop.
    """
    global _compiled_graph
    if _compiled_graph is None:
        logger.info("graph_compiling")
        from core.graph import compile_graph
        _compiled_graph = compile_graph()
        logger.info("graph_compiled")
    return _compiled_graph


# Settings dependency            

def get_settings_dep():
    """Return the Settings singleton for use as a FastAPI dependency."""
    from config.settings import get_settings
    return get_settings()


# DB path dependency            

def get_db_path() -> str:
    """Return the resolved audit DB path from settings."""
    from config.settings import get_settings
    return get_settings().audit_db_path


# Type aliases for cleaner route signatures       

CompiledGraph = Annotated[object, Depends(get_compiled_graph)]
DBPath = Annotated[str, Depends(get_db_path)]