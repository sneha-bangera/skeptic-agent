"""
Sentence-transformer embedding service.

Wraps sentence-transformers to produce dense vector embeddings for:
- Source content deduplication (Ingress agent)
- Claim-pair similarity for contradiction detection (Contradiction Detector agent)

Design decisions:
- Singleton pattern: model is loaded once and reused across all requests.
- All embedding calls go through this service — agents never import sentence_transformers directly.
- encode() always returns numpy arrays normalized to unit length (for cosine similarity via dot product).
"""

from __future__ import annotations

import numpy as np
from typing import List, Union

import structlog
from core.exceptions import EmbeddingError

logger = structlog.get_logger(__name__)

# Singleton model holder 
_model = None
_model_name: str | None = None


def _get_model(model_name: str):
    """Load the sentence-transformer model (once per process)."""
    global _model, _model_name
    if _model is None or _model_name != model_name:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("embedder_loading_model", model=model_name)
            _model = SentenceTransformer(model_name)
            _model_name = model_name
            logger.info("embedder_model_loaded", model=model_name)
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load embedding model '{model_name}': {exc}"
            ) from exc
    return _model


# Public API 

def embed_texts(
    texts: List[str],
    model_name: str | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Embed a list of texts and return a 2D numpy array of shape (n, embedding_dim).

    Args:
        texts:       List of strings to embed. Empty strings are handled gracefully.
        model_name:  Model name override (defaults to settings.embedding_model).
        normalize:   If True, normalize embeddings to unit length (enables cosine sim via dot product).

    Returns:
        numpy array of shape (len(texts), embedding_dim).

    Raises:
        EmbeddingError: If the model fails to load or encode.
    """
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)  # 384 default dim

    # Resolve model name
    if model_name is None:
        try:
            from config.settings import get_settings
            model_name = get_settings().embedding_model
        except Exception:
            model_name = "all-MiniLM-L6-v2"

    # Replace empty strings to avoid tokenizer issues
    safe_texts = [t if t.strip() else "[empty]" for t in texts]

    try:
        model = _get_model(model_name)
        embeddings = model.encode(
            safe_texts,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            batch_size=32,
        )
        return np.array(embeddings, dtype=np.float32)
    except EmbeddingError:
        raise
    except Exception as exc:
        raise EmbeddingError(f"Embedding failed: {exc}") from exc


def embed_single(
    text: str,
    model_name: str | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Embed a single text and return a 1D numpy array.
    Convenience wrapper around embed_texts.
    """
    result = embed_texts([text], model_name=model_name, normalize=normalize)
    return result[0]


def get_embedding_dim(model_name: str | None = None) -> int:
    """Return the embedding dimension for the configured model."""
    if model_name is None:
        try:
            from config.settings import get_settings
            model_name = get_settings().embedding_model
        except Exception:
            model_name = "all-MiniLM-L6-v2"
    model = _get_model(model_name)
    return model.get_embedding_dimension()