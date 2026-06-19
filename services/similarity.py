"""
Cosine similarity helpers and claim-pair contradiction detection utilities.

Uses scikit-learn for cosine similarity — no external vector DB required.
All heavy lifting (embedding) delegates to services/embedder.py.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any
import numpy as np

import structlog
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

logger = structlog.get_logger(__name__)


#  Core similarity functions 

def cosine_similarity_pair(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1D vectors.
    Returns a float in [-1, 1]; for normalized unit vectors this is [0, 1].
    """
    if vec_a.ndim == 1:
        vec_a = vec_a.reshape(1, -1)
    if vec_b.ndim == 1:
        vec_b = vec_b.reshape(1, -1)
    score = sk_cosine_similarity(vec_a, vec_b)[0][0]
    return float(np.clip(score, -1.0, 1.0))


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute an N×N cosine similarity matrix from an (N, dim) embedding array.
    Returns a 2D numpy array of shape (N, N).
    """
    if embeddings.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return sk_cosine_similarity(embeddings).astype(np.float32)


def find_similar_pairs(
    embeddings: np.ndarray,
    ids: List[str],
    threshold: float,
    upper_triangle_only: bool = True,
) -> List[Tuple[str, str, float]]:
    """
    Find all pairs of items whose cosine similarity exceeds `threshold`.

    Args:
        embeddings:          (N, dim) embedding matrix.
        ids:                 List of N string identifiers (e.g. claim_ids or source_ids).
        threshold:           Similarity threshold (e.g. 0.70 for contradictions, 0.95 for dedup).
        upper_triangle_only: If True, only return (i, j) pairs where i < j (no duplicates).

    Returns:
        List of (id_a, id_b, similarity_score) tuples sorted by score descending.
    """
    if len(embeddings) < 2:
        return []

    sim_matrix = cosine_similarity_matrix(embeddings)
    n = len(ids)
    pairs: List[Tuple[str, str, float]] = []

    for i in range(n):
        start_j = i + 1 if upper_triangle_only else 0
        for j in range(start_j, n):
            if i == j:
                continue
            score = float(sim_matrix[i][j])
            if score >= threshold:
                pairs.append((ids[i], ids[j], score))

    # Sort highest similarity first
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


#  Source deduplication helpers 

def find_duplicate_sources(
    source_texts: List[str],
    source_ids: List[str],
    similarity_threshold: float = 0.95,
) -> List[Tuple[str, str, float]]:
    """
    Find source pairs that are near-duplicates (similarity >= threshold).
    Used by the Ingress agent to identify content duplicates.

    Returns:
        List of (source_id_a, source_id_b, score) for pairs above threshold.
    """
    if len(source_texts) < 2:
        return []

    from services.embedder import embed_texts
    embeddings = embed_texts(source_texts, normalize=True)
    return find_similar_pairs(embeddings, source_ids, threshold=similarity_threshold)


#  Claim pair analysis 

def find_potentially_contradicting_claim_pairs(
    claim_texts: List[str],
    claim_ids: List[str],
    similarity_threshold: float = 0.70,
) -> List[Tuple[str, str, float]]:
    """
    Find claim pairs that are topically similar enough to potentially contradict.
    These pairs are then passed to the LLM contradiction judge.

    Returns:
        List of (claim_id_a, claim_id_b, similarity_score) tuples.
    """
    if len(claim_texts) < 2:
        return []

    from services.embedder import embed_texts
    embeddings = embed_texts(claim_texts, normalize=True)
    return find_similar_pairs(embeddings, claim_ids, threshold=similarity_threshold)


#  Relevance helpers 

def compute_question_source_similarities(
    question_text: str,
    source_texts: List[str],
) -> List[float]:
    """
    Compute cosine similarity between the question and each source text.
    Used as a fast pre-filter before the LLM relevance scoring call.

    Returns:
        List of float similarity scores, one per source.
    """
    if not source_texts:
        return []

    from services.embedder import embed_texts, embed_single
    q_emb = embed_single(question_text, normalize=True).reshape(1, -1)
    s_embs = embed_texts(source_texts, normalize=True)

    if s_embs.shape[0] == 0:
        return []

    scores = sk_cosine_similarity(q_emb, s_embs)[0]
    return [float(np.clip(s, 0.0, 1.0)) for s in scores]


#  Named entity for structured output 

class SimilarityPair:
    """Lightweight data class for a scored similarity pair."""

    def __init__(self, id_a: str, id_b: str, score: float):
        self.id_a = id_a
        self.id_b = id_b
        self.score = score

    def to_dict(self) -> Dict[str, Any]:
        return {"id_a": self.id_a, "id_b": self.id_b, "similarity_score": self.score}

    def __repr__(self) -> str:
        return f"SimilarityPair({self.id_a}, {self.id_b}, score={self.score:.3f})"