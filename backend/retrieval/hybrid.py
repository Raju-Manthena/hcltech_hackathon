from backend.retrieval.bm25 import search_bm25
from backend.retrieval.vector_store import search_vector_store


def transform_bm25_score(score: float, k: float = 2.0) -> float:
    """Transform BM25 scores into a bounded [0, 1] range."""

    if score < 0:
        raise ValueError("BM25 scores cannot be negative.")

    return score / (score + k)


def transform_semantic_score(score: float) -> float:
    """Keep cosine similarity within the [0, 1] range."""

    return max(0.0, min(1.0, score))


def hybrid_search(
    query: str,
    bm25_index,
    vector_index,
    model,
    chunks: list[dict],
    top_k: int = 3,
    candidate_k: int = 10,
    bm25_weight: float = 0.8,
    semantic_weight: float = 0.2
) -> list[dict]:
    """
    Perform hybrid retrieval using BM25 and FAISS.

    Scores are transformed independently before weighted fusion.
    """

    if top_k <= 0 or candidate_k <= 0:
        raise ValueError("top_k and candidate_k must be positive.")

    if bm25_weight < 0 or semantic_weight < 0:
        raise ValueError("Retrieval weights cannot be negative.")

    if bm25_weight + semantic_weight == 0:
        raise ValueError("At least one retrieval weight must be positive.")

    total_weight = bm25_weight + semantic_weight
    bm25_weight /= total_weight
    semantic_weight /= total_weight

    bm25_results = search_bm25(
        query, bm25_index, chunks, top_k=candidate_k
    )

    semantic_results = search_vector_store(
        query, model, vector_index, chunks, top_k=candidate_k
    )

    candidates = {}

    # Add BM25 results
    for result in bm25_results:
        chunk_id = result["chunk_id"]

        candidates[chunk_id] = {
            **result,
            "bm25_score": result["bm25_score"],
            "semantic_score": 0.0
        }

    # Add semantic results
    for result in semantic_results:
        chunk_id = result["chunk_id"]

        if chunk_id not in candidates:
            candidates[chunk_id] = {
                **result,
                "bm25_score": 0.0,
                "semantic_score": result["semantic_score"]
            }
        else:
            candidates[chunk_id]["semantic_score"] = (
                result["semantic_score"]
            )

    if not candidates:
        return []

    candidate_list = list(candidates.values())

    # Transform scores independently
    for candidate in candidate_list:

        candidate["normalized_bm25_score"] = transform_bm25_score(
            candidate["bm25_score"]
        )

        candidate["normalized_semantic_score"] = transform_semantic_score(
            candidate["semantic_score"]
        )

        # Weighted score fusion
        candidate["hybrid_score"] = (
            bm25_weight * candidate["normalized_bm25_score"]
            + semantic_weight * candidate["normalized_semantic_score"]
        )

    ranked_results = sorted(
        candidate_list,
        key=lambda item: item["hybrid_score"],
        reverse=True
    )

    return ranked_results[:top_k]