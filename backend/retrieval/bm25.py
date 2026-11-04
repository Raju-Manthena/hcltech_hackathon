import re
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    """Build a BM25 index from document chunks."""

    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_chunks)


def search_bm25(
    query: str,
    index: BM25Okapi,
    chunks: list[dict],
    top_k: int = 3
) -> list[dict]:
    """Retrieve the top-k chunks using BM25 scores."""

    tokenized_query = tokenize(query)

    scores = index.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for idx in ranked_indices[:top_k]:

        if scores[idx] <= 0:
            continue

        results.append({
            **chunks[idx],
            "bm25_score": float(scores[idx])
        })

    return results