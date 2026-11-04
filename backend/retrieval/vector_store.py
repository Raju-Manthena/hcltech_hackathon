import pickle
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = MODEL_NAME):
    """Load the Sentence Transformer embedding model."""

    return SentenceTransformer(model_name)


def build_vector_index(chunks: list[dict], model):
    """
    Generate embeddings and build a FAISS index.

    Embeddings are normalized so that inner product
    search is equivalent to cosine similarity.
    """

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def save_vector_store(
    index,
    chunks: list[dict],
    index_path: str = "storage/faiss.index",
    metadata_path: str = "storage/metadata.pkl"
):
    """Save the FAISS index and corresponding chunk metadata."""

    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, index_path)

    with open(metadata_path, "wb") as file:
        pickle.dump(chunks, file)


def load_vector_store(
    index_path: str = "storage/faiss.index",
    metadata_path: str = "storage/metadata.pkl"
):
    """Load the saved FAISS index and chunk metadata."""

    if not Path(index_path).exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    if not Path(metadata_path).exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    index = faiss.read_index(index_path)

    with open(metadata_path, "rb") as file:
        chunks = pickle.load(file)

    if index.ntotal != len(chunks):
        raise ValueError("FAISS index and chunk metadata are inconsistent.")

    return index, chunks


def search_vector_store(
    query: str,
    model,
    index,
    chunks: list[dict],
    top_k: int = 3
) -> list[dict]:
    """Retrieve semantically similar chunks using FAISS."""

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        results.append({
            **chunks[idx],
            "semantic_score": float(score)
        })

    return results