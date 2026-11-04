from backend.ingestion.loader import load_documents
from backend.ingestion.chunker import create_chunks

from backend.retrieval.bm25 import build_bm25_index
from backend.retrieval.vector_store import (
    load_embedding_model,
    build_vector_index
)
from backend.retrieval.hybrid import hybrid_search


# Load documents and create chunks
documents = load_documents("data/docs.json")
chunks = create_chunks(documents, chunk_size=5, overlap=1)

# Build BM25 index
bm25_index = build_bm25_index(chunks)

# Build FAISS index
model = load_embedding_model()
vector_index = build_vector_index(chunks, model)

# Test queries
queries = [
    "How do I reset my password?",
    "What is the maximum number of rows for CSV export?",
    "What are the API rate limits for Growth?",
    "Can I integrate Solstice with Salesforce?"
]

for query in queries:

    print(f"\n{'=' * 70}")
    print(f"Query: {query}")

    results = hybrid_search(
        query=query,
        bm25_index=bm25_index,
        vector_index=vector_index,
        model=model,
        chunks=chunks,
        top_k=3,
        candidate_k=10
    )

    for result in results:

        print(f"\nDocument: {result['doc_id']} - {result['title']}")
        print(f"BM25 Score: {result['bm25_score']:.4f}")
        print(f"Semantic Score: {result['semantic_score']:.4f}")
        print(
            f"Normalized Scores: "
            f"BM25={result['normalized_bm25_score']:.4f}, "
            f"Semantic={result['normalized_semantic_score']:.4f}"
        )
        print(f"Hybrid Score: {result['hybrid_score']:.4f}")
        print(f"Text: {result['text']}")