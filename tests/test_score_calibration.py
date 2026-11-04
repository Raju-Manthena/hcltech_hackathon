from backend.ingestion.loader import load_documents
from backend.ingestion.chunker import create_chunks

from backend.retrieval.bm25 import build_bm25_index, search_bm25
from backend.retrieval.vector_store import (
    load_embedding_model,
    build_vector_index,
    search_vector_store
)


# Load documents and create chunks
documents = load_documents("data/docs.json")
chunks = create_chunks(documents, chunk_size=5, overlap=1)

# Build retrieval indexes
bm25_index = build_bm25_index(chunks)

model = load_embedding_model()
vector_index = build_vector_index(chunks, model)


# Calibration queries and expected documents
test_cases = [
    ("How do I reset my password?", "D-01"),
    ("When does the password reset link expire?", "D-01"),
    ("What is the CSV export row limit?", "D-03"),
    ("How many API requests can Growth make per minute?", "D-06"),
    ("How do I invite a new team member?", "D-02"),
    ("Does Solstice support Salesforce integration?", None),
    ("Can I use OAuth for authentication?", None),
    ("What is the maximum storage capacity?", None)
]


for query, expected_doc in test_cases:

    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print(f"EXPECTED DOCUMENT: {expected_doc or 'NONE (ABSTAIN)'}")

    # BM25 retrieval
    bm25_results = search_bm25(
        query,
        bm25_index,
        chunks,
        top_k=10
    )

    print("\n--- BM25 RESULTS ---")

    for result in bm25_results:
        print(
            f"{result['doc_id']} | "
            f"Score: {result['bm25_score']:.4f} | "
            f"{result['chunk_id']}"
        )

    # Semantic retrieval
    semantic_results = search_vector_store(
        query,
        model,
        vector_index,
        chunks,
        top_k=10
    )

    print("\n--- FAISS RESULTS ---")

    for result in semantic_results:
        print(
            f"{result['doc_id']} | "
            f"Score: {result['semantic_score']:.4f} | "
            f"{result['chunk_id']}"
        )