from backend.ingestion.loader import load_documents
from backend.ingestion.chunker import create_chunks
from backend.retrieval.bm25 import build_bm25_index, search_bm25


# Load documents and create chunks
documents = load_documents("data/docs.json")
chunks = create_chunks(documents, chunk_size=5, overlap=1)

# Build BM25 index
bm25_index = build_bm25_index(chunks)

# Test queries
queries = [
    "How do I reset my password?",
    "What is the maximum number of rows for CSV export?",
    "What are the API rate limits for Growth?"
]

for query in queries:

    print(f"\n{'=' * 60}")
    print(f"Query: {query}")

    results = search_bm25(
        query,
        bm25_index,
        chunks,
        top_k=3
    )

    for result in results:
        print(f"\nDocument: {result['doc_id']} - {result['title']}")
        print(f"Score: {result['bm25_score']:.4f}")
        print(f"Text: {result['text']}")