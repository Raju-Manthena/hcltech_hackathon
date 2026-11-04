from backend.ingestion.loader import load_documents
from backend.ingestion.chunker import create_chunks

from backend.retrieval.vector_store import (
    load_embedding_model,
    build_vector_index,
    save_vector_store,
    load_vector_store,
    search_vector_store
)


# Load documents and create chunks
documents = load_documents("data/docs.json")
chunks = create_chunks(documents, chunk_size=5, overlap=1)

# Load embedding model
model = load_embedding_model()

# Build FAISS index
index = build_vector_index(chunks, model)

print("Total vectors indexed:", index.ntotal)
print("Embedding dimension:", index.d)

# Save index and metadata
save_vector_store(index, chunks)

# Load saved index and metadata
loaded_index, loaded_chunks = load_vector_store()

print("\nFAISS index loaded successfully.")
print("Loaded vectors:", loaded_index.ntotal)

# Test queries
queries = [
    "How can I change my password?",
    "What is the maximum CSV export limit?",
    "How many API requests are allowed on the Growth plan?",
    "Can I integrate Salesforce with Solstice?"
]

for query in queries:

    print(f"\n{'=' * 60}")
    print(f"Query: {query}")

    results = search_vector_store(
        query,
        model,
        loaded_index,
        loaded_chunks,
        top_k=3
    )

    for result in results:
        print(f"\nDocument: {result['doc_id']} - {result['title']}")
        print(f"Similarity: {result['semantic_score']:.4f}")
        print(f"Text: {result['text']}")