from backend.ingestion.loader import load_documents
from backend.ingestion.chunker import create_chunks


documents = load_documents("data/docs.json")

chunks = create_chunks(
    documents,
    chunk_size=5,
    overlap=1
)

print("Documents loaded:", len(documents))
print("Chunks created:", len(chunks))

for chunk in chunks[:5]:
    print("\n", chunk)