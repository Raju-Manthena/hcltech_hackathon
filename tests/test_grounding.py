from backend.ingestion.loader import load_documents
from backend.ingestion.chunker import create_chunks
from backend.retrieval.bm25 import build_bm25_index
from backend.retrieval.vector_store import (
    load_embedding_model,
    build_vector_index
)
from backend.retrieval.hybrid import hybrid_search
from backend.generation.grounding import (
    verify_grounding,
    generate_verified_answer
)
from backend.generation.verifier import create_llm


# Load documents
documents = load_documents("data/docs.json")
chunks = create_chunks(documents, chunk_size=5, overlap=1)

# Build retrieval indexes
bm25_index = build_bm25_index(chunks)

model = load_embedding_model()
vector_index = build_vector_index(chunks, model)

# Initialize Groq
llm = create_llm()

# Retrieve context for API rate limit question
question = "What are the API rate limits for Growth?"

retrieved_chunks = hybrid_search(
    question,
    bm25_index,
    vector_index,
    model,
    chunks,
    top_k=3
)

# Test 1: Correct answer
correct_answer = (
    "Growth plans are limited to 60 requests per minute "
    "per API key."
)

# Test 2: Answer with an invented claim
incorrect_answer = (
    "Growth plans are limited to 60 requests per minute "
    "per API key, with a daily limit of 50,000 requests."
)

print("\n" + "=" * 60)
print("TEST 1: Supported answer")

result = verify_grounding(
    question,
    correct_answer,
    retrieved_chunks,
    llm
)

print(result)


print("\n" + "=" * 60)
print("TEST 2: Answer containing unsupported information")

result = verify_grounding(
    question,
    incorrect_answer,
    retrieved_chunks,
    llm
)

print(result)


print("\n" + "=" * 60)
print("TEST 3: Complete generation and verification workflow")

result = generate_verified_answer(
    question,
    retrieved_chunks,
    llm
)

print("Answer:", result["answer"])
print("Grounded:", result["grounded"])
print("Retry attempted:", result["retry_attempted"])
print("Abstained:", result["abstained"])