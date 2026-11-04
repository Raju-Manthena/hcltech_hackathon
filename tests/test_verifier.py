from backend.ingestion.loader import load_documents
from backend.ingestion.chunker import create_chunks
from backend.retrieval.bm25 import build_bm25_index
from backend.retrieval.vector_store import (
    load_embedding_model,
    build_vector_index
)
from backend.retrieval.hybrid import hybrid_search
from backend.generation.verifier import evaluate_evidence, create_llm
from backend.generation.answer import generate_answer

# Load documents and create chunks
documents = load_documents("data/docs.json")
chunks = create_chunks(documents, chunk_size=5, overlap=1)

# Build retrieval indexes
bm25_index = build_bm25_index(chunks)

model = load_embedding_model()
vector_index = build_vector_index(chunks, model)

# Initialize Groq
llm = create_llm()

test_questions = [
    "How do I reset my password?",
    "What is the maximum number of rows for CSV export?",
    "What are the API rate limits for Growth?",
    "Can I integrate Solstice with Salesforce?"
]

for question in test_questions:

    retrieved_chunks = hybrid_search(
        question,
        bm25_index,
        vector_index,
        model,
        chunks,
        top_k=3
    )

    evaluation = evaluate_evidence(
        question,
        retrieved_chunks,
        llm
    )

    print("\n" + "=" * 60)
    print("Question:", question)
    print("Evidence sufficient:", evaluation["sufficient"])
    print("Reason:", evaluation["reason"])

    if evaluation["sufficient"]:

        result = generate_answer(
            question,
            retrieved_chunks,
            llm
        )

        print("\nGenerated Answer:")
        print(result["answer"])

    else:
        print("\nAssistant:")
        print(
            "I don't have enough information in the available "
            "documentation to answer this question."
        )