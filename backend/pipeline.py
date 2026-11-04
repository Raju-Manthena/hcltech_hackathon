from backend.ingestion.loader import load_documents
from backend.ingestion.chunker import create_chunks

from backend.retrieval.bm25 import build_bm25_index
from backend.retrieval.vector_store import (
    load_embedding_model,
    build_vector_index,
    save_vector_store,
    load_vector_store
)
from backend.retrieval.hybrid import hybrid_search

from backend.generation.verifier import (
    create_llm,
    evaluate_evidence
)
from backend.generation.grounding import generate_verified_answer


ABSTENTION_MESSAGE = (
    "I don't have enough information in the available "
    "documentation to answer this question."
)


def initialize_pipeline(
    data_path: str = "data/docs.json",
    chunk_size: int = 5,
    overlap: int = 1,
    top_k: int = 3,
    candidate_k: int = 10
) -> dict:
    """
    Load documents, initialize retrieval indexes and Groq LLM.

    Reuse the saved FAISS index when its chunk metadata matches
    the current documentation. Otherwise, rebuild and save it.
    """

    # 1. Load and chunk documentation
    documents = load_documents(data_path)

    chunks = create_chunks(
        documents,
        chunk_size=chunk_size,
        overlap=overlap
    )

    if not chunks:
        raise ValueError("No document chunks were created.")

    # 2. Build BM25 index
    bm25_index = build_bm25_index(chunks)

    # 3. Load embedding model
    embedding_model = load_embedding_model()

    # 4. Load existing FAISS index or build a new one
    try:
        vector_index, saved_chunks = load_vector_store()

        if saved_chunks == chunks:
            print("Loaded existing FAISS index.")
        else:
            raise ValueError(
                "Saved index metadata does not match current documents."
            )

    except (FileNotFoundError, ValueError):
        print("Building a new FAISS index...")

        vector_index = build_vector_index(
            chunks,
            embedding_model
        )

        save_vector_store(
            vector_index,
            chunks
        )

        print("FAISS index saved successfully.")

    # 5. Initialize Groq LLM
    llm = create_llm()

    return {
        "chunks": chunks,
        "bm25_index": bm25_index,
        "vector_index": vector_index,
        "embedding_model": embedding_model,
        "llm": llm,
        "top_k": top_k,
        "candidate_k": candidate_k
    }


def answer_question(
    question: str,
    pipeline: dict
) -> dict:
    """
    Execute the complete RAG pipeline for a user question.

    Returns the final answer and its processing status.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    # 1. Hybrid retrieval
    retrieved_chunks = hybrid_search(
        query=question,
        bm25_index=pipeline["bm25_index"],
        vector_index=pipeline["vector_index"],
        model=pipeline["embedding_model"],
        chunks=pipeline["chunks"],
        top_k=pipeline["top_k"],
        candidate_k=pipeline["candidate_k"]
    )

    if not retrieved_chunks:
        return {
            "answer": ABSTENTION_MESSAGE,
            "status": "abstained"
        }

    # 2. Evidence sufficiency evaluation
    evidence_result = evaluate_evidence(
        question=question,
        retrieved_chunks=retrieved_chunks,
        llm=pipeline["llm"]
    )

    if not evidence_result["sufficient"]:
        return {
            "answer": ABSTENTION_MESSAGE,
            "status": "abstained"
        }

    # 3. Answer generation + grounding verification + retry
    result = generate_verified_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
        llm=pipeline["llm"]
    )

    if result["abstained"]:
        return {
            "answer": result["answer"],
            "status": "abstained"
        }

    return {
        "answer": result["answer"],
        "status": "answered"
    }