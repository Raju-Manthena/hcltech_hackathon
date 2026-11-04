from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.pipeline import initialize_pipeline, answer_question


# Shared pipeline instance
pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the RAG pipeline when the API starts."""

    global pipeline

    print("Initializing RAG pipeline...")
    pipeline = initialize_pipeline()
    print("RAG pipeline initialized.")

    yield

    print("Shutting down API...")


app = FastAPI(
    title="Solstice Docs Q&A Assistant",
    description="Documentation-based question answering using RAG.",
    version="1.0.0",
    lifespan=lifespan
)


class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=1000,
        description="User's documentation-related question"
    )


class AnswerResponse(BaseModel):
    answer: str
    status: str


@app.get("/health")
def health_check():
    """Check whether the API and pipeline are ready."""

    return {
        "status": "healthy",
        "pipeline_ready": pipeline is not None
    }


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """Answer a user question using the RAG pipeline."""

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not initialized."
        )

    try:
        result = answer_question(
            request.question,
            pipeline
        )

        return result

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Unable to process the question. Please try again."
        )