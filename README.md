# Solstice Learning – Documentation Q&A Assistant

A Retrieval-Augmented Generation (RAG) application that allows corporate training administrators to ask questions in natural language and receive answers based on Solstice Learning's documentation. The system abstains when the available documentation does not contain sufficient information to answer a question.

## Features

- **Hybrid Retrieval:** Combines BM25 keyword retrieval and FAISS-based semantic search.
- **Weighted Ranking:** Uses an 80% keyword and 20% semantic score combination.
- **Context-Aware Answers:** Uses Groq's `openai/gpt-oss-120b` model to generate answers grounded in retrieved documentation.
- **Evidence Evaluation:** Checks whether the retrieved content sufficiently answers the question before generating a response.
- **Grounding Verification:** Verifies generated answers against retrieved evidence, with one retry for unsupported claims.
- **Abstention:** Avoids generating unsupported answers when sufficient evidence is unavailable.
- **Chat Interface:** Streamlit frontend with sample FAQs and documentation topics.

## Tech Stack

- **Frontend:** Streamlit
- **Backend:** FastAPI
- **LLM:** Groq – GPT-OSS-120B
- **Retrieval:** BM25, Sentence Transformers, FAISS
- **Embedding Model:** `all-MiniLM-L6-v2`
- **Language:** Python

## Architecture

1. Documentation is loaded from `data/docs.json`.
2. Documents are split into overlapping sentence-based chunks.
3. BM25 and FAISS indexes are built using the same chunks.
4. A user question is processed through both retrieval methods.
5. Retrieved results are combined using weighted score fusion.
6. The LLM evaluates whether the retrieved evidence is sufficient.
7. If sufficient, an answer is generated and checked for grounding.
8. If evidence is insufficient or the answer cannot be grounded, the system abstains.

## Project Structure

```text
├── app.py                    # Streamlit frontend
├── backend/
│   ├── main.py               # FastAPI application
│   ├── pipeline.py           # RAG pipeline orchestration
│   ├── ingestion/            # Document loading and chunking
│   ├── retrieval/            # BM25, FAISS and hybrid retrieval
│   └── generation/           # Answer generation and verification
├── data/
│   └── docs.json             # Source documentation
├── storage/                  # Persisted FAISS index and metadata
├── tests/                    # Component and pipeline tests
├── requirements.txt
├── .env.example
└── SOLUTION_DESIGN.md
```

## Setup and Execution

### 1. Clone the repository

```bash
git clone https://github.com/Raju-Manthena/hcltech_hackathon.git
cd hcltech_hackathon
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root using `.env.example`:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

### 5. Start the backend

```bash
uvicorn backend.main:app --reload
```

The FastAPI service runs at `http://127.0.0.1:8000`.

API documentation is available at `http://127.0.0.1:8000/docs`.

### 6. Start the frontend

Open another terminal, activate the virtual environment, and run:

```bash
streamlit run app.py
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Checks backend and pipeline status |
| POST | `/ask` | Accepts a question and returns an answer or abstention |

Example request:

```json
{
  "question": "What is the maximum number of rows supported in a CSV export?"
}
```

## Testing

The `tests/` directory contains tests covering:

- Document loading and chunking
- BM25 and semantic retrieval
- Hybrid score fusion
- FAISS index persistence
- Evidence sufficiency evaluation
- Answer grounding and retry behavior
- End-to-end question-answering pipeline

Run the tests using:

```bash
pytest tests/
```

## Deployment

The application uses a separate FastAPI backend and Streamlit frontend. Backend deployment was attempted on Render, and frontend deployment was attempted on Streamlit Community Cloud. The deployment was not fully completed within the available time.

## Limitations

- Answers are restricted to the documentation provided to the system.
- The current knowledge base uses the supplied sample documentation.
- LLM-based evidence evaluation and grounding verification depend on model responses.
- The application requires a valid Groq API key.
