# Solution Design — Solstice Docs Q&A Assistant

## 1. Problem Understanding

Solstice Learning's administrators frequently ask questions about product features, account management, billing and other operational topics. Although the answers already exist in the company's documentation, finding relevant information requires manually searching through multiple articles.

The proposed solution is a **Documentation Q&A Assistant** that allows administrators to ask questions in natural language and receive concise, accurate answers grounded in the available documentation.

A key requirement is preventing hallucinations. When the documentation does not contain sufficient information to answer a question, the system must explicitly acknowledge this rather than generate an unsupported response.

## 2. Assumptions

1. The primary users are Solstice Learning administrators who need quick access to product documentation.
2. The supplied six documentation articles constitute the initial knowledge base and evaluation dataset, rather than the maximum size of the intended system.
3. Users can submit questions in English using natural language.
4. Answers must be grounded exclusively in retrieved documentation, without relying on the LLM's general knowledge.
5. The system should abstain when the retrieved evidence is insufficient to answer a question reliably.
6. Answers should include source references to allow users to verify the information.
7. The initial implementation uses the supplied JSON documents. The ingestion architecture should allow additional document formats and larger collections to be supported in the future.
8. The initial prototype does not include user authentication or persistent conversation history.

## 3. Proposed Solution and Architecture

The solution will use a Retrieval-Augmented Generation (RAG) architecture with weighted hybrid retrieval, combining keyword-based and semantic search.

**Technology Stack:**
- Frontend: Streamlit
- Backend: FastAPI
- LLM: Groq
- Embedding Model: Sentence Transformers (`all-MiniLM-L6-v2`)
- Keyword Retrieval: BM25
- Vector Search: FAISS
- Programming Language: Python

### Document Processing

The supplied documentation will undergo preprocessing and sentence-based chunking. Each chunk will retain metadata such as document ID, title and chunk ID to support source attribution.

The same collection of chunks will be indexed by both retrieval mechanisms:

- **Keyword Retrieval (BM25):** Identifies relevant passages through lexical matching.
- **Semantic Retrieval (FAISS):** Generates embeddings and retrieves semantically similar passages using cosine similarity.

The indexing process will be designed to support re-indexing when the underlying documentation changes.

### Hybrid Retrieval

Both retrieval mechanisms will contribute to the final relevance score. Since exact keyword matching is particularly useful for product-specific terminology, keyword retrieval will receive greater importance.

The initial weighting will be:

`Final Score = 0.7 × BM25 Score + 0.3 × Semantic Similarity Score`

Both scores will be normalized before combination.

The system will retrieve the top three relevant chunks and pass them to the answer-generation component, preserving their document metadata.

### Answer Generation and Abstention

The retrieved passages and user question will be provided to the Groq LLM through a structured prompt.

The prompt will define the model's role, objective, available context, output expectations, rules and constraints. The LLM must:

- Answer using only the supplied documentation context.
- Avoid introducing unsupported information.
- Provide concise and relevant responses.
- Acknowledge missing information instead of guessing.

An evidence-relevance check will be applied before answer generation. Insufficient evidence will trigger an abstention response. Generated answers will also be checked for grounding against the retrieved context.

Answers will include supporting document titles and relevant passages.

## 4. Application Features

The Streamlit interface will provide:

1. **Documentation Q&A:** A natural-language question interface for administrators.
2. **Source References:** Supporting document titles and passages displayed alongside answers.
3. **Abstention Handling:** Explicit responses when the documentation does not provide sufficient evidence.
4. **Frequently Asked Questions:** Predefined common questions that users can select directly.
5. **Related Questions:** Contextual question suggestions following an answer, encouraging users to explore related documentation.
6. **Document Overview:** A simple view of the available documentation articles.

FAQ selections and related questions will use the same RAG pipeline as manually submitted questions.

## 5. Trade-offs and Design Considerations

- **Hybrid retrieval over individual retrieval methods:** BM25 provides stronger lexical matching, while semantic retrieval handles variations in wording. Combining both improves retrieval flexibility, at the cost of additional indexing and score-normalization logic.
- **FAISS over a managed vector database:** Provides vector indexing and similarity search without introducing an additional database service. Persistent index storage and metadata management will be handled by the application.
- **Sentence-based chunking:** Preserves meaningful text boundaries and allows focused retrieval, although chunk size and overlap may require tuning for larger documents.
- **Groq for answer generation:** Provides LLM-powered natural-language responses while keeping retrieval and answer generation as separate components.
- **FastAPI and Streamlit separation:** Provides a clear separation between the user interface and backend processing, although it introduces an additional API integration layer.

## 6. Testing and Verification

The implementation will be evaluated using the five supplied example questions, including the deliberately unanswerable Salesforce integration question.

Additional tests will cover:

- Different formulations of the same question.
- Questions involving information from multiple documents.
- Unanswerable and partially answerable questions.
- Questions containing misleading keywords.
- Retrieval accuracy and source attribution.
- API failures and invalid inputs.

The primary success criteria are accurate retrieval, grounded answers, correct source references and reliable abstention.

## 7. Limitations and Future Improvements

The initial version will focus on the supplied JSON documentation and a working end-to-end Q&A workflow.

Potential future improvements include:

- Support for PDF, DOCX and other document formats.
- Document upload, deletion and automatic re-indexing.
- Improved retrieval through metadata filtering and query expansion.
- Retrieval and abstention threshold optimization using a larger evaluation dataset.
- User feedback collection and retrieval-quality analytics.
- Conversation history and contextual follow-up questions.
- Authentication and role-based access control for enterprise usage.
- Scaling document storage and indexing for larger knowledge bases.

The initial implementation prioritizes a reliable, explainable and extensible documentation Q&A workflow while keeping the development scope appropriate to the available time and resources.