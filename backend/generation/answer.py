from backend.generation.verifier import create_llm


def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
    llm=None,
    feedback: str = ""
) -> dict:
    """
    Generate an answer using only the retrieved documentation.

    Optional feedback is used to correct unsupported claims
    during the single regeneration attempt.
    """

    if not retrieved_chunks:
        raise ValueError("Cannot generate an answer without evidence.")

    if llm is None:
        llm = create_llm()

    context = "\n\n".join(
        f"Document ID: {chunk['doc_id']}\n"
        f"Title: {chunk['title']}\n"
        f"Content: {chunk['text']}"
        for chunk in retrieved_chunks
    )

    prompt = f"""
You are a documentation question-answering assistant
for Solstice Learning.

Answer the user's question using ONLY the provided documentation.

USER QUESTION:
{question}

RETRIEVED DOCUMENTATION:
{context}

INSTRUCTIONS:

1. Use only facts explicitly supported by the documentation.
2. Do not introduce external knowledge or assumptions.
3. If multiple passages contain relevant information, combine
   them into a clear and coherent answer.
4. Do not invent features, capabilities, limits or procedures.
5. Keep the answer concise and easy to understand.
6. Do not include document IDs, source references, or citations.
7. Do not mention information that is not supported by the
   retrieved documentation.
8. Return only the answer to the user's question.

REGENERATION FEEDBACK:
{feedback}

If regeneration feedback is provided, correct the identified
unsupported claims. Use only the retrieved documentation.
If the documentation cannot support the requested detail,
omit that detail rather than inventing an answer.
"""

    response = llm.invoke(prompt)

    answer = response.content

    if isinstance(answer, list):
        answer = "".join(
            item.get("text", "") if isinstance(item, dict)
            else str(item)
            for item in answer
        )

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("The LLM returned an empty answer.")

    return {
        "answer": answer.strip()
    }