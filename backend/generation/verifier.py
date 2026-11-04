import json
import os
import re

from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()


def create_llm():
    """Initialize the Groq chat model."""

    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing from .env")

    if not model_name:
        raise ValueError("GROQ_MODEL is missing from .env")

    return ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=0
    )


def evaluate_evidence(
    question: str,
    retrieved_chunks: list[dict],
    llm=None
) -> dict:
    """
    Determine whether retrieved documentation contains
    sufficient evidence to answer the user's question.
    """

    if not retrieved_chunks:
        return {
            "sufficient": False,
            "reason": "No relevant documentation was retrieved.",
            "missing_information": "Supporting documentation."
        }

    if llm is None:
        llm = create_llm()

    context = "\n\n".join(
        f"Document: {chunk['title']} ({chunk['doc_id']})\n"
        f"Content: {chunk['text']}"
        for chunk in retrieved_chunks
    )

    prompt = f"""
You are an evidence sufficiency evaluator for a documentation
question-answering system.

Your task is to determine whether the provided documentation
contains sufficient information to answer the user's question.

USER QUESTION:
{question}

RETRIEVED DOCUMENTATION:
{context}

EVALUATION RULES:

1. Evaluate the combined retrieved documentation.
2. The documentation must directly support the answer.
3. Related topics, keyword overlap, or general conceptual
   similarity do not count as sufficient evidence.
4. Do not assume that a feature exists just because related
   functionality is documented.
5. Different wording is acceptable if the documentation
   clearly provides the requested information.
6. If the question has multiple parts, all parts must be
   answerable from the retrieved documentation.
7. If the evidence is incomplete or ambiguous, mark it
   insufficient.

Return ONLY a valid JSON object with these fields:

{{
    "sufficient": true or false,
    "reason": "Brief explanation",
    "missing_information": "Missing information, or empty string if sufficient"
}}
"""

    response = llm.invoke(prompt)

    response_text = response.content

    if isinstance(response_text, list):
        response_text = "".join(
            item.get("text", "") if isinstance(item, dict)
            else str(item)
            for item in response_text
        )

    # Extract JSON if the model adds surrounding text or code fences.
    match = re.search(r"\{.*\}", response_text, re.DOTALL)

    if not match:
        raise ValueError("Groq did not return valid JSON.")

    result = json.loads(match.group())

    if not isinstance(result.get("sufficient"), bool):
        raise ValueError("Invalid 'sufficient' value in evaluator response.")

    if not isinstance(result.get("reason"), str):
        raise ValueError("Invalid 'reason' value in evaluator response.")

    if not isinstance(result.get("missing_information"), str):
        raise ValueError("Invalid 'missing_information' value.")

    return result