import json
import re

from backend.generation.verifier import create_llm


def verify_grounding(
    question: str,
    answer: str,
    retrieved_chunks: list[dict],
    llm=None
) -> dict:
    """
    Check whether every factual claim in the generated answer
    is supported by the retrieved documentation.
    """

    if not retrieved_chunks:
        return {
            "grounded": False,
            "unsupported_claims": ["No supporting documentation available."],
            "reason": "No evidence was provided for verification."
        }

    if llm is None:
        llm = create_llm()

    context = "\n\n".join(
        f"Document: {chunk['title']} ({chunk['doc_id']})\n"
        f"Content: {chunk['text']}"
        for chunk in retrieved_chunks
    )

    prompt = f"""
You are a grounding verifier for a documentation
question-answering system.

Your task is to verify whether the generated answer is
fully supported by the provided documentation.

USER QUESTION:
{question}

RETRIEVED DOCUMENTATION:
{context}

GENERATED ANSWER:
{answer}

EVALUATION RULES:

1. Examine every factual claim in the generated answer.
2. Each factual claim must be directly supported by the
   retrieved documentation.
3. Do not use external knowledge to validate claims.
4. Different wording is acceptable if the meaning is
   clearly supported by the documentation.
5. Do not mark a claim unsupported merely because it is
   paraphrased.
6. Identify invented numbers, limits, features, procedures,
   conditions, or assumptions.
7. If even one material factual claim is unsupported,
   mark the answer as not grounded.
8. Ignore purely stylistic statements that make no factual claim.

Return ONLY a valid JSON object:

{{
    "grounded": true or false,
    "unsupported_claims": [
        "List each unsupported factual claim"
    ],
    "reason": "Brief explanation of the decision"
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

    match = re.search(r"\{.*\}", response_text, re.DOTALL)

    if not match:
        raise ValueError("Grounding verifier did not return valid JSON.")

    result = json.loads(match.group())

    if not isinstance(result.get("grounded"), bool):
        raise ValueError("Invalid 'grounded' value.")

    if not isinstance(result.get("unsupported_claims"), list):
        raise ValueError("Invalid 'unsupported_claims' value.")

    if not all(
        isinstance(claim, str)
        for claim in result["unsupported_claims"]
    ):
        raise ValueError("Unsupported claims must be strings.")

    if not isinstance(result.get("reason"), str):
        raise ValueError("Invalid 'reason' value.")

    return result



def generate_verified_answer(
    question: str,
    retrieved_chunks: list[dict],
    llm=None
) -> dict:
    """
    Generate an answer, verify its grounding, and retry once
    if unsupported claims are found.
    """

    from backend.generation.answer import generate_answer

    if llm is None:
        llm = create_llm()

    # First answer generation
    result = generate_answer(
        question,
        retrieved_chunks,
        llm
    )

    answer = result["answer"]

    # First grounding check
    verification = verify_grounding(
        question,
        answer,
        retrieved_chunks,
        llm
    )

    if verification["grounded"]:
        return {
            "answer": answer,
            "grounded": True,
            "retry_attempted": False,
            "abstained": False,
            "verification": verification
        }

    # One regeneration attempt using verifier feedback
    feedback = (
        "The previous answer contained unsupported claims:\n"
        + "\n".join(
            f"- {claim}"
            for claim in verification["unsupported_claims"]
        )
        + f"\nReason: {verification['reason']}"
    )

    revised_result = generate_answer(
        question,
        retrieved_chunks,
        llm,
        feedback=feedback
    )

    revised_answer = revised_result["answer"]

    # Verify the revised answer
    revised_verification = verify_grounding(
        question,
        revised_answer,
        retrieved_chunks,
        llm
    )

    if revised_verification["grounded"]:
        return {
            "answer": revised_answer,
            "grounded": True,
            "retry_attempted": True,
            "abstained": False,
            "verification": revised_verification
        }

    # Fail safely if the revised answer is still unsupported
    return {
        "answer": (
            "I couldn't generate a sufficiently supported answer "
            "from the available documentation."
        ),
        "grounded": False,
        "retry_attempted": True,
        "abstained": True,
        "verification": revised_verification
    }