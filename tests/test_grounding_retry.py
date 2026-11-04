from types import SimpleNamespace

from backend.generation.grounding import generate_verified_answer


class MockLLM:
    """Mock LLM that returns predefined responses in sequence."""

    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0

    def invoke(self, prompt):
        if self.call_count >= len(self.responses):
            raise RuntimeError("No more mock responses available.")

        response = self.responses[self.call_count]
        self.call_count += 1

        return SimpleNamespace(content=response)


# Minimal retrieved context
retrieved_chunks = [
    {
        "chunk_id": "chunk_8",
        "doc_id": "D-06",
        "title": "API Rate Limits",
        "text": (
            "Growth plans are limited to 60 requests per minute, "
            "and Enterprise plans are limited to 300 requests per minute, "
            "per API key."
        )
    }
]

question = "What are the API rate limits for Growth?"


# --------------------------------------------------
# TEST 4: Initial answer fails, retry succeeds
# --------------------------------------------------

print("\n" + "=" * 60)
print("TEST 4: Successful retry")

mock_llm = MockLLM([
    # 1. Initial generated answer (unsupported)
    "Growth plans allow 60 requests per minute and "
    "50,000 requests per day.",

    # 2. First grounding verification (fails)
    """
    {
        "grounded": false,
        "unsupported_claims": [
            "50,000 requests per day"
        ],
        "reason": "The documentation does not specify a daily limit."
    }
    """,

    # 3. Revised answer (corrected)
    "Growth plans are limited to 60 requests per minute per API key.",

    # 4. Second grounding verification (passes)
    """
    {
        "grounded": true,
        "unsupported_claims": [],
        "reason": "The answer is supported by the documentation."
    }
    """
])

result = generate_verified_answer(
    question,
    retrieved_chunks,
    mock_llm
)

print("Answer:", result["answer"])
print("Grounded:", result["grounded"])
print("Retry attempted:", result["retry_attempted"])
print("Abstained:", result["abstained"])
print("LLM calls:", mock_llm.call_count)


# --------------------------------------------------
# TEST 5: Initial answer fails, retry also fails
# --------------------------------------------------

print("\n" + "=" * 60)
print("TEST 5: Failed retry - abstention")

mock_llm = MockLLM([
    # 1. Initial generated answer (unsupported)
    "Growth plans allow 60 requests per minute and "
    "50,000 requests per day.",

    # 2. First grounding verification (fails)
    """
    {
        "grounded": false,
        "unsupported_claims": [
            "50,000 requests per day"
        ],
        "reason": "The documentation does not specify a daily limit."
    }
    """,

    # 3. Revised answer (still unsupported)
    "Growth plans allow 60 requests per minute and "
    "100,000 requests per day.",

    # 4. Second grounding verification (fails)
    """
    {
        "grounded": false,
        "unsupported_claims": [
            "100,000 requests per day"
        ],
        "reason": "The documentation does not specify a daily limit."
    }
    """
])

result = generate_verified_answer(
    question,
    retrieved_chunks,
    mock_llm
)

print("Answer:", result["answer"])
print("Grounded:", result["grounded"])
print("Retry attempted:", result["retry_attempted"])
print("Abstained:", result["abstained"])
print("LLM calls:", mock_llm.call_count)