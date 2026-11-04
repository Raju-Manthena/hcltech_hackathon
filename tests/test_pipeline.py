from backend.pipeline import (
    initialize_pipeline,
    answer_question
)


# Initialize the pipeline once
print("Initializing RAG pipeline...")

pipeline = initialize_pipeline()

print("\nPipeline initialized successfully.")


test_questions = [
    "How do I reset my password?",
    "What is the maximum number of rows for CSV export?",
    "What are the API rate limits for Growth?",
    "Can I integrate Solstice with Salesforce?"
]


for question in test_questions:

    print("\n" + "=" * 60)
    print("Question:", question)

    result = answer_question(
        question,
        pipeline
    )

    print("\nAssistant:")
    print(result["answer"])

    print("\nStatus:", result["status"])