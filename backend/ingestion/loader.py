import json
from pathlib import Path


def load_documents(file_path: str = "data/docs.json") -> list[dict]:
    """
    Load documentation articles from a JSON file.

    Returns:
        List of dictionaries containing doc_id, title and content.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict) or "docs" not in data:
        raise ValueError("Expected a JSON object containing a 'docs' key.")

    documents = data["docs"]

    if not isinstance(documents, list):
        raise ValueError("'docs' must contain a list of documents.")

    required_fields = {"doc_id", "title", "content"}

    for index, document in enumerate(documents):

        if not isinstance(document, dict):
            raise ValueError(f"Document at index {index} must be an object.")

        missing_fields = required_fields - document.keys()

        if missing_fields:
            raise ValueError(
                f"Document at index {index} is missing fields: {missing_fields}"
            )

    return documents