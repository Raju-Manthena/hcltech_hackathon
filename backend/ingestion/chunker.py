import re


def split_sentences(text: str) -> list[str]:
    """Split text into sentences."""

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    return [sentence.strip() for sentence in sentences if sentence.strip()]


def create_chunks(
    documents: list[dict],
    chunk_size: int = 5,
    overlap: int = 1
) -> list[dict]:
    """
    Convert documents into sentence-based chunks.

    Each chunk retains its source document metadata.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    chunks = []
    chunk_counter = 0

    for document in documents:

        sentences = split_sentences(document["content"])

        if not sentences:
            continue

        start = 0

        while start < len(sentences):

            end = min(start + chunk_size, len(sentences))

            chunk_text = " ".join(sentences[start:end])

            chunks.append({
                "chunk_id": f"chunk_{chunk_counter}",
                "doc_id": document["doc_id"],
                "title": document["title"],
                "text": chunk_text
            })

            chunk_counter += 1

            if end == len(sentences):
                break

            start = end - overlap

    return chunks