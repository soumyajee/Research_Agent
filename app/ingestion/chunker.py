import re

import numpy as np

from app.ingestion.embeddings import embed_texts


def token_chunk(
    text: str,
    chunk_size: int = 450,
    overlap: int = 70,
):
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))

        if end >= len(words):
            break

        start = max(end - overlap, start + 1)

    return chunks


def sentence_split(text: str):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


async def semantic_chunk(
    text: str,
    max_words: int = 450,
    similarity_threshold: float = 0.55,
):
    sentences = sentence_split(text)

    if len(sentences) <= 1:
        return token_chunk(text)

    vectors = np.asarray(await embed_texts(sentences))

    chunks = []
    current = [sentences[0]]
    current_word_count = len(sentences[0].split())

    for index in range(1, len(sentences)):
        similarity = float(np.dot(vectors[index - 1], vectors[index]))
        sentence_words = len(sentences[index].split())

        split_here = (
            similarity < similarity_threshold
            or current_word_count + sentence_words > max_words
        )

        if split_here:
            chunks.append(" ".join(current))
            current = [sentences[index]]
            current_word_count = sentence_words
        else:
            current.append(sentences[index])
            current_word_count += sentence_words

    if current:
        chunks.append(" ".join(current))

    return chunks


async def hybrid_chunk(text: str, source_type: str):
    semantic_sources = {
        "web",
        "web_search",
        "pdf",
        "docx",
        "article",
        "research_report",
    }

    if source_type in semantic_sources:
        return await semantic_chunk(text)

    return token_chunk(text)
