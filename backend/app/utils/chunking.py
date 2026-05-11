from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TextChunk:
    index: int
    text: str


def sliding_window_chunks(text: str, chunk_size: int, overlap: int) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[TextChunk] = []
    start = 0
    index = 0
    step = chunk_size - overlap
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(TextChunk(index=index, text=chunk))
            index += 1
        start += step
    return chunks
