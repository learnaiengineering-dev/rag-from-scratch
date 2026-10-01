from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    index: int


def chunk_text(text: str, source: str = "doc", size: int = 120, overlap: int = 20) -> list[Chunk]:
    """Split text into word windows of `size` words, overlapping by `overlap`."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    words = text.split()
    step = size - overlap
    chunks = []
    for i, start in enumerate(range(0, max(len(words), 1), step)):
        window = words[start : start + size]
        if not window:
            break
        chunks.append(Chunk(" ".join(window), source, i))
        if start + size >= len(words):
            break
    return chunks

