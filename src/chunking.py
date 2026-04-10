from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        # Split text into sentences using regex for ". ", "! ", "? ", or ".\n"
        # We keep the delimiter by using a capturing group and re-attach it
        parts = re.split(r'(?<=[.!?])(?:\s|\n)', text)
        sentences = [s.strip() for s in parts if s.strip()]

        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunk = " ".join(group).strip()
            if chunk:
                chunks.append(chunk)

        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        separators = list(self.separators) if self.separators else [""]
        return self._split(text, separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Base case: text fits in chunk_size
        if len(current_text) <= self.chunk_size:
            return [current_text] if current_text.strip() else []

        # No separators left — force-split character by character
        if not remaining_separators:
            results = []
            for i in range(0, len(current_text), self.chunk_size):
                piece = current_text[i : i + self.chunk_size]
                if piece:
                    results.append(piece)
            return results

        sep = remaining_separators[0]
        next_separators = remaining_separators[1:]

        if sep == "":
            # Split every character — just fixed-size cut
            results = []
            for i in range(0, len(current_text), self.chunk_size):
                piece = current_text[i : i + self.chunk_size]
                if piece:
                    results.append(piece)
            return results

        # Try to split by current separator
        parts = current_text.split(sep)

        # If splitting gives only one part, the separator wasn't found — try next
        if len(parts) <= 1:
            return self._split(current_text, next_separators)

        # Merge small parts together, recurse on parts that are still too large
        results: list[str] = []
        current_group = ""

        for part in parts:
            candidate = (current_group + sep + part).strip() if current_group else part.strip()

            if len(candidate) <= self.chunk_size:
                current_group = candidate
            else:
                # Flush current group
                if current_group.strip():
                    results.append(current_group.strip())
                # Process this part: if it's too large, recurse with next separators
                if len(part.strip()) > self.chunk_size:
                    results.extend(self._split(part.strip(), next_separators))
                    current_group = ""
                else:
                    current_group = part.strip()

        if current_group.strip():
            results.append(current_group.strip())

        return [r for r in results if r]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    mag_a = math.sqrt(sum(x * x for x in vec_a))
    mag_b = math.sqrt(sum(x * x for x in vec_b))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return _dot(vec_a, vec_b) / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=0),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }

        result = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            count = len(chunks)
            avg_length = sum(len(c) for c in chunks) / count if count > 0 else 0.0
            result[name] = {
                "count": count,
                "avg_length": round(avg_length, 2),
                "chunks": chunks,
            }

        return result
