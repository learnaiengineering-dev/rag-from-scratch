from typing import Callable

from .bm25 import BM25
from .chunk import Chunk

PROMPT = """Answer the question using ONLY the numbered sources below.
Cite sources like [1]. If the sources do not contain the answer, say you don't know.

{context}

Question: {question}
Answer:"""


class RAG:
    def __init__(self, llm: Callable[[str], str], retriever: BM25, k: int = 3) -> None:
        self.llm, self.retriever, self.k = llm, retriever, k

    def build_prompt(self, question: str, hits: list[Chunk]) -> str:
        context = "\n\n".join(f"[{i}] ({c.source}) {c.text}" for i, c in enumerate(hits, 1))
        return PROMPT.format(context=context, question=question)

    def answer(self, question: str) -> tuple[str, list[Chunk]]:
        hits = [c for c, _ in self.retriever.search(question, self.k)]
        if not hits:
            return "I don't know: no relevant sources found.", []
        return self.llm(self.build_prompt(question, hits)), hits

