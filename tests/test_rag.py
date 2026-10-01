import pytest

from rag_scratch import BM25, RAG, chunk_text


def test_chunk_overlap():
    text = " ".join(str(i) for i in range(10))
    chunks = chunk_text(text, size=4, overlap=1)
    assert chunks[0].text == "0 1 2 3"
    assert chunks[1].text.startswith("3 ")
    assert chunks[-1].text.endswith("9")


def test_chunk_validates():
    with pytest.raises(ValueError):
        chunk_text("a b c", size=2, overlap=2)


def test_bm25_ranks_relevant_first():
    docs = {
        "a": "cats purr and sleep all day",
        "b": "refunds are available within thirty days",
        "c": "the weather is sunny",
    }
    chunks = [c for n, t in docs.items() for c in chunk_text(t, source=n)]
    top = BM25(chunks).search("refund days", k=1)[0][0]
    assert top.source == "b"


def test_no_hits_skips_llm():
    called = []
    rag = RAG(lambda p: called.append(p) or "x", BM25(chunk_text("hello world")))
    answer, sources = rag.answer("zebra")
    assert sources == [] and not called


def test_prompt_contains_numbered_sources():
    chunks = chunk_text("refunds within 30 days", source="r.md")
    rag = RAG(lambda p: p, BM25(chunks))
    answer, sources = rag.answer("refunds")
    assert "[1] (r.md)" in answer and sources
