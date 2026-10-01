# rag-from-scratch

Retrieval-Augmented Generation with **zero retrieval dependencies**: a chunker, a BM25 retriever and a citation-aware prompt, all in plain Python you can read in one sitting. Understand the pieces before you reach for a vector database.

## Why BM25 first?

Keyword retrieval is a strong, cheap baseline. Measure it before adding embeddings; if it already answers your questions, you've saved infrastructure. When you do add embeddings, hybrid (BM25 + vectors) usually beats either alone.

## Pipeline

```
documents -> chunk() -> BM25 index -> search(question, k) -> prompt with numbered sources -> LLM -> answer + sources
```

## Quickstart

```bash
git clone https://github.com/learnaiengineering-dev/rag-from-scratch
cd rag-from-scratch
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python examples/demo.py          # runs offline with a stub LLM
```

To use a real model, pass any `Callable[[str], str]`:

```python
import anthropic
from rag_scratch import RAG, BM25, chunk_text

client = anthropic.Anthropic()
def llm(prompt: str) -> str:
    r = client.messages.create(model="claude-sonnet-5-5", max_tokens=512,
                               messages=[{"role": "user", "content": prompt}])
    return r.content[0].text

docs = {"handbook.md": open("handbook.md").read()}
chunks = [c for name, t in docs.items() for c in chunk_text(t, source=name)]
rag = RAG(llm, BM25(chunks), k=3)
print(rag.answer("What is the refund policy?"))
```

## Things to try (exercises)

1. Change chunk size/overlap and watch retrieval quality shift.
2. Add a reranker step.
3. Add embeddings and fuse scores (reciprocal rank fusion).
4. Build an eval set and measure recall@k (see the `llm-eval-harness` repo).

---

## Part of Learn AI Engineering

This repo is a free resource from [Learn AI Engineering](https://learnaiengineering.dev/?utm_source=github&utm_medium=repo&utm_campaign=rag-from-scratch) — *Build production AI systems.*

- Structured learning paths, labs and portfolio projects: [https://learnaiengineering.dev](https://learnaiengineering.dev/?utm_source=github&utm_medium=repo&utm_campaign=rag-from-scratch)
- Want the production-ready version (enterprise templates, deployment, evals, runbooks)? See the **FDE Toolkit** on the portal.
- More free repos: [github.com/learnaiengineering-dev](https://github.com/learnaiengineering-dev)

Licensed under MIT.
