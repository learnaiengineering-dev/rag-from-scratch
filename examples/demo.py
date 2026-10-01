from rag_scratch import BM25, RAG, chunk_text

DOCS = {
    "refunds.md": "Refunds are available within 30 days of purchase. Digital products are refunded in full.",
    "support.md": "Support is available Monday to Friday, 9am to 5pm AEST by email.",
}

chunks = [c for name, text in DOCS.items() for c in chunk_text(text, source=name)]
rag = RAG(lambda prompt: "(stub LLM) Here is the prompt I would send:\n" + prompt, BM25(chunks))
answer, sources = rag.answer("How long do I have to ask for a refund?")
print(answer)

