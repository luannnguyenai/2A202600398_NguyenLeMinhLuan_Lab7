"""
Run all experiments for REPORT.md:
  1) Chunking strategy comparison on all 6 data files
  2) Cosine similarity predictions on 5 sentence pairs
  3) Benchmark queries with search + search_with_filter + agent
"""
from __future__ import annotations
import json
from pathlib import Path

from src.chunking import (
    ChunkingStrategyComparator,
    FixedSizeChunker,
    SentenceChunker,
    RecursiveChunker,
    compute_similarity,
)
from src.embeddings import _mock_embed
from src.models import Document
from src.store import EmbeddingStore
from src.agent import KnowledgeBaseAgent

SEPARATOR = "=" * 70

# ─────────────────────────────────────────────────────────────
# PART A: Chunking Strategy Comparison
# ─────────────────────────────────────────────────────────────
def run_chunking_comparison():
    print(SEPARATOR)
    print("EXPERIMENT 1: Chunking Strategy Comparison")
    print(SEPARATOR)

    data_dir = Path("data")
    files = sorted(data_dir.glob("*.txt")) + sorted(data_dir.glob("*.md"))

    comparator = ChunkingStrategyComparator()

    for fpath in files:
        text = fpath.read_text(encoding="utf-8")
        char_count = len(text)
        print(f"\n--- {fpath.name} ({char_count} chars) ---")
        result = comparator.compare(text, chunk_size=500)
        for strategy, stats in result.items():
            print(f"  {strategy:15s}  count={stats['count']:3d}  avg_len={stats['avg_length']:7.1f}")
    print()


# ─────────────────────────────────────────────────────────────
# PART B: Cosine Similarity Predictions
# ─────────────────────────────────────────────────────────────
def run_similarity_predictions():
    print(SEPARATOR)
    print("EXPERIMENT 2: Cosine Similarity Predictions")
    print(SEPARATOR)

    pairs = [
        ("Python is a programming language used for AI.",
         "Python is widely used in machine learning and data science.",
         "high"),
        ("A vector store keeps embeddings for similarity search.",
         "Vector databases are used to retrieve semantically similar documents.",
         "high"),
        ("The customer support team handles billing issues.",
         "Recursive chunking splits text by paragraph boundaries.",
         "low"),
        ("Dogs are loyal companions and working animals.",
         "Cats are independent pets that enjoy sleeping.",
         "low"),
        ("Retrieval-augmented generation grounds answers in retrieved text.",
         "RAG systems use retrieved context to produce accurate responses.",
         "high"),
    ]

    for i, (sa, sb, prediction) in enumerate(pairs, 1):
        va = _mock_embed(sa)
        vb = _mock_embed(sb)
        score = compute_similarity(va, vb)
        label = "high" if score > 0.3 else "medium" if score > 0.1 else "low"
        match = "✓" if prediction == label else "✗"
        print(f"  Pair {i}: prediction={prediction:5s}  actual={score:.4f} ({label})  {match}")
        print(f"    A: {sa[:70]}...")
        print(f"    B: {sb[:70]}...")
    print()


# ─────────────────────────────────────────────────────────────
# PART C: Benchmark Queries
# ─────────────────────────────────────────────────────────────
def run_benchmark_queries():
    print(SEPARATOR)
    print("EXPERIMENT 3: Benchmark Queries")
    print(SEPARATOR)

    # Load all documents with metadata
    docs = [
        Document("python_intro",
                 Path("data/python_intro.txt").read_text(encoding="utf-8"),
                 {"source": "data/python_intro.txt", "category": "programming", "lang": "en"}),
        Document("vector_store_notes",
                 Path("data/vector_store_notes.md").read_text(encoding="utf-8"),
                 {"source": "data/vector_store_notes.md", "category": "ai_infrastructure", "lang": "en"}),
        Document("rag_system_design",
                 Path("data/rag_system_design.md").read_text(encoding="utf-8"),
                 {"source": "data/rag_system_design.md", "category": "ai_infrastructure", "lang": "en"}),
        Document("customer_support",
                 Path("data/customer_support_playbook.txt").read_text(encoding="utf-8"),
                 {"source": "data/customer_support_playbook.txt", "category": "support", "lang": "en"}),
        Document("chunking_experiment",
                 Path("data/chunking_experiment_report.md").read_text(encoding="utf-8"),
                 {"source": "data/chunking_experiment_report.md", "category": "ai_infrastructure", "lang": "en"}),
        Document("vi_retrieval_notes",
                 Path("data/vi_retrieval_notes.md").read_text(encoding="utf-8"),
                 {"source": "data/vi_retrieval_notes.md", "category": "ai_infrastructure", "lang": "vi"}),
    ]

    store = EmbeddingStore(collection_name="benchmark_store", embedding_fn=_mock_embed)
    store.add_documents(docs)
    print(f"\nStore size: {store.get_collection_size()} documents")

    def demo_llm(prompt: str) -> str:
        preview = prompt[:400].replace("\n", " ")
        return f"[DEMO LLM] {preview}..."

    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)

    # 5 benchmark queries with gold answers
    benchmark = [
        {
            "query": "What are the main use cases of Python in production?",
            "gold": "Python is used to build APIs, data pipelines, internal tools, and model-serving layers using frameworks like FastAPI, Django, Flask.",
        },
        {
            "query": "How does a vector search pipeline work?",
            "gold": "A vector search pipeline has 4 stages: chunk documents, embed each chunk, store vector and metadata, embed query and rank by similarity.",
        },
        {
            "query": "What is the goal of a RAG system?",
            "gold": "Build a retrieval-augmented generation system that finds relevant internal documents before producing an answer, reducing hallucinations by grounding responses in retrieved text.",
        },
        {
            "query": "How should support content be written for retrieval?",
            "gold": "Authors should specify exact pages, buttons, or log sources instead of vague statements, making chunks more useful for matching concrete troubleshooting terms.",
        },
        {
            "query": "Which chunking strategy performed best in experiments?",
            "gold": "Recursive chunking offered the best balance, preserving context while staying within target size range. It's a strong default for mixed technical documentation.",
        },
    ]

    print("\n--- Unfiltered Search Results ---\n")
    for i, bm in enumerate(benchmark, 1):
        q = bm["query"]
        results = store.search(q, top_k=3)
        print(f"  Q{i}: {q}")
        print(f"  Gold: {bm['gold'][:80]}...")
        for j, r in enumerate(results, 1):
            print(f"    [{j}] score={r['score']:.4f}  id={r['id']}  content={r['content'][:80].replace(chr(10), ' ')}...")
        answer = agent.answer(q, top_k=3)
        print(f"  Agent: {answer[:120]}...")
        print()

    # Metadata-filtered search demo
    print("\n--- Filtered Search Demo (category=ai_infrastructure) ---\n")
    q_filter = "How does a vector search pipeline work?"
    results_filtered = store.search_with_filter(
        q_filter, top_k=3, metadata_filter={"category": "ai_infrastructure"}
    )
    for j, r in enumerate(results_filtered, 1):
        print(f"  [{j}] score={r['score']:.4f}  id={r['id']}  category={r['metadata'].get('category')}")

    print("\n--- Filtered Search Demo (lang=vi) ---\n")
    q_vi = "Chất lượng chunking ảnh hưởng đến retrieval thế nào?"
    results_vi = store.search_with_filter(
        q_vi, top_k=3, metadata_filter={"lang": "vi"}
    )
    for j, r in enumerate(results_vi, 1):
        print(f"  [{j}] score={r['score']:.4f}  id={r['id']}  lang={r['metadata'].get('lang')}")

    print()


# ─────────────────────────────────────────────────────────────
# PART D: Delete Document Test
# ─────────────────────────────────────────────────────────────
def run_delete_test():
    print(SEPARATOR)
    print("EXPERIMENT 4: Delete Document")
    print(SEPARATOR)

    store = EmbeddingStore(collection_name="delete_test", embedding_fn=_mock_embed)
    docs = [
        Document("keep_doc", "Content to keep", {}),
        Document("remove_doc", "Content to remove", {}),
    ]
    store.add_documents(docs)
    print(f"  Before delete: {store.get_collection_size()} docs")
    result = store.delete_document("remove_doc")
    print(f"  delete_document('remove_doc') -> {result}")
    print(f"  After delete: {store.get_collection_size()} docs")
    result2 = store.delete_document("nonexistent")
    print(f"  delete_document('nonexistent') -> {result2}")
    print()


if __name__ == "__main__":
    run_chunking_comparison()
    run_similarity_predictions()
    run_benchmark_queries()
    run_delete_test()
    print(SEPARATOR)
    print("ALL EXPERIMENTS COMPLETE")
    print(SEPARATOR)
