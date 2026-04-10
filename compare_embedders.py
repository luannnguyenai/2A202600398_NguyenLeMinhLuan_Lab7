"""
compare_embedders.py
====================
So sanh hieu qua retrieval giua MockEmbedder va Jina v5 Nano (LM Studio)
tren file Luat_Dan_Su.md voi 5 benchmark queries phap ly.

Chay:
    python3 compare_embedders.py
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv

from src.chunking import RecursiveChunker
from src.embeddings import _mock_embed, OpenAIEmbedder, MockEmbedder
from src.models import Document
from src.store import EmbeddingStore

# ─── Load .env (OPENAI_API_BASE, OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL) ──────
load_dotenv(override=False)

LAW_FILE = Path("data/Luat_Dan_Su.md")
CHUNK_SIZE = 1000

# 5 benchmark queries voi expected answer (dieu luat lien quan)
BENCHMARK = [
    {
        "query": "Các nguyên tắc cơ bản của pháp luật dân sự là gì?",
        "expected_keyword": "Điều 3",
    },
    {
        "query": "Năng lực hành vi dân sự của người từ đủ 15 đến dưới 18 tuổi quy định thế nào?",
        "expected_keyword": "Điều 21",
    },
    {
        "query": "Cá nhân có quyền thay đổi họ trong những trường hợp nào?",
        "expected_keyword": "Điều 27",
    },
    {
        "query": "Điều kiện để một cá nhân được làm người giám hộ là gì?",
        "expected_keyword": "Điều 49",
    },
    {
        "query": "Tòa án tuyên bố một người mất tích khi nào?",
        "expected_keyword": "Điều 68",
    },
]

SEP = "=" * 65


def load_and_chunk() -> list[Document]:
    """Chunk Luat_Dan_Su.md once, reuse for both embedders."""
    if not LAW_FILE.exists():
        print(f"[ERROR] File not found: {LAW_FILE}")
        raise SystemExit(1)

    text = LAW_FILE.read_text(encoding="utf-8")
    chunks = RecursiveChunker(chunk_size=CHUNK_SIZE).chunk(text)
    print(f"Chunked {LAW_FILE.name} → {len(chunks)} chunks  "
          f"(avg {sum(len(c) for c in chunks) // len(chunks)} chars)")
    # NOMIC v1.5 requires 'search_document: ' prefix
    return [
        Document(id=f"chunk_{i}", content=f"search_document: {c}",
                 metadata={"source": LAW_FILE.name, "chunk_idx": i})
        for i, c in enumerate(chunks)
    ]


def build_store(name: str, embedder) -> EmbeddingStore:
    docs = load_and_chunk()
    store = EmbeddingStore(collection_name=name, embedding_fn=embedder)
    t0 = time.time()
    store.add_documents(docs)
    elapsed = time.time() - t0
    print(f"Indexed {store.get_collection_size()} chunks in {elapsed:.1f}s\n")
    return store


def relevant(result: dict, expected_keyword: str) -> bool:
    """Check if expected Dieu N appears in the returned chunk content."""
    return expected_keyword in result["content"]


def run_benchmark(store: EmbeddingStore, embedder_label: str) -> dict:
    """Run all 5 queries, return precision summary."""
    print(f"\n{'─'*65}")
    print(f"  EMBEDDER: {embedder_label}")
    print(f"{'─'*65}")

    total_relevant = 0
    query_results = []

    for bm in BENCHMARK:
        q = bm["query"]
        kw = bm["expected_keyword"]
        # NOMIC v1.5 requires 'search_query: ' prefix
        search_q = f"search_query: {q}" if "Nomic" in embedder_label else q
        
        t0 = time.time()
        results = store.search(search_q, top_k=3)
        latency = (time.time() - t0) * 1000  # ms

        top3_content = [r["content"] for r in results]
        hit = any(relevant(r, kw) for r in results)
        if hit:
            total_relevant += 1

        print(f"\n  Q: {q}")
        print(f"  Expected to find: {kw}  |  Found in top-3: {'✓ YES' if hit else '✗ NO'}  "
              f"[{latency:.0f}ms]")
        for j, r in enumerate(results, 1):
            snippet = r["content"].replace("\n", " ").strip()[:90]
            marker = "  ← ✓" if relevant(r, kw) else ""
            print(f"    [{j}] score={r['score']:.4f}  {snippet}...{marker}")

        query_results.append({"query": q, "hit": hit, "latency_ms": latency})

    precision = total_relevant / len(BENCHMARK) * 100
    print(f"\n  → Retrieval Precision: {total_relevant}/{len(BENCHMARK)} = {precision:.0f}%")
    return {"label": embedder_label, "precision": precision, "results": query_results}


def print_comparison(mock_stats: dict, jina_stats: dict) -> None:
    print(f"\n{SEP}")
    print("  COMPARISON SUMMARY")
    print(SEP)
    print(f"  {'Query':<55} {'Mock':>6} {'Jina':>6}")
    print(f"  {'─'*55} {'─'*6} {'─'*6}")
    for m, j in zip(mock_stats["results"], jina_stats["results"]):
        m_icon = "✓" if m["hit"] else "✗"
        j_icon = "✓" if j["hit"] else "✗"
        short_q = m["query"][:52] + "..." if len(m["query"]) > 52 else m["query"]
        print(f"  {short_q:<55} {m_icon:>6} {j_icon:>6}")
    print(f"  {'─'*55} {'─'*6} {'─'*6}")
    print(f"  {'Precision':<55} "
          f"{mock_stats['precision']:>5.0f}% {jina_stats['precision']:>5.0f}%")
    delta = jina_stats["precision"] - mock_stats["precision"]
    sign = "+" if delta >= 0 else ""
    print(f"\n  Jina vs Mock improvement: {sign}{delta:.0f}%")
    print(SEP)


def main():
    print(SEP)
    print("  MOCK vs JINA v5 NANO — Retrieval Comparison")
    print(f"  File: {LAW_FILE.name}  |  Chunk size: {CHUNK_SIZE}")
    print(SEP)

    # ── 1) Mock embedder ──────────────────────────────────────────────────
    print("\n[STEP 1] Building store with MockEmbedder...")
    mock_store = build_store("mock_law", _mock_embed)
    mock_stats = run_benchmark(mock_store, "MockEmbedder (MD5 hash, no semantics)")

    # ── 2) Jina via LM Studio ─────────────────────────────────────────────
    print("\n\n[STEP 2] Building store with Jina v5 Nano (LM Studio)...")
    jina_model = os.getenv("OPENAI_EMBEDDING_MODEL", "jina-embeddings-v5-text-nano-retrieval")
    base_url = os.getenv("OPENAI_API_BASE", "http://localhost:1234/v1")
    print(f"  → base_url : {base_url}")
    print(f"  → model    : {jina_model}")
    print("  → Make sure LM Studio Server is running!\n")

    try:
        jina_embedder = OpenAIEmbedder()  # reads from .env automatically
        # quick smoke test
        _ = jina_embedder("test connection")
        jina_store = build_store("jina_law", jina_embedder)
        jina_stats = run_benchmark(jina_store, f"Jina v5 Nano via LM Studio ({jina_model})")
        print_comparison(mock_stats, jina_stats)

    except Exception as exc:
        print(f"\n  [!] Could not connect to LM Studio: {exc}")
        print("  Please check:")
        print("    1. LM Studio is open")
        print("    2. 'Local Server' tab → model loaded → 'Start Server' clicked")
        print(f"    3. Server is running at {base_url}")
        print("\n  Showing Mock results only:")
        print(f"  MockEmbedder Precision: {mock_stats['precision']:.0f}%")
        print(f"  (Expected Jina precision: 80-100% for this structured legal document)")


if __name__ == "__main__":
    main()
