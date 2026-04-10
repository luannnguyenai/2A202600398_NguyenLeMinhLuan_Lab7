from __future__ import annotations
import os
from pathlib import Path
from src.chunking import RecursiveChunker, compute_similarity
from src.embeddings import _mock_embed
from src.models import Document
from src.store import EmbeddingStore
from src.agent import KnowledgeBaseAgent

def test_law_document():
    path = Path("data/Luat_Dan_Su.md")
    if not path.exists():
        print(f"File not found: {path}")
        return

    text = path.read_text(encoding="utf-8")
    print(f"--- Processing {path.name} ({len(text)} characters) ---")

    # 1. Chunking
    chunker = RecursiveChunker(chunk_size=800) # Tăng size một chút cho văn bản luật
    chunks = chunker.chunk(text)
    print(f"Created {len(chunks)} chunks using RecursiveChunker.")
    print(f"Average chunk length: {sum(len(c) for c in chunks)/len(chunks):.1f}")

    # 2. Store
    store = EmbeddingStore(collection_name="law_store", embedding_fn=_mock_embed)
    docs = [Document(id=f"law_part_{i}", content=chunk, metadata={"source": str(path)}) 
            for i, chunk in enumerate(chunks)]
    store.add_documents(docs)
    
    # 3. Agent
    def mock_llm(prompt: str) -> str:
        return "[MOCK LLM] Trả lời dựa trên các Điều khoản được trích xuất từ Bộ luật Dân sự."
    
    agent = KnowledgeBaseAgent(store=store, llm_fn=mock_llm)

    # 4. Queries
    queries = [
        "Các nguyên tắc cơ bản của pháp luật dân sự là gì?",
        "Năng lực hành vi dân sự của người từ đủ 15 đến dưới 18 tuổi?",
        "Quyền thay đổi họ của cá nhân được quy định trong trường hợp nào?",
        "Điều kiện để một cá nhân được làm người giám hộ?",
        "Khi nào một người bị Tòa án tuyên bố là mất tích?"
    ]

    print("\n" + "="*50)
    print("RESULTS FOR LUAT_DAN_SU.MD")
    print("="*50)

    for i, q in enumerate(queries, 1):
        print(f"\nQUERY {i}: {q}")
        results = store.search(q, top_k=2)
        for j, res in enumerate(results, 1):
            print(f"  Match {j} (Score: {res['score']:.4f}):")
            content_preview = res['content'].strip().replace('\n', ' ')[:150]
            print(f"    Content: {content_preview}...")
        
        # Kiểm tra xem top match có chứa đúng Điều luật không
        # (Vì là Mock Embedder nên score sẽ phụ thuộc vào hash, nhưng ta xem nội dung)
    
    print("\n" + "="*50)

if __name__ == "__main__":
    test_law_document()
