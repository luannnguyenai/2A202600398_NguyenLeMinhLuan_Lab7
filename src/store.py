from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb

            client = chromadb.Client()
            self._collection = client.get_or_create_collection(name=collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """Build a normalized stored record for one document."""
        embedding = self._embedding_fn(doc.content)
        return {
            "id": doc.id,
            "content": doc.content,
            "embedding": embedding,
            "metadata": dict(doc.metadata) if doc.metadata else {},
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Run in-memory similarity search over provided records."""
        if not records:
            return []

        query_embedding = self._embedding_fn(query)

        scored = []
        for record in records:
            score = _dot(query_embedding, record["embedding"])
            scored.append({
                "id": record["id"],
                "content": record["content"],
                "metadata": record["metadata"],
                "score": score,
            })

        # Sort by score descending, take top_k
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if self._use_chroma and self._collection is not None:
            ids = []
            documents = []
            embeddings = []
            metadatas = []
            for doc in docs:
                embedding = self._embedding_fn(doc.content)
                ids.append(doc.id)
                documents.append(doc.content)
                embeddings.append(embedding)
                metadatas.append(doc.metadata if doc.metadata else {})
            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        else:
            for doc in docs:
                record = self._make_record(doc)
                self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma and self._collection is not None:
            query_embedding = self._embedding_fn(query)
            n_results = min(top_k, self._collection.count())
            if n_results == 0:
                return []
            chroma_results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
            )
            results = []
            ids = chroma_results.get("ids", [[]])[0]
            documents = chroma_results.get("documents", [[]])[0]
            metadatas = chroma_results.get("metadatas", [[]])[0]
            distances = chroma_results.get("distances", [[]])[0]
            for i in range(len(ids)):
                # ChromaDB returns distances (lower = more similar), convert to score
                score = 1.0 - distances[i] if distances else 0.0
                results.append({
                    "id": ids[i],
                    "content": documents[i],
                    "metadata": metadatas[i] if metadatas else {},
                    "score": score,
                })
            return results
        else:
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if metadata_filter is None:
            return self.search(query, top_k=top_k)

        if self._use_chroma and self._collection is not None:
            query_embedding = self._embedding_fn(query)
            # Build ChromaDB where clause
            where_clause = {}
            if len(metadata_filter) == 1:
                key, value = next(iter(metadata_filter.items()))
                where_clause = {key: {"$eq": value}}
            elif len(metadata_filter) > 1:
                where_clause = {
                    "$and": [{k: {"$eq": v}} for k, v in metadata_filter.items()]
                }
            n_results = min(top_k, self._collection.count())
            if n_results == 0:
                return []
            chroma_results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause if where_clause else None,
            )
            results = []
            ids = chroma_results.get("ids", [[]])[0]
            documents = chroma_results.get("documents", [[]])[0]
            metadatas = chroma_results.get("metadatas", [[]])[0]
            distances = chroma_results.get("distances", [[]])[0]
            for i in range(len(ids)):
                score = 1.0 - distances[i] if distances else 0.0
                results.append({
                    "id": ids[i],
                    "content": documents[i],
                    "metadata": metadatas[i] if metadatas else {},
                    "score": score,
                })
            return results
        else:
            # In-memory: filter records by metadata
            filtered_records = []
            for record in self._store:
                meta = record.get("metadata", {})
                if all(meta.get(k) == v for k, v in metadata_filter.items()):
                    filtered_records.append(record)
            return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma and self._collection is not None:
            # ChromaDB: delete by id
            try:
                existing = self._collection.get(ids=[doc_id])
                if existing and existing.get("ids") and len(existing["ids"]) > 0:
                    self._collection.delete(ids=[doc_id])
                    return True
                return False
            except Exception:
                return False
        else:
            before = len(self._store)
            self._store = [
                record for record in self._store
                if record.get("id") != doc_id and record.get("metadata", {}).get("doc_id") != doc_id
            ]
            after = len(self._store)
            return after < before
