"""
Gallery Guide — RAG Engine
Hybrid search (dense + BM25) over Qdrant collection.
"""

import os
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
# from qdrant_client.models import (
#     Fusion,
#     FusionQuery,
#     Prefetch,
#     SparseVector,
# )
from qdrant_client.models import SparseVector
from qdrant_client import models as qmodels
load_dotenv()

COLLECTION   = "gallery_guide"
DENSE_MODEL  = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "prithivida/Splade_PP_en_v1"
QDRANT_HOST  = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT  = int(os.getenv("QDRANT_PORT", 6333))


class RAGEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        print("Initializing RAG engine…")
        self.client = QdrantClient(
            url=f"https://{QDRANT_HOST}",
            api_key=os.getenv("QDRANT_API_KEY", ""),
        )
        self.dense  = TextEmbedding(DENSE_MODEL)
        self.sparse = SparseTextEmbedding(SPARSE_MODEL)

    def _embed_dense(self, text: str) -> list[float]:
        return list(self.dense.embed([text]))[0].tolist()

    def _embed_sparse(self, text: str) -> SparseVector:
        r = list(self.sparse.embed([text]))[0]
        return SparseVector(indices=r.indices.tolist(), values=r.values.tolist())

    def search(self, query: str, limit: int = 5) -> list[dict]:
        try:
            dense_vec  = self._embed_dense(query)
            sparse_vec = self._embed_sparse(query)

            # results = self.client.query_points(
            #     collection_name=COLLECTION,
            #     prefetch=[
            #         Prefetch(query=dense_vec,  using="dense",  limit=12),
            #         Prefetch(query=sparse_vec, using="sparse", limit=12),
            #     ],
            #     query=FusionQuery(fusion=Fusion.RRF),
            #     limit=limit,
            #     with_payload=True,
            # )
            results = self.client.query_points(
                collection_name=COLLECTION,
                prefetch=[
                    qmodels.Prefetch(query=dense_vec,  using="dense",  limit=12),
                    qmodels.Prefetch(query=sparse_vec, using="sparse", limit=12),
                ],
                query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
            return [hit.payload for hit in results.points]
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def search_by_description(self, description: str, limit: int = 5) -> list[dict]:
        return self.search(description, limit=limit)


# Singleton
rag = RAGEngine()
