"""
Gallery Guide — RAG Engine
Hybrid search (dense + BM25) over Qdrant collection.
"""

import logging
import os
import time

from dotenv import load_dotenv
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector
from qdrant_client import models as qmodels

load_dotenv()
logger = logging.getLogger("rag")

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
        t = time.perf_counter()
        result = list(self.dense.embed([text]))[0].tolist()
        logger.info("  dense_embed  %5.0f ms", (time.perf_counter() - t) * 1000)
        return result

    def _embed_sparse(self, text: str) -> SparseVector:
        t = time.perf_counter()
        r = list(self.sparse.embed([text]))[0]
        logger.info("  sparse_embed %5.0f ms", (time.perf_counter() - t) * 1000)
        return SparseVector(indices=r.indices.tolist(), values=r.values.tolist())

    def search(self, query: str, limit: int = 5) -> list[dict]:
        t0 = time.perf_counter()
        try:
            dense_vec  = self._embed_dense(query)
            sparse_vec = self._embed_sparse(query)

            t_q = time.perf_counter()
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
            logger.info("  qdrant_query %5.0f ms | hits=%d",
                        (time.perf_counter() - t_q) * 1000, len(results.points))
            logger.info("  rag_total    %5.0f ms", (time.perf_counter() - t0) * 1000)
            return [hit.payload for hit in results.points]
        except Exception as e:
            logger.error("search_error: %s", e)
            return []

    def search_by_description(self, description: str, limit: int = 5) -> list[dict]:
        return self.search(description, limit=limit)


# Singleton
rag = RAGEngine()
