"""
ScholarBridge Evidence & RAG Configuration
==========================================
Defines environment-driven settings for Qdrant, BM25, embeddings,
hybrid retrieval fusion, reranking, and citation provenance.
"""

import os
from pathlib import Path

# Base Paths
PACKAGE_DIR = Path(__file__).resolve().parent.parent
EVIDENCE_DATA_DIR = PACKAGE_DIR / "data" / "evidence"
EVIDENCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_QDRANT_STORAGE = str(EVIDENCE_DATA_DIR / "qdrant_db")

# Qdrant Vector Database Settings (File-based persistence by default)
QDRANT_PATH = os.environ.get("SCHOLARBRIDGE_QDRANT_PATH", DEFAULT_QDRANT_STORAGE)
QDRANT_URL = os.environ.get("SCHOLARBRIDGE_QDRANT_URL", None)
QDRANT_COLLECTION_NAME = os.environ.get("SCHOLARBRIDGE_QDRANT_COLLECTION", "scholarbridge_evidence")

# Embedding Provider Settings
EMBEDDING_PROVIDER = os.environ.get("SCHOLARBRIDGE_EMBEDDING_PROVIDER", "deterministic")
EMBEDDING_MODEL_NAME = os.environ.get("SCHOLARBRIDGE_EMBEDDING_MODEL", "deterministic-feature-hash-384")
EMBEDDING_DIMENSIONS = int(os.environ.get("SCHOLARBRIDGE_EMBEDDING_DIM", "384"))

# BM25 Keyword Search Settings
BM25_K1 = float(os.environ.get("SCHOLARBRIDGE_BM25_K1", "1.5"))
BM25_B = float(os.environ.get("SCHOLARBRIDGE_BM25_B", "0.75"))

# Hybrid Retrieval & Fusion Settings
RETRIEVAL_TOP_K_DENSE = int(os.environ.get("SCHOLARBRIDGE_DENSE_TOP_K", "30"))
RETRIEVAL_TOP_K_BM25 = int(os.environ.get("SCHOLARBRIDGE_BM25_TOP_K", "30"))
FUSION_METHOD = os.environ.get("SCHOLARBRIDGE_FUSION_METHOD", "rrf")  # 'rrf' or 'weighted_sum'
RRF_K_CONSTANT = int(os.environ.get("SCHOLARBRIDGE_RRF_K", "60"))
DENSE_FUSION_WEIGHT = float(os.environ.get("SCHOLARBRIDGE_DENSE_WEIGHT", "0.55"))
BM25_FUSION_WEIGHT = float(os.environ.get("SCHOLARBRIDGE_BM25_WEIGHT", "0.45"))

# Reranker Settings
RERANKER_TYPE = os.environ.get("SCHOLARBRIDGE_RERANKER", "cross_feature") # 'cross_feature', 'cross_encoder', 'noop'
RERANKER_TOP_N = int(os.environ.get("SCHOLARBRIDGE_RERANKER_TOP_N", "10"))

# Chunking Parameters (Structure-aware section and table boundaries)
DEFAULT_CHUNK_SIZE_TOKENS = 350
DEFAULT_ACADEMIC_YEAR = "2026-27"
DEFAULT_JURISDICTION = "Gujarat"
