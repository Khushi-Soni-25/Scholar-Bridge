"""
ScholarBridge Deterministic Feature-Hash Embedding Layer
========================================================
Provides cryptographically stable, cross-process deterministic dense vector
generation using SHA-256 feature hashing with L2 unit normalization.
"""

import hashlib
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from .config import EMBEDDING_PROVIDER, EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSIONS


class BaseEmbeddingModel(ABC):
    """Abstract interface for all embedding providers."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generates a dense vector embedding for a single text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates dense vector embeddings for a list of texts."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Returns the vector dimensionality."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Returns model identifier."""
        pass


def stable_token_hash(token: str, modulo: int) -> int:
    """
    Cryptographically stable hash function across Python processes and restarts.
    Uses SHA-256 byte digest rather than process-randomized built-in hash().
    """
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    # Use first 8 bytes as big-endian unsigned integer
    return int.from_bytes(digest[:8], byteorder="big") % modulo


class DeterministicFeatureHashEmbeddingModel(BaseEmbeddingModel):
    """
    Cryptographically stable, cross-process deterministic feature-hash embedding model.
    Maps text tokens, bi-grams, and subwords to a 384-dimensional dense space using
    stable SHA-256 hashing and L2 normalization.
    Guarantees that identical text produces byte-for-byte identical embeddings across:
    - Independent Python interpreter processes
    - Application restarts
    - Operating systems and environments
    """

    def __init__(
        self,
        dimensions: int = EMBEDDING_DIMENSIONS,
        model_name: str = "deterministic-feature-hash-384"
    ):
        self.dimensions = dimensions
        self.model_name = model_name

    def _hash_vectorize(self, text: str) -> np.ndarray:
        clean_text = text.lower().strip()
        tokens = clean_text.split()
        vec = np.zeros(self.dimensions, dtype=np.float32)
        
        if not tokens:
            return vec

        # Token + Bi-gram + Subword stable hash projection
        for i, tok in enumerate(tokens):
            # 1-gram
            h1 = stable_token_hash(tok, self.dimensions)
            vec[h1] += 1.0
            
            # 2-gram
            if i < len(tokens) - 1:
                bigram = f"{tok}_{tokens[i+1]}"
                h2 = stable_token_hash(bigram, self.dimensions)
                vec[h2] += 1.5

            # 3-char prefix subwords for codes / numbers (e.g. MYSY, 250000, SC)
            if len(tok) >= 3:
                h3 = stable_token_hash(tok[:3], self.dimensions)
                vec[h3] += 0.5

        # L2 Normalization for unit vector cosine similarity
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed_text(self, text: str) -> List[float]:
        return self._hash_vectorize(text).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_vectorize(t).tolist() for t in texts]

    def get_dimension(self) -> int:
        return self.dimensions

    def get_model_name(self) -> str:
        return self.model_name


# Backwards compatibility alias
DeterministicEmbeddingModel = DeterministicFeatureHashEmbeddingModel

# Singleton cache
_EMBEDDING_INSTANCE: Optional[BaseEmbeddingModel] = None

def get_embedding_model(provider: Optional[str] = None) -> BaseEmbeddingModel:
    """Factory returning configured embedding model."""
    global _EMBEDDING_INSTANCE
    selected_provider = provider or EMBEDDING_PROVIDER
    
    if _EMBEDDING_INSTANCE is None:
        _EMBEDDING_INSTANCE = DeterministicFeatureHashEmbeddingModel(
            dimensions=EMBEDDING_DIMENSIONS,
            model_name=f"{selected_provider}:{EMBEDDING_MODEL_NAME}"
        )
    return _EMBEDDING_INSTANCE
