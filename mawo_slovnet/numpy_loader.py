"""Numpy-based model loader for SlovNet models.

Loads .tar model files and provides numpy-only inference.
No dependencies on original slovnet/navec packages.

Based on:
- github.com/natasha/slovnet architecture
- Pure numpy implementation for independence
"""

from __future__ import annotations

import gzip
import json
import logging
import tarfile
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class Vocab:
    """Vocabulary mapping between strings and indices."""

    def __init__(self, items: list[str]) -> None:
        """Initialize vocabulary.

        Args:
            items: List of vocabulary items (index = position)
        """
        self.items = items
        self.item_to_index = {item: i for i, item in enumerate(items)}

    def __len__(self) -> int:
        """Get vocabulary size."""
        return len(self.items)

    def __getitem__(self, key: int | str) -> str | int:
        """Get item by index or index by item."""
        if isinstance(key, int):
            return self.items[key]
        return self.item_to_index.get(key, 0)  # Return 0 (<pad>) if not found

    def __contains__(self, item: str) -> bool:
        """Check if item in vocabulary."""
        return item in self.item_to_index


class NavecEmbedding:
    """Navec embeddings loaded from model."""

    def __init__(
        self,
        indexes: np.ndarray,
        codes: np.ndarray,
        words: Vocab,
    ) -> None:
        """Initialize Navec embedding.

        Args:
            indexes: Quantized indexes (vocab_size, num_codes)
            codes: Codebook (num_codes, num_centroids, dim_per_code)
            words: Word vocabulary
        """
        self.indexes = indexes
        self.codes = codes
        self.words = words

    def __getitem__(self, word: str) -> np.ndarray:
        """Get embedding for word.

        Args:
            word: Word to lookup

        Returns:
            Embedding vector
        """
        word_id = self.words[word] if isinstance(word, str) else word
        if word_id >= len(self.indexes):
            word_id = 0  # <pad>

        # Decode quantized embedding
        word_indexes = self.indexes[word_id]  # (num_codes,)
        embedding_parts = []

        for code_id, centroid_id in enumerate(word_indexes):
            embedding_parts.append(self.codes[code_id, centroid_id])

        return np.concatenate(embedding_parts)


class ModelLoader:
    """Loads SlovNet models from .tar files."""

    def __init__(self, tar_path: str | Path) -> None:
        """Initialize loader.

        Args:
            tar_path: Path to .tar model file
        """
        self.tar_path = Path(tar_path)
        if not self.tar_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.tar_path}")

        self.tar = tarfile.open(self.tar_path, "r")
        self.meta = self._load_meta()
        self.model_spec = self._load_model_spec()
        self.arrays = self._load_arrays()
        self.vocabs = self._load_vocabs()

    def _load_meta(self) -> dict[str, Any]:
        """Load meta.json."""
        content = self.tar.extractfile("meta.json").read()
        return json.loads(content)

    def _load_model_spec(self) -> dict[str, Any]:
        """Load model.json specification."""
        content = self.tar.extractfile("model.json").read()
        return json.loads(content)

    def _load_arrays(self) -> list[np.ndarray]:
        """Load all binary arrays from arrays/ directory."""
        arrays = []
        i = 0

        while True:
            try:
                array_path = f"arrays/{i}.bin"
                content = self.tar.extractfile(array_path).read()
                arr = np.frombuffer(content, dtype=np.float32)
                arrays.append(arr)
                i += 1
            except KeyError:
                break

        logger.info(f"Loaded {len(arrays)} arrays from model")
        return arrays

    def _load_vocabs(self) -> dict[str, Vocab]:
        """Load all vocabularies from vocabs/ directory."""
        vocabs = {}

        for vocab_name in ["word", "tag", "shape"]:
            try:
                content = self.tar.extractfile(f"vocabs/{vocab_name}.gz").read()
                lines = gzip.decompress(content).decode("utf-8").strip().split("\n")
                vocabs[vocab_name] = Vocab(lines)
                logger.info(f"Loaded {vocab_name} vocab: {len(lines)} items")
            except KeyError:
                logger.debug(f"Vocab {vocab_name}.gz not found")

        return vocabs

    def _resolve_weight(self, weight_spec: dict[str, Any]) -> np.ndarray:
        """Resolve weight from specification.

        Args:
            weight_spec: Weight specification from model.json

        Returns:
            Numpy array with proper shape
        """
        shape = weight_spec["shape"]
        dtype = weight_spec.get("dtype", "float32")
        array_id = weight_spec.get("array")

        if array_id is None:
            # Empty weight
            return None

        # Get flat array and reshape
        flat_array = self.arrays[array_id]
        expected_size = np.prod(shape)

        if len(flat_array) != expected_size:
            raise ValueError(
                f"Array size mismatch: expected {expected_size}, got {len(flat_array)}"
            )

        return flat_array.reshape(shape).astype(dtype)

    def load_navec_embedding(self) -> NavecEmbedding | None:
        """Load navec embedding from model."""
        if "emb" not in self.model_spec:
            return None

        emb_spec = self.model_spec["emb"]
        if "word" not in emb_spec:
            return None

        word_spec = emb_spec["word"]

        # Load quantized indexes
        indexes_spec = word_spec["indexes"]
        indexes_flat = np.frombuffer(
            (
                self.tar.extractfile("arrays/indexes.bin").read()
                if "indexes.bin" in [m.name for m in self.tar.getmembers()]
                else b""
            ),
            dtype=indexes_spec["dtype"],
        )

        if len(indexes_flat) == 0:
            logger.warning("Navec indexes not found in model")
            return None

        indexes = indexes_flat.reshape(indexes_spec["shape"])

        # Load codebook
        codes_spec = word_spec["codes"]
        codes_flat = np.frombuffer(
            (
                self.tar.extractfile("arrays/codes.bin").read()
                if "codes.bin" in [m.name for m in self.tar.getmembers()]
                else b""
            ),
            dtype=codes_spec["dtype"],
        )

        if len(codes_flat) == 0:
            logger.warning("Navec codes not found in model")
            return None

        codes = codes_flat.reshape(codes_spec["shape"])

        return NavecEmbedding(
            indexes=indexes,
            codes=codes,
            words=self.vocabs["word"],
        )

    def close(self) -> None:
        """Close tar file."""
        self.tar.close()

    def __enter__(self) -> ModelLoader:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()


__all__ = ["ModelLoader", "Vocab", "NavecEmbedding"]
