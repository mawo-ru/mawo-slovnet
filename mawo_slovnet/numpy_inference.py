"""Numpy-based inference engine for SlovNet models.

Implements neural network layers using pure numpy.
No dependencies on PyTorch, TensorFlow, or original slovnet.

Based on slovnet architecture:
- WordShapeEmbedding: concatenates word + shape embeddings
- CNNEncoder: stacked Conv1d -> ReLU -> BatchNorm layers
- MorphHead: linear projection to output tags
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class Conv1d:
    """1D Convolution layer (numpy implementation)."""

    def __init__(
        self,
        weight: np.ndarray,
        bias: np.ndarray | None = None,
        padding: int = 0,
    ) -> None:
        """Initialize Conv1d layer.

        Args:
            weight: Convolution weights (out_channels, in_channels, kernel_size)
            bias: Bias term (out_channels,)
            padding: Padding size
        """
        self.weight = weight
        self.bias = bias
        self.padding = padding

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply convolution.

        Args:
            x: Input (batch, in_channels, seq_len)

        Returns:
            Output (batch, out_channels, seq_len)
        """
        batch_size, in_channels, seq_len = x.shape
        out_channels, _, kernel_size = self.weight.shape

        # Apply padding
        if self.padding > 0:
            x = np.pad(
                x,
                ((0, 0), (0, 0), (self.padding, self.padding)),
                mode="constant",
            )

        # Calculate output sequence length
        out_seq_len = x.shape[2] - kernel_size + 1

        # Initialize output
        output = np.zeros((batch_size, out_channels, out_seq_len), dtype=np.float32)

        # Perform convolution
        for i in range(out_seq_len):
            # Extract window: (batch, in_channels, kernel_size)
            window = x[:, :, i : i + kernel_size]

            # Conv: (batch, in_channels, kernel) @ (out, in, kernel) -> (batch, out)
            # Reshape for broadcasting
            for out_ch in range(out_channels):
                conv_result = np.sum(
                    window * self.weight[out_ch : out_ch + 1, :, :],
                    axis=(1, 2),
                )
                output[:, out_ch, i] = conv_result

        # Add bias
        if self.bias is not None:
            output += self.bias[None, :, None]

        return output


class BatchNorm1d:
    """Batch normalization layer (inference mode)."""

    def __init__(
        self,
        weight: np.ndarray,
        bias: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        eps: float = 1e-5,
    ) -> None:
        """Initialize BatchNorm1d.

        Args:
            weight: Scale parameter (num_features,)
            bias: Shift parameter (num_features,)
            mean: Running mean (num_features,)
            std: Running std (num_features,)
            eps: Epsilon for numerical stability
        """
        self.weight = weight
        self.bias = bias
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply batch normalization.

        Args:
            x: Input (batch, num_features, seq_len)

        Returns:
            Normalized output (batch, num_features, seq_len)
        """
        # Normalize: (x - mean) / sqrt(std^2 + eps)
        normalized = (x - self.mean[None, :, None]) / np.sqrt(
            self.std[None, :, None] ** 2 + self.eps
        )

        # Scale and shift
        return self.weight[None, :, None] * normalized + self.bias[None, :, None]


class ReLU:
    """ReLU activation."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply ReLU.

        Args:
            x: Input

        Returns:
            max(0, x)
        """
        return np.maximum(0, x)


class CNNEncoderLayer:
    """Single CNN encoder layer: Conv1d -> ReLU -> BatchNorm1d."""

    def __init__(
        self,
        conv: Conv1d,
        norm: BatchNorm1d,
    ) -> None:
        """Initialize encoder layer.

        Args:
            conv: Convolution layer
            norm: Batch normalization layer
        """
        self.conv = conv
        self.relu = ReLU()
        self.norm = norm

    def __call__(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Forward pass.

        Args:
            x: Input (batch, channels, seq_len)
            mask: Binary mask (batch, seq_len)

        Returns:
            Output (batch, channels, seq_len)
        """
        x = self.conv(x)
        x = self.relu(x)
        x = self.norm(x)

        # Apply mask (zero out padded positions)
        if mask is not None:
            x = x * mask[:, None, :]

        return x


class CNNEncoder:
    """Stacked CNN encoder layers."""

    def __init__(self, layers: list[CNNEncoderLayer]) -> None:
        """Initialize encoder.

        Args:
            layers: List of encoder layers
        """
        self.layers = layers

    def __call__(
        self,
        x: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Forward pass through all layers.

        Args:
            x: Input (batch, seq_len, features)
            mask: Binary mask (batch, seq_len)

        Returns:
            Encoded output (batch, seq_len, features)
        """
        # Swap axes: (batch, seq, features) -> (batch, features, seq)
        x = np.swapaxes(x, 1, 2)

        # Apply all layers
        for layer in self.layers:
            x = layer(x, mask)

        # Swap back: (batch, features, seq) -> (batch, seq, features)
        x = np.swapaxes(x, 1, 2)

        return x


class WordShapeEmbedding:
    """Word + shape embeddings."""

    def __init__(
        self,
        word_emb: Any,  # NavecEmbedding or similar
        shape_weight: np.ndarray,
        word_vocab: Any,
        shape_vocab: Any,
    ) -> None:
        """Initialize embedding.

        Args:
            word_emb: Word embedding lookup
            shape_weight: Shape embedding weights (num_shapes, dim)
            word_vocab: Word vocabulary
            shape_vocab: Shape vocabulary
        """
        self.word_emb = word_emb
        self.shape_weight = shape_weight
        self.word_vocab = word_vocab
        self.shape_vocab = shape_vocab

    def _get_shape_id(self, word: str) -> int:
        """Get shape ID for word.

        Shape encodes character patterns:
        - X: uppercase
        - x: lowercase
        - d: digit
        - -: dash
        """
        if not word:
            return 0  # <pad>

        shape_chars = []
        for char in word:
            if char.isupper():
                shape_chars.append("X")
            elif char.islower():
                shape_chars.append("x")
            elif char.isdigit():
                shape_chars.append("d")
            elif char == "-":
                shape_chars.append("-")
            else:
                shape_chars.append("x")

        shape = "".join(shape_chars)

        # Lookup in vocab
        if shape in self.shape_vocab:
            return self.shape_vocab[shape]

        return 0  # <pad>

    def __call__(self, words: list[str]) -> np.ndarray:
        """Get embeddings for words.

        Args:
            words: List of words

        Returns:
            Embeddings (batch, seq_len, word_dim + shape_dim)
        """
        # Get word embeddings
        word_embs = []
        for word in words:
            if self.word_emb is not None:
                # Try to get embedding from navec
                try:
                    # Check if word in vocab (original navec has .vocab.words)
                    if hasattr(self.word_emb, "vocab"):
                        if word in self.word_emb.vocab.words:
                            word_embs.append(self.word_emb[word])
                        else:
                            # Unknown word - use zero embedding
                            word_embs.append(np.zeros(300, dtype=np.float32))
                    else:
                        # Fallback: try direct lookup
                        word_embs.append(self.word_emb[word])
                except (KeyError, AttributeError):
                    # Unknown word - use zero embedding
                    word_embs.append(np.zeros(300, dtype=np.float32))
            else:
                # No embeddings - use zero
                word_embs.append(np.zeros(300, dtype=np.float32))

        word_embs = np.array(word_embs)  # (seq_len, word_dim)

        # Get shape embeddings
        shape_ids = [self._get_shape_id(word) for word in words]
        shape_embs = self.shape_weight[shape_ids]  # (seq_len, shape_dim)

        # Concatenate
        combined = np.concatenate([word_embs, shape_embs], axis=1)

        # Add batch dimension
        return combined[None, :, :]  # (1, seq_len, word_dim + shape_dim)


class MorphHead:
    """Morphology prediction head."""

    def __init__(self, weight: np.ndarray, bias: np.ndarray) -> None:
        """Initialize head.

        Args:
            weight: Linear projection weights (num_tags, features)
            bias: Bias term (num_tags,)
        """
        self.weight = weight
        self.bias = bias

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Predict tags.

        Args:
            x: Encoded features (batch, seq_len, features)

        Returns:
            Tag logits (batch, seq_len, num_tags)
        """
        # Linear projection: (batch, seq, features) @ (features, tags)
        # weight shape: (in_features, out_features) = (features, tags)
        batch_size, seq_len, features = x.shape
        x_flat = x.reshape(-1, features)  # (batch * seq, features)

        logits_flat = x_flat @ self.weight + self.bias  # (batch * seq, tags)
        logits = logits_flat.reshape(batch_size, seq_len, -1)

        return logits

    def decode(self, logits: np.ndarray) -> np.ndarray:
        """Decode logits to tag IDs.

        Args:
            logits: Tag logits (batch, seq_len, num_tags)

        Returns:
            Tag IDs (batch, seq_len)
        """
        return np.argmax(logits, axis=-1)


class MorphTagger:
    """Complete morphology tagger."""

    def __init__(
        self,
        emb: WordShapeEmbedding,
        encoder: CNNEncoder,
        head: MorphHead,
        tag_vocab: Any,
    ) -> None:
        """Initialize tagger.

        Args:
            emb: Word + shape embedding
            encoder: CNN encoder
            head: Morphology head
            tag_vocab: Tag vocabulary
        """
        self.emb = emb
        self.encoder = encoder
        self.head = head
        self.tag_vocab = tag_vocab

    def __call__(self, words: list[str]) -> list[str]:
        """Tag words with morphological tags.

        Args:
            words: List of words

        Returns:
            List of tags
        """
        if not words:
            return []

        # Embed words
        x = self.emb(words)  # (1, seq_len, emb_dim)

        # Encode
        encoded = self.encoder(x)  # (1, seq_len, features)

        # Predict tags
        logits = self.head(encoded)  # (1, seq_len, num_tags)
        tag_ids = self.head.decode(logits)[0]  # (seq_len,)

        # Convert IDs to tags
        tags = []
        for tag_id in tag_ids:
            # Ensure tag_id is Python int, not numpy int64
            tag_id_int = int(tag_id)
            tag = self.tag_vocab[tag_id_int]
            tags.append(tag)

        return tags


__all__ = [
    "Conv1d",
    "BatchNorm1d",
    "ReLU",
    "CNNEncoderLayer",
    "CNNEncoder",
    "WordShapeEmbedding",
    "MorphHead",
    "MorphTagger",
]
