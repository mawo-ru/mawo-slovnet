"""High-level API for numpy-based SlovNet models.

Provides simple load() methods for Morph, NER, and Syntax models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .numpy_inference import (
    BatchNorm1d,
    CNNEncoder,
    CNNEncoderLayer,
    Conv1d,
    MorphHead,
    MorphTagger,
    WordShapeEmbedding,
)
from .numpy_loader import ModelLoader

logger = logging.getLogger(__name__)


class Morph:
    """Morphology tagger with numpy-only implementation."""

    def __init__(self, tagger: MorphTagger, loader: ModelLoader) -> None:
        """Initialize Morph.

        Args:
            tagger: MorphTagger instance
            loader: ModelLoader instance
        """
        self.tagger = tagger
        self.loader = loader

    @staticmethod
    def _load_navec_from_tar(navec_path: Path) -> Any:
        """Load navec embeddings using original navec library.

        Args:
            navec_path: Path to navec .tar file

        Returns:
            Navec instance or None if failed
        """
        try:
            # Use original navec library (like in mawo-natasha)
            from navec import Navec

            navec = Navec.load(str(navec_path))
            vocab_size = len(navec.vocab.words)
            logger.info(f"✅ Navec loaded: {vocab_size:,} words")
            return navec

        except ImportError:
            logger.warning("⚠️ navec library not installed (pip install navec)")
            logger.info(
                "   Word embeddings will use zero vectors (install navec for better quality)"
            )
            return None
        except Exception as e:
            logger.warning(f"Failed to load navec: {e}")
            return None

    @classmethod
    def load(cls, path: str | Path, navec_path: str | Path | None = None) -> Morph:
        """Load morphology tagger from .tar file.

        Args:
            path: Path to slovnet_morph_*.tar file
            navec_path: Optional path to navec embeddings .tar file
                       If None, looks for navec.tar in same directory

        Returns:
            Morph instance

        Example:
            >>> morph = Morph.load('slovnet_morph_news_v1.tar')
            >>> tags = morph(['Мама', 'мыла', 'раму'])
        """
        loader = ModelLoader(path)

        # Try to load navec
        navec_emb = None
        if navec_path is None:
            # Look for navec.tar in ../navec/ directory
            model_dir = Path(path).parent.parent
            potential_navec = model_dir / "navec" / "navec.tar"
            if potential_navec.exists():
                navec_path = potential_navec
                logger.info(f"Found navec at: {navec_path}")

        if navec_path and Path(navec_path).exists():
            # Load using original navec library
            navec_emb = cls._load_navec_from_tar(Path(navec_path))

        # Load vocabularies
        word_vocab = loader.vocabs.get("word")
        shape_vocab = loader.vocabs.get("shape")
        tag_vocab = loader.vocabs.get("tag")

        if not all([word_vocab, shape_vocab, tag_vocab]):
            raise ValueError("Missing required vocabularies")

        logger.info(
            f"Loaded vocabs: {len(word_vocab)} words, "
            f"{len(shape_vocab)} shapes, {len(tag_vocab)} tags"
        )

        # Load shape embedding
        shape_weight = loader._resolve_weight(loader.model_spec["emb"]["shape"]["weight"])

        # Create embedding layer
        emb = WordShapeEmbedding(
            word_emb=navec_emb,
            shape_weight=shape_weight,
            word_vocab=word_vocab,
            shape_vocab=shape_vocab,
        )

        # Load encoder layers
        encoder_layers = []
        for layer_spec in loader.model_spec["encoder"]["layers"]:
            # Conv layer
            conv_weight = loader._resolve_weight(layer_spec["conv"]["weight"])
            conv_bias = loader._resolve_weight(layer_spec["conv"]["bias"])
            padding = layer_spec["conv"].get("padding", 0)

            conv = Conv1d(
                weight=conv_weight,
                bias=conv_bias,
                padding=padding,
            )

            # BatchNorm layer
            norm_weight = loader._resolve_weight(layer_spec["norm"]["weight"])
            norm_bias = loader._resolve_weight(layer_spec["norm"]["bias"])
            norm_mean = loader._resolve_weight(layer_spec["norm"]["mean"])
            norm_std = loader._resolve_weight(layer_spec["norm"]["std"])

            norm = BatchNorm1d(
                weight=norm_weight,
                bias=norm_bias,
                mean=norm_mean,
                std=norm_std,
            )

            encoder_layers.append(CNNEncoderLayer(conv, norm))

        encoder = CNNEncoder(encoder_layers)

        logger.info(f"Loaded encoder with {len(encoder_layers)} layers")

        # Load head (proj layer)
        head_weight = loader._resolve_weight(loader.model_spec["head"]["proj"]["weight"])
        head_bias = loader._resolve_weight(loader.model_spec["head"]["proj"]["bias"])

        head = MorphHead(weight=head_weight, bias=head_bias)

        # Create tagger
        tagger = MorphTagger(
            emb=emb,
            encoder=encoder,
            head=head,
            tag_vocab=tag_vocab,
        )

        logger.info("✅ Morph model loaded successfully (numpy-only)")

        return cls(tagger, loader)

    def __call__(self, words: list[str]) -> list[dict[str, str]]:
        """Tag words with morphological information.

        Args:
            words: List of words to analyze

        Returns:
            List of token dictionaries with pos, feats, etc.

        Example:
            >>> morph(['Мама', 'мыла', 'раму'])
            [
                {'text': 'Мама', 'pos': 'NOUN', 'feats': 'Case=Nom|Gender=Fem|Number=Sing'},
                {'text': 'мыла', 'pos': 'VERB', 'feats': 'Gender=Fem|Number=Sing|Tense=Past'},
                {'text': 'раму', 'pos': 'NOUN', 'feats': 'Case=Acc|Gender=Fem|Number=Sing'},
            ]
        """
        if not words:
            return []

        # Get tags
        tags = self.tagger(words)

        # Parse tags into structured format
        results = []
        for word, tag in zip(words, tags, strict=True):
            # Parse tag: "NOUN|Case=Nom|Gender=Fem" -> pos, feats
            parts = tag.split("|")
            pos = parts[0] if parts else "X"
            feats = "|".join(parts[1:]) if len(parts) > 1 else ""

            results.append(
                {
                    "text": word,
                    "pos": pos,
                    "tag": tag,
                    "feats": feats,
                }
            )

        return results

    def close(self) -> None:
        """Close model loader."""
        self.loader.close()

    def __enter__(self) -> Morph:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()


__all__ = ["Morph"]
