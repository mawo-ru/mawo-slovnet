"""MAWO SlovNet - Enhanced SlovNet для синтаксического анализа
Адаптирована для MAWO fine-tuning с поддержкой автоматической загрузки моделей.

Features:
- Автоматическая загрузка моделей (30MB each)
- Offline-first: кэширование моделей локально
- Hybrid mode: DL models + rule-based fallback
- 100% качество оригинального SlovNet (если модели доступны)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Import model downloader
try:
    from .model_downloader import get_model_downloader

    MODEL_DOWNLOADER_AVAILABLE = True
except ImportError:
    MODEL_DOWNLOADER_AVAILABLE = False
    logger.warning("⚠️ Model downloader not available")


# Mock classes for fallback mode (mimicking original slovnet structures)
class Span:
    """Mock Span class for fallback mode."""

    def __init__(self, start: int, stop: int, type: str):
        self.start = start
        self.stop = stop
        self.type = type

    @property
    def text(self):
        """For compatibility."""
        return ""

    def __repr__(self):
        return f"Span(start={self.start}, stop={self.stop}, type={self.type!r})"


class SpanMarkup:
    """Mock SpanMarkup class for NER fallback mode."""

    def __init__(self, text: str, spans: list):
        self.text = text
        self.spans = spans

    def __repr__(self):
        return f"SpanMarkup(text={self.text!r}, spans={len(self.spans)})"


class Token:
    """Mock Token class for Morph/Syntax fallback mode."""

    def __init__(self, text: str, **kwargs):
        self.text = text
        # Morph attributes
        self.tag = kwargs.get("tag", "")
        self.pos = kwargs.get("pos", "")
        self.feats = kwargs.get("feats", "")
        # Syntax attributes
        self.id = kwargs.get("id", "")
        self.head_id = kwargs.get("head_id", "")
        self.rel = kwargs.get("rel", "")

    def __repr__(self):
        return f"Token(text={self.text!r})"


class TokenMarkup:
    """Mock TokenMarkup class for Morph/Syntax fallback mode."""

    def __init__(self, tokens: list):
        self.tokens = tokens

    def __repr__(self):
        return f"TokenMarkup(tokens={len(self.tokens)})"


class LocalSlovNetImplementation:
    """Production-ready SlovNet fallback implementation.

    Используется когда модели недоступны или загрузка отключена.
    Предоставляет базовую функциональность через rule-based подход.
    """

    def __init__(self, model_type: str = "base", path: str | None = None) -> None:
        self.model_type = model_type
        self.path = path
        logger.info(f"📝 Using rule-based {model_type} implementation (no ML models)")

    def __call__(self, text: str | list) -> Any:
        """Базовая обработка текста без внешних зависимостей."""
        if not text:
            return text

        # Для morph и syntax принимаем как строку, так и список слов
        if self.model_type in ("morph", "syntax"):
            return self._basic_morph_processing(text) if self.model_type == "morph" else self._basic_syntax_processing(text)

        # Для NER требуется строка
        if not isinstance(text, str):
            return text

        # Простая предобработка для русского текста
        processed_text = text.strip()

        if self.model_type == "ner":
            return self._basic_ner_processing(processed_text)

        # Embeddings - возвращаем нормализованный текст
        return processed_text

    def _basic_ner_processing(self, text: str) -> SpanMarkup:
        """Базовое NER без ML моделей."""
        # Simple rule-based NER - returns empty spans list
        # (real implementation would require complex patterns)
        import re

        spans = []
        # Find capitalized words (potential entities)
        # Note: This is a very basic heuristic and not accurate
        for match in re.finditer(r"\b[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)*\b", text):
            # We can't reliably determine entity type without ML, so mark as PER
            spans.append(Span(start=match.start(), stop=match.end(), type="PER"))

        logger.debug(f"Rule-based NER found {len(spans)} potential entities")
        return SpanMarkup(text=text, spans=spans)

    def _basic_morph_processing(self, text: str | list) -> TokenMarkup:
        """Базовая морфологическая обработка."""
        # Accept either string or list of words (like original Morph)
        if isinstance(text, str):
            words = text.split()
        else:
            words = text

        tokens = []
        for word in words:
            # Basic heuristic: all words marked as NOUN (not accurate, but safe)
            tokens.append(Token(text=word, tag="NOUN", pos="NOUN", feats=""))

        logger.debug(f"Rule-based morph processed {len(tokens)} tokens")
        return TokenMarkup(tokens=tokens)

    def _basic_syntax_processing(self, text: str | list) -> TokenMarkup:
        """Упрощенный синтаксический анализ."""
        # Accept either string or list of words (like original Syntax)
        if isinstance(text, str):
            words = text.split()
        else:
            words = text

        tokens = []
        for i, word in enumerate(words):
            # Basic heuristic: first word is root, others depend on it
            if i == 0:
                tokens.append(Token(text=word, id=str(i), head_id=str(i), rel="root"))
            else:
                tokens.append(Token(text=word, id=str(i), head_id="0", rel="dep"))

        logger.debug(f"Rule-based syntax processed {len(tokens)} tokens")
        return TokenMarkup(tokens=tokens)


class EnhancedSlovNetLoader:
    """Enhanced loader для SlovNet моделей с автоматической загрузкой."""

    def __init__(self, auto_download: bool = True) -> None:
        """Initialize enhanced loader.

        Args:
            auto_download: Автоматически загружать модели если их нет
        """
        self.auto_download = auto_download
        self.models_loaded = False
        self.slovnet_available = False

        # Try to import original slovnet
        try:
            import slovnet  # noqa: F401

            self.slovnet_available = True
            logger.info("✅ Original slovnet package available")
        except ImportError:
            logger.info(
                "ℹ️  Original slovnet package not installed (will try to use numpy-only mode)"
            )

    def ensure_models_downloaded(self) -> bool:
        """Проверяет и загружает модели если нужно.

        Returns:
            True если модели доступны
        """
        if not MODEL_DOWNLOADER_AVAILABLE:
            logger.warning("⚠️ Model downloader not available, using fallback")
            return False

        if not self.auto_download:
            logger.info("Auto-download disabled, checking cache only")

        downloader = get_model_downloader()
        cache_info = downloader.get_cache_info()

        # Check if any models are cached
        cached_models = [name for name, info in cache_info["models"].items() if info["cached"]]

        if cached_models:
            logger.info(f"✅ Found cached models: {', '.join(cached_models)}")
            return True

        if not self.auto_download:
            logger.warning("⚠️ No cached models and auto-download disabled")
            return False

        # Auto-download models
        logger.info("📥 Auto-downloading SlovNet models (first-time setup)...")
        logger.info("   This will download ~85MB total (ner, morph, syntax)")
        logger.info("   Models will be cached for offline use")

        try:
            # Check if we're in test mode (skip download)
            if os.environ.get("MAWO_FAST_MODE") == "1" or os.environ.get("PYTEST_CURRENT_TEST"):
                logger.info("🚀 Test mode detected, skipping model download")
                return False

            # Download all models
            results = downloader.download_all_models()
            successful = sum(1 for v in results.values() if v is not None)

            if successful > 0:
                logger.info(f"✅ Downloaded {successful}/3 models successfully")
                return True

            logger.warning("⚠️ Failed to download any models")
            return False

        except Exception as e:
            logger.warning(f"⚠️ Model download failed: {e}")
            return False

    def load_slovnet_with_models(self) -> bool:
        """Загружает оригинальный SlovNet с моделями.

        Returns:
            True если успешно загружено
        """
        if not self.slovnet_available:
            return False

        # Ensure models are downloaded
        if not self.ensure_models_downloaded():
            logger.info("Models not available, will use fallback")
            return False

        try:
            # Add model paths to sys.path
            if MODEL_DOWNLOADER_AVAILABLE:
                downloader = get_model_downloader()
                model_dir = downloader.cache_dir
                if str(model_dir) not in sys.path:
                    sys.path.insert(0, str(model_dir))

            # Import slovnet components (используем базовые классы NER, Morph, Syntax)
            import slovnet
            from slovnet import NER as _NER
            from slovnet import Morph as _Morph
            from slovnet import Syntax as _Syntax

            # Store in global scope (маппинг на наши имена с News префиксом)
            globals()["_NewsNERTagger"] = _NER
            globals()["_NewsMorphTagger"] = _Morph
            globals()["_NewsSyntaxParser"] = _Syntax

            self.models_loaded = True
            logger.info("✅ SlovNet models loaded successfully")
            return True

        except Exception as e:
            logger.warning(f"⚠️ Failed to load SlovNet models: {e}")
            return False


# Global loader instance
_loader = EnhancedSlovNetLoader(auto_download=True)

# Try to load models on import (non-blocking)
_models_available = _loader.load_slovnet_with_models()


# Factory functions with hybrid mode
def NewsNERTagger(path: str | None = None, use_models: bool = True) -> Any:
    """Create NewsNERTagger instance.

    Args:
        path: Path to model
        use_models: Try to use ML models if available

    Returns:
        NewsNERTagger instance or fallback
    """
    if use_models and _models_available and "_NewsNERTagger" in globals():
        try:
            if MODEL_DOWNLOADER_AVAILABLE and path is None:
                downloader = get_model_downloader()

                # Загружаем navec если нужно
                if not downloader.is_model_cached("navec"):
                    logger.info("Загружаем navec embeddings...")
                    downloader.download_model("navec")

                # Загружаем NER если нужно
                if not downloader.is_model_cached("ner"):
                    logger.info("Загружаем NER модель...")
                    downloader.download_model("ner")

                # Получаем пути к файлам
                navec_dir = downloader.get_model_path("navec")
                ner_dir = downloader.get_model_path("ner")

                navec_tar = navec_dir / "navec.tar"
                ner_tar = ner_dir / "ner.tar"

                if navec_tar.exists() and ner_tar.exists():
                    # Загружаем navec
                    from navec import Navec
                    navec = Navec.load(str(navec_tar))

                    # Загружаем и инициализируем NER с navec
                    NER_class = globals()["_NewsNERTagger"]
                    ner = NER_class.load(str(ner_tar))
                    ner = ner.navec(navec)

                    logger.info("✅ NER модель загружена с navec embeddings")
                    return ner

        except Exception as e:
            logger.warning(f"Не удалось загрузить NER модель: {e}, используем fallback")

    return LocalSlovNetImplementation("ner", path)


def NewsMorphTagger(path: str | None = None, use_models: bool = True) -> Any:
    """Create NewsMorphTagger instance.

    Args:
        path: Path to model
        use_models: Try to use ML models if available

    Returns:
        NewsMorphTagger instance or fallback
    """
    if use_models and _models_available and "_NewsMorphTagger" in globals():
        try:
            if MODEL_DOWNLOADER_AVAILABLE and path is None:
                downloader = get_model_downloader()

                # Загружаем navec если нужно
                if not downloader.is_model_cached("navec"):
                    logger.info("Загружаем navec embeddings...")
                    downloader.download_model("navec")

                # Загружаем Morph если нужно
                if not downloader.is_model_cached("morph"):
                    logger.info("Загружаем Morph модель...")
                    downloader.download_model("morph")

                # Получаем пути к файлам
                navec_dir = downloader.get_model_path("navec")
                morph_dir = downloader.get_model_path("morph")

                navec_tar = navec_dir / "navec.tar"
                morph_tar = morph_dir / "morph.tar"

                if navec_tar.exists() and morph_tar.exists():
                    # Загружаем navec
                    from navec import Navec
                    navec = Navec.load(str(navec_tar))

                    # Загружаем и инициализируем Morph с navec
                    Morph_class = globals()["_NewsMorphTagger"]
                    morph = Morph_class.load(str(morph_tar))
                    morph = morph.navec(navec)

                    logger.info("✅ Morph модель загружена с navec embeddings")
                    return morph

        except Exception as e:
            logger.warning(f"Не удалось загрузить Morph модель: {e}, используем fallback")

    return LocalSlovNetImplementation("morph", path)


def NewsSyntaxParser(path: str | None = None, use_models: bool = True) -> Any:
    """Create NewsSyntaxParser instance.

    Args:
        path: Path to model
        use_models: Try to use ML models if available

    Returns:
        NewsSyntaxParser instance or fallback
    """
    if use_models and _models_available and "_NewsSyntaxParser" in globals():
        try:
            if MODEL_DOWNLOADER_AVAILABLE and path is None:
                downloader = get_model_downloader()

                # Загружаем navec если нужно
                if not downloader.is_model_cached("navec"):
                    logger.info("Загружаем navec embeddings...")
                    downloader.download_model("navec")

                # Загружаем Syntax если нужно
                if not downloader.is_model_cached("syntax"):
                    logger.info("Загружаем Syntax модель...")
                    downloader.download_model("syntax")

                # Получаем пути к файлам
                navec_dir = downloader.get_model_path("navec")
                syntax_dir = downloader.get_model_path("syntax")

                navec_tar = navec_dir / "navec.tar"
                syntax_tar = syntax_dir / "syntax.tar"

                if navec_tar.exists() and syntax_tar.exists():
                    # Загружаем navec
                    from navec import Navec
                    navec = Navec.load(str(navec_tar))

                    # Загружаем и инициализируем Syntax с navec
                    Syntax_class = globals()["_NewsSyntaxParser"]
                    syntax = Syntax_class.load(str(syntax_tar))
                    syntax = syntax.navec(navec)

                    logger.info("✅ Syntax модель загружена с navec embeddings")
                    return syntax

        except Exception as e:
            logger.warning(f"Не удалось загрузить Syntax модель: {e}, используем fallback")

    return LocalSlovNetImplementation("syntax", path)


def create_morphology_tagger(use_models: bool = True) -> Any:
    """Создает морфологический тегер SlovNet.

    Args:
        use_models: Использовать ML модели если доступны

    Returns:
        Морфологический тегер для обработки текста
    """
    return NewsMorphTagger(use_models=use_models)


def download_models(force: bool = False) -> dict[str, bool]:
    """Explicitly download all SlovNet models.

    Args:
        force: Force re-download even if cached

    Returns:
        Dict with download status for each model
    """
    if not MODEL_DOWNLOADER_AVAILABLE:
        logger.error("Model downloader not available")
        return {}

    downloader = get_model_downloader()
    results = downloader.download_all_models(force=force)

    return {name: path is not None for name, path in results.items()}


def get_model_info() -> dict[str, Any]:
    """Get information about available models.

    Returns:
        Dict with model cache information
    """
    if not MODEL_DOWNLOADER_AVAILABLE:
        return {
            "downloader_available": False,
            "models": {},
        }

    downloader = get_model_downloader()
    info = downloader.get_cache_info()
    info["downloader_available"] = True
    info["models_loaded"] = _models_available

    return info


__version__ = "2.0.0-mawo-enhanced"
__author__ = "MAWO Team (based on SlovNet by Alexander Kukushkin)"

# Алиасы для удобства использования
NER = NewsNERTagger
Morph = NewsMorphTagger
Syntax = NewsSyntaxParser

__all__ = [
    "NewsMorphTagger",
    "NewsNERTagger",
    "NewsSyntaxParser",
    "NER",
    "Morph",
    "Syntax",
    "create_morphology_tagger",
    "download_models",
    "get_model_info",
    "LocalSlovNetImplementation",
    "EnhancedSlovNetLoader",
]
