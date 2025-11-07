"""MAWO SlovNet Model Downloader & Cache Manager
Автоматическая загрузка и кэширование моделей SlovNet для offline работы.

Based on:
- SlovNet v0.6.0 (github.com/natasha/slovnet)
- MAWO offline-first architecture
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tarfile
from pathlib import Path
from typing import Any, Optional
from urllib.request import urlopen

logger = logging.getLogger(__name__)


class SlovNetModelDownloader:
    """Загрузка и кэширование моделей SlovNet."""

    # Official SlovNet models from Yandex Cloud Storage
    MODELS = {
        "ner": {
            "url": "https://storage.yandexcloud.net/natasha-slovnet/08_slovnet_ner_news_v1.tar",
            "size_mb": 30,
            "sha256": "b5f7f8f0c8c6c5c4c3c2c1c0",  # Placeholder
            "description": "Named Entity Recognition для русских новостей",
        },
        "morph": {
            "url": "https://storage.yandexcloud.net/natasha-slovnet/08_slovnet_morph_news_v1.tar",
            "size_mb": 27,
            "sha256": "c6f8f9f1c9c7c6c5c4c3c2c1",  # Placeholder
            "description": "Морфологический анализ для русских новостей",
        },
        "syntax": {
            "url": "https://storage.yandexcloud.net/natasha-slovnet/08_slovnet_syntax_news_v1.tar",
            "size_mb": 28,
            "sha256": "d7f9faf2cac8c7c6c5c4c3c2",  # Placeholder
            "description": "Синтаксический парсинг для русских новостей",
        },
    }

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        """Initialize model downloader.

        Args:
            cache_dir: Directory for model cache. If None, uses default.
        """
        if cache_dir is None:
            # Default: local_libs/mawo_slovnet/models
            cache_dir = Path(__file__).parent / "models"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"SlovNet model cache: {self.cache_dir}")

    def is_model_cached(self, model_name: str) -> bool:
        """Проверяет, закэширована ли модель.

        Args:
            model_name: Название модели (ner, morph, syntax)

        Returns:
            True если модель уже загружена
        """
        if model_name not in self.MODELS:
            return False

        model_dir = self.cache_dir / model_name
        return model_dir.exists() and (model_dir / ".download_complete").exists()

    def get_model_path(self, model_name: str) -> Path:
        """Получает путь к кэшированной модели.

        Args:
            model_name: Название модели

        Returns:
            Path к директории модели
        """
        return self.cache_dir / model_name

    def download_model(
        self, model_name: str, force: bool = False, progress_callback: Any = None
    ) -> Path:
        """Загружает модель если её нет в кэше.

        Args:
            model_name: Название модели (ner, morph, syntax)
            force: Принудительная перезагрузка
            progress_callback: Callback для прогресса загрузки

        Returns:
            Path к директории модели

        Raises:
            ValueError: Если модель неизвестна
            RuntimeError: Если загрузка не удалась
        """
        if model_name not in self.MODELS:
            available = ", ".join(self.MODELS.keys())
            msg = f"Unknown model: {model_name}. Available: {available}"
            raise ValueError(msg)

        # Проверяем кэш
        if not force and self.is_model_cached(model_name):
            logger.info(f"⚡ Model '{model_name}' already cached")
            return self.get_model_path(model_name)

        model_info = self.MODELS[model_name]
        model_dir = self.get_model_path(model_name)

        logger.info(f"📥 Downloading '{model_name}' model ({model_info['size_mb']}MB)...")
        logger.info(f"   {model_info['description']}")

        try:
            # Скачиваем во временную директорию
            temp_dir = self.cache_dir / f"{model_name}.tmp"
            temp_dir.mkdir(parents=True, exist_ok=True)

            tar_path = temp_dir / f"{model_name}.tar"

            # Download with progress
            self._download_file(
                model_info["url"], tar_path, model_info["size_mb"], progress_callback
            )

            # Extract tar archive
            logger.info(f"📦 Extracting model archive...")
            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(temp_dir)

            # Move to final location
            if model_dir.exists():
                shutil.rmtree(model_dir)

            # Find extracted directory (usually the same name as model)
            extracted_dirs = [
                d for d in temp_dir.iterdir() if d.is_dir() and d.name != "__pycache__"
            ]
            if extracted_dirs:
                shutil.move(str(extracted_dirs[0]), str(model_dir))
            else:
                # If extracted files are in temp_dir directly
                model_dir.mkdir(parents=True, exist_ok=True)
                for item in temp_dir.iterdir():
                    if item.name != f"{model_name}.tar":
                        shutil.move(str(item), str(model_dir / item.name))

            # Mark as complete
            (model_dir / ".download_complete").touch()

            # Cleanup
            shutil.rmtree(temp_dir)

            logger.info(f"✅ Model '{model_name}' downloaded successfully")
            return model_dir

        except Exception as e:
            logger.exception(f"❌ Failed to download model '{model_name}': {e}")
            # Cleanup on failure
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            msg = f"Model download failed: {e}"
            raise RuntimeError(msg) from e

    def _download_file(
        self, url: str, dest: Path, size_mb: float, progress_callback: Any = None
    ) -> None:
        """Загружает файл с прогрессом.

        Args:
            url: URL для загрузки
            dest: Путь назначения
            size_mb: Ожидаемый размер в MB
            progress_callback: Callback для прогресса
        """
        try:
            # Try with tqdm for nice progress bar
            from tqdm import tqdm

            with urlopen(url) as response:
                total_size = int(response.headers.get("content-length", size_mb * 1024 * 1024))

                with open(dest, "wb") as f:
                    with tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc=f"Downloading {dest.name}",
                        leave=False,
                    ) as pbar:
                        chunk_size = 8192
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                            pbar.update(len(chunk))
                            if progress_callback:
                                progress_callback(len(chunk), total_size)

        except ImportError:
            # Fallback without tqdm
            logger.info("   (tqdm not available, progress bar disabled)")
            with urlopen(url) as response:
                with open(dest, "wb") as f:
                    chunk_size = 8192
                    downloaded = 0
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        # Log progress every 5MB
                        if downloaded % (5 * 1024 * 1024) < chunk_size:
                            progress_mb = downloaded / (1024 * 1024)
                            logger.info(f"   Downloaded: {progress_mb:.1f} MB...")
                        if progress_callback:
                            progress_callback(len(chunk), size_mb * 1024 * 1024)

    def download_all_models(self, force: bool = False) -> dict[str, Path]:
        """Загружает все доступные модели.

        Args:
            force: Принудительная перезагрузка

        Returns:
            Dict с путями к моделям
        """
        results = {}
        total = len(self.MODELS)

        logger.info(f"📥 Downloading {total} SlovNet models...")

        for i, model_name in enumerate(self.MODELS, 1):
            logger.info(f"[{i}/{total}] {model_name.upper()}")
            try:
                model_path = self.download_model(model_name, force=force)
                results[model_name] = model_path
            except Exception as e:
                logger.error(f"Failed to download {model_name}: {e}")
                results[model_name] = None

        successful = sum(1 for v in results.values() if v is not None)
        logger.info(f"✅ Downloaded {successful}/{total} models successfully")

        return results

    def clear_cache(self, model_name: str | None = None) -> None:
        """Очищает кэш моделей.

        Args:
            model_name: Название модели или None для очистки всех
        """
        if model_name:
            model_dir = self.get_model_path(model_name)
            if model_dir.exists():
                shutil.rmtree(model_dir)
                logger.info(f"🗑️  Cleared cache for '{model_name}'")
        else:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("🗑️  Cleared all model cache")

    def get_cache_info(self) -> dict[str, Any]:
        """Получает информацию о кэше.

        Returns:
            Dict с информацией о кэшированных моделях
        """
        info = {
            "cache_dir": str(self.cache_dir),
            "total_size_mb": 0,
            "models": {},
        }

        for model_name in self.MODELS:
            model_dir = self.get_model_path(model_name)
            cached = self.is_model_cached(model_name)

            size_mb = 0
            if cached:
                # Calculate directory size
                size_bytes = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
                size_mb = size_bytes / (1024 * 1024)
                info["total_size_mb"] += size_mb

            info["models"][model_name] = {
                "cached": cached,
                "size_mb": round(size_mb, 1) if cached else 0,
                "path": str(model_dir) if cached else None,
            }

        info["total_size_mb"] = round(info["total_size_mb"], 1)
        return info


# Global instance for convenience
_global_downloader: SlovNetModelDownloader | None = None


def get_model_downloader(cache_dir: Path | str | None = None) -> SlovNetModelDownloader:
    """Get global model downloader instance.

    Args:
        cache_dir: Custom cache directory (optional)

    Returns:
        SlovNetModelDownloader instance
    """
    global _global_downloader

    if _global_downloader is None or cache_dir is not None:
        _global_downloader = SlovNetModelDownloader(cache_dir)

    return _global_downloader


__all__ = ["SlovNetModelDownloader", "get_model_downloader"]
