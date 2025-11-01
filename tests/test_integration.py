"""
Строгие интеграционные тесты для mawo-slovnet
Тестируют библиотеку как самодостаточный проект
"""

import pytest
from pathlib import Path


class TestImports:
    """Тесты импортов"""

    def test_main_module_import(self):
        """Тест: главный модуль импортируется"""
        try:
            import mawo_slovnet
        except ImportError as e:
            pytest.fail(f"Failed to import mawo_slovnet: {e}")

    def test_ner_class_import(self):
        """Тест: класс NER импортируется"""
        try:
            from mawo_slovnet import NER
        except ImportError as e:
            pytest.fail(f"Failed to import NER: {e}")

    def test_morph_class_import(self):
        """Тест: класс Morph импортируется"""
        try:
            from mawo_slovnet import Morph
        except ImportError as e:
            pytest.fail(f"Failed to import Morph: {e}")

    def test_syntax_class_import(self):
        """Тест: класс Syntax импортируется"""
        try:
            from mawo_slovnet import Syntax
        except ImportError as e:
            pytest.fail(f"Failed to import Syntax: {e}")


class TestNERInitialization:
    """Тесты инициализации NER"""

    def test_ner_initialization(self):
        """Тест: NER инициализируется"""
        from mawo_slovnet import NER

        try:
            ner = NER()
            assert ner is not None
        except Exception as e:
            pytest.fail(f"Failed to initialize NER: {e}")

    def test_ner_model_exists(self):
        """Тест: модель NER существует"""
        from mawo_slovnet import NER

        ner = NER()
        # Проверяем, что модель загружена
        assert hasattr(ner, "model") or hasattr(ner, "_model")


class TestMorphInitialization:
    """Тесты инициализации Morph"""

    def test_morph_initialization(self):
        """Тест: Morph инициализируется"""
        from mawo_slovnet import Morph

        try:
            morph = Morph()
            assert morph is not None
        except Exception as e:
            pytest.fail(f"Failed to initialize Morph: {e}")


class TestSyntaxInitialization:
    """Тесты инициализации Syntax"""

    def test_syntax_initialization(self):
        """Тест: Syntax инициализируется"""
        from mawo_slovnet import Syntax

        try:
            syntax = Syntax()
            assert syntax is not None
        except Exception as e:
            pytest.fail(f"Failed to initialize Syntax: {e}")


class TestNERFunctionality:
    """Тесты функциональности NER"""

    def test_ner_simple_text(self):
        """Тест: NER обрабатывает простой текст"""
        from mawo_slovnet import NER

        ner = NER()
        text = "Владимир Путин выступил в Москве"

        try:
            result = ner(text)
            assert result is not None
            assert isinstance(result, (list, tuple))
        except Exception as e:
            pytest.fail(f"NER failed on simple text: {e}")

    def test_ner_finds_person(self):
        """Тест: NER находит имя человека"""
        from mawo_slovnet import NER

        ner = NER()
        text = "Александр Пушкин родился в Москве"
        result = ner(text)

        # Проверяем, что нашли хотя бы одну сущность
        assert len(result) > 0, "Expected to find at least one entity"

        # Проверяем, что есть сущность типа PER
        has_person = any(span.type == "PER" for span in result if hasattr(span, "type"))
        assert has_person, "Expected to find PER entity for 'Александр Пушкин'"

    def test_ner_finds_location(self):
        """Тест: NER находит местоположение"""
        from mawo_slovnet import NER

        ner = NER()
        text = "Он живёт в Санкт-Петербурге"
        result = ner(text)

        assert len(result) > 0
        has_location = any(span.type == "LOC" for span in result if hasattr(span, "type"))
        assert has_location, "Expected to find LOC entity for 'Санкт-Петербурге'"

    def test_ner_empty_text(self):
        """Тест: NER обрабатывает пустой текст"""
        from mawo_slovnet import NER

        ner = NER()
        result = ner("")

        assert isinstance(result, (list, tuple))

    def test_ner_no_entities(self):
        """Тест: NER обрабатывает текст без сущностей"""
        from mawo_slovnet import NER

        ner = NER()
        text = "Я люблю программировать"
        result = ner(text)

        assert isinstance(result, (list, tuple))


class TestMorphFunctionality:
    """Тесты функциональности Morph"""

    def test_morph_simple_sentence(self):
        """Тест: Morph обрабатывает простое предложение"""
        from mawo_slovnet import Morph

        morph = Morph()
        text = "кот спит"

        try:
            result = morph(text)
            assert result is not None
            assert isinstance(result, (list, tuple))
        except Exception as e:
            pytest.fail(f"Morph failed on simple sentence: {e}")

    def test_morph_analyzes_noun(self):
        """Тест: Morph анализирует существительное"""
        from mawo_slovnet import Morph

        morph = Morph()
        text = "кот"
        result = morph(text)

        assert len(result) > 0
        # Проверяем, что есть разметка
        first_token = result[0]
        assert hasattr(first_token, "pos") or hasattr(first_token, "tag")

    def test_morph_empty_text(self):
        """Тест: Morph обрабатывает пустой текст"""
        from mawo_slovnet import Morph

        morph = Morph()
        result = morph("")

        assert isinstance(result, (list, tuple))


class TestSyntaxFunctionality:
    """Тесты функциональности Syntax"""

    def test_syntax_simple_sentence(self):
        """Тест: Syntax обрабатывает простое предложение"""
        from mawo_slovnet import Syntax

        syntax = Syntax()
        text = "кот спит"

        try:
            result = syntax(text)
            assert result is not None
            assert isinstance(result, (list, tuple))
        except Exception as e:
            pytest.fail(f"Syntax failed on simple sentence: {e}")

    def test_syntax_finds_dependencies(self):
        """Тест: Syntax находит зависимости"""
        from mawo_slovnet import Syntax

        syntax = Syntax()
        text = "Мама мыла раму"
        result = syntax(text)

        assert len(result) > 0
        # Проверяем, что есть синтаксические связи
        first_token = result[0]
        assert hasattr(first_token, "head") or hasattr(first_token, "rel")


class TestDataFiles:
    """Тесты наличия файлов данных"""

    def test_ner_model_file_exists(self):
        """Тест: файл модели NER существует"""
        from pathlib import Path
        import mawo_slovnet

        # Находим директорию модуля
        module_path = Path(mawo_slovnet.__file__).parent

        # Проверяем наличие модели NER
        ner_model = module_path / "slovnet_ner_news_v1.tar.neural.gz"
        assert ner_model.exists(), f"NER model not found at {ner_model}"
        assert ner_model.stat().st_size > 1_000_000, "NER model file is too small"

    def test_morph_model_file_exists(self):
        """Тест: файл модели Morph существует"""
        from pathlib import Path
        import mawo_slovnet

        module_path = Path(mawo_slovnet.__file__).parent
        morph_model = module_path / "slovnet_morph_news_v1.tar.neural.gz"

        assert morph_model.exists(), f"Morph model not found at {morph_model}"
        assert morph_model.stat().st_size > 1_000_000, "Morph model file is too small"

    def test_syntax_model_file_exists(self):
        """Тест: файл модели Syntax существует"""
        from pathlib import Path
        import mawo_slovnet

        module_path = Path(mawo_slovnet.__file__).parent
        syntax_model = module_path / "slovnet_syntax_news_v1.tar.neural.gz"

        assert syntax_model.exists(), f"Syntax model not found at {syntax_model}"
        assert syntax_model.stat().st_size > 1_000_000, "Syntax model file is too small"


class TestEdgeCases:
    """Тесты граничных случаев"""

    def test_ner_long_text(self):
        """Тест: NER обрабатывает длинный текст"""
        from mawo_slovnet import NER

        ner = NER()
        text = "Александр Пушкин родился в Москве. " * 10
        result = ner(text)

        assert isinstance(result, (list, tuple))

    def test_ner_special_characters(self):
        """Тест: NER обрабатывает спецсимволы"""
        from mawo_slovnet import NER

        ner = NER()
        text = "!@#$%^&*()_+"
        result = ner(text)

        assert isinstance(result, (list, tuple))

    def test_morph_numbers(self):
        """Тест: Morph обрабатывает числа"""
        from mawo_slovnet import Morph

        morph = Morph()
        text = "123 456"
        result = morph(text)

        assert isinstance(result, (list, tuple))

    def test_syntax_single_word(self):
        """Тест: Syntax обрабатывает одно слово"""
        from mawo_slovnet import Syntax

        syntax = Syntax()
        text = "привет"
        result = syntax(text)

        assert isinstance(result, (list, tuple))


class TestMultipleInstances:
    """Тесты множественных экземпляров"""

    def test_multiple_ner_instances(self):
        """Тест: можно создать несколько экземпляров NER"""
        from mawo_slovnet import NER

        ner1 = NER()
        ner2 = NER()

        text = "Москва"
        result1 = ner1(text)
        result2 = ner2(text)

        assert result1 is not None
        assert result2 is not None

    def test_multiple_models_simultaneously(self):
        """Тест: можно использовать несколько моделей одновременно"""
        from mawo_slovnet import NER, Morph, Syntax

        ner = NER()
        morph = Morph()
        syntax = Syntax()

        text = "Владимир живёт в Москве"

        ner_result = ner(text)
        morph_result = morph(text)
        syntax_result = syntax(text)

        assert ner_result is not None
        assert morph_result is not None
        assert syntax_result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
