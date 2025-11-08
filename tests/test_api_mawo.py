"""
Адаптированные API тесты для mawo-slovnet
Используют наш fork с локальными моделями
"""

import pytest


@pytest.fixture(scope='module')
def ner():
    """Создает NER модель из нашего форка"""
    try:
        from slovnet import NER
        # Попробуем использовать оригинальный slovnet API
        from navec import Navec
        from os.path import join, dirname

        # Попробуем использовать локальные модели если они есть
        module_dir = dirname(dirname(__file__))
        navec_path = join(module_dir, 'data', 'test', 'navec_news_v1_1B_250K_300d_100q.tar')
        ner_path = join(module_dir, 'data', 'test', 'slovnet_ner_news_v1.tar')

        try:
            navec = Navec.load(navec_path)
            return NER.load(ner_path).navec(navec)
        except Exception:
            # Fallback: используем без navec
            return NER.load(ner_path)
    except Exception as e:
        pytest.skip(f"Cannot load NER model: {e}")


@pytest.fixture(scope='module')
def morph():
    """Создает Morph модель из нашего форка"""
    try:
        from slovnet import Morph
        from navec import Navec
        from os.path import join, dirname

        module_dir = dirname(dirname(__file__))
        navec_path = join(module_dir, 'data', 'test', 'navec_news_v1_1B_250K_300d_100q.tar')
        morph_path = join(module_dir, 'data', 'test', 'slovnet_morph_news_v1.tar')

        try:
            navec = Navec.load(navec_path)
            return Morph.load(morph_path).navec(navec)
        except Exception:
            return Morph.load(morph_path)
    except Exception as e:
        pytest.skip(f"Cannot load Morph model: {e}")


@pytest.fixture(scope='module')
def syntax():
    """Создает Syntax модель из нашего форка"""
    try:
        from slovnet import Syntax
        from navec import Navec
        from os.path import join, dirname

        module_dir = dirname(dirname(__file__))
        navec_path = join(module_dir, 'data', 'test', 'navec_news_v1_1B_250K_300d_100q.tar')
        syntax_path = join(module_dir, 'data', 'test', 'slovnet_syntax_news_v1.tar')

        try:
            navec = Navec.load(navec_path)
            return Syntax.load(syntax_path).navec(navec)
        except Exception:
            return Syntax.load(syntax_path)
    except Exception as e:
        pytest.skip(f"Cannot load Syntax model: {e}")


def test_ner(ner):
    """Тест NER на реальном тексте"""
    text = 'На них удержали лидерство действующие руководители и партии — Денис Пушилин и «Донецкая республика» в ДНР и Леонид Пасечник с движением «Мир Луганщине» в ЛНР.'

    markup = ner(text)

    pred = []
    for span in markup.spans:
        chunk = markup.text[span.start:span.stop]
        pred.append([span.type, chunk])

    # Проверяем что найдены основные сущности
    assert len(pred) > 0, "Expected to find entities"

    # Проверяем точное совпадение (как в оригинале)
    assert pred == [
        ['PER', 'Денис Пушилин'],
        ['ORG', 'Донецкая республика'],
        ['LOC', 'ДНР'],
        ['PER', 'Леонид Пасечник'],
        ['ORG', 'Мир Луганщине'],
        ['LOC', 'ЛНР']
    ]


def test_morph(morph):
    """Тест морфологического анализа"""
    words = ['Об', 'этом', 'говорится', 'в', 'документе', ',', 'опубликованном', 'в', 'официальном', 'журнале', 'Евросоюза', '.']

    markup = morph(words)

    pred = [
        [_.text, _.tag]
        for _ in markup.tokens
    ]

    # Проверяем точное совпадение (как в оригинале)
    assert pred == [
        ['Об', 'ADP'],
        ['этом', 'PRON|Animacy=Inan|Case=Loc|Gender=Neut|Number=Sing'],
        ['говорится', 'VERB|Aspect=Imp|Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin|Voice=Pass'],
        ['в', 'ADP'],
        ['документе', 'NOUN|Animacy=Inan|Case=Loc|Gender=Masc|Number=Sing'],
        [',', 'PUNCT'],
        ['опубликованном', 'VERB|Aspect=Perf|Case=Loc|Gender=Masc|Number=Sing|Tense=Past|VerbForm=Part|Voice=Pass'],
        ['в', 'ADP'],
        ['официальном', 'ADJ|Case=Loc|Degree=Pos|Gender=Masc|Number=Sing'],
        ['журнале', 'NOUN|Animacy=Inan|Case=Loc|Gender=Masc|Number=Sing'],
        ['Евросоюза', 'PROPN|Animacy=Inan|Case=Gen|Gender=Masc|Number=Sing'],
        ['.', 'PUNCT']
    ]


def test_syntax(syntax):
    """Тест синтаксического анализа"""
    words = ['Опубликованы', 'новые', 'данные', 'по', 'заражению', 'коронавирусом', 'в', 'Москве']

    markup = syntax(words)

    ids = {_.id: _ for _ in markup.tokens}
    pred = []
    for token in markup.tokens:
        head = ids.get(token.head_id)
        if head:
            pred.append([token.text, head.rel, head.text])
        else:
            pred.append(token.text)

    # Проверяем точное совпадение (как в оригинале)
    assert pred == [
        'Опубликованы',
        ['новые', 'nsubj:pass', 'данные'],
        ['данные', 'root', 'Опубликованы'],
        ['по', 'nmod', 'заражению'],
        ['заражению', 'nsubj:pass', 'данные'],
        ['коронавирусом', 'nmod', 'заражению'],
        ['в', 'obl', 'Москве'],
        ['Москве', 'nmod', 'коронавирусом']
    ]
