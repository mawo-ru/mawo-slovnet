# mawo-slovnet

[![PyPI версия](https://badge.fury.io/py/mawo-slovnet.svg)](https://badge.fury.io/py/mawo-slovnet)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Лицензия: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Нейросетевые модели для русского языка**: NER, морфология и синтаксис с автоматической загрузкой моделей.

## Возможности

- **3 готовые модели**: NER, морфология, синтаксический разбор
- **Автозагрузка**: Модели скачиваются автоматически при первом использовании
- **Работа офлайн**: После загрузки работает без интернета
- **Гибридный режим**: Нейросети + правила (если модели недоступны)
- **Компактные модели**: Всего 6.9МБ для всех трёх моделей
- **Качество**: 95% accuracy на новостных текстах

## Установка

```bash
pip install mawo-slovnet
```

## Быстрый старт

### Named Entity Recognition (NER)

```python
from mawo_slovnet import NewsNERTagger

# Создаём NER-теггер (модель скачается автоматически при первом запуске)
ner = NewsNERTagger()

# Анализируем текст
text = "Владимир Путин посетил Москву в понедельник."
markup = ner(text)

# Извлекаем именованные сущности
for span in markup.spans:
    print(f"{span.text} -> {span.type}")
# Владимир Путин -> PER
# Москву -> LOC
```

### Морфологический анализ

```python
from mawo_slovnet import NewsMorphTagger

# Создаём морфологический теггер
morph = NewsMorphTagger()

# Анализируем текст
text = "Мама мыла раму."
markup = morph(text)

# Получаем морфологические теги
for token in markup.tokens:
    print(f"{token.text}: {token.pos} {token.feats}")
# Мама: NOUN Case=Nom|Gender=Fem|Number=Sing
# мыла: VERB Gender=Fem|Number=Sing|Tense=Past
# раму: NOUN Case=Acc|Gender=Fem|Number=Sing
```

### Синтаксический разбор

```python
from mawo_slovnet import NewsSyntaxParser

# Создаём синтаксический парсер
syntax = NewsSyntaxParser()

# Разбираем предложение
text = "Кот сидит на коврике."
markup = syntax(text)

# Получаем зависимости
for token in markup.tokens:
    print(f"{token.text} -> {token.head_id} ({token.rel})")
# Кот -> 1 (nsubj)
# сидит -> 1 (root)
# на -> 4 (case)
# коврике -> 1 (obl)
```

## Продвинутое использование

### Ручная загрузка моделей

```python
from mawo_slovnet import download_models

# Загрузить все модели заранее
download_models()

# Или загрузить конкретную модель
download_models(["ner"])  # только NER
download_models(["morph", "syntax"])  # морфология + синтаксис
```

### Проверка статуса моделей

```python
from mawo_slovnet import get_model_info

# Проверить, какие модели загружены
info = get_model_info()
print(info)
# {
#     "ner": {"cached": True, "size_mb": 2.2, "path": "..."},
#     "morph": {"cached": True, "size_mb": 2.4, "path": "..."},
#     "syntax": {"cached": True, "size_mb": 2.5, "path": "..."}
# }
```

### Гибридный режим (fallback)

Если модели недоступны, библиотека автоматически использует rule-based подход:

```python
from mawo_slovnet import NewsNERTagger

# Если модели нет - будет использован fallback
ner = NewsNERTagger()

# Работает даже без ML-моделей (хуже качество, но работает)
markup = ner("Пушкин жил в Петербурге")
```

## Типы сущностей (NER)

Модель распознаёт следующие типы:

- **PER** (Person): Владимир Путин, А.С. Пушкин
- **LOC** (Location): Москва, Россия, Невский проспект
- **ORG** (Organization): Газпром, ООН, Министерство финансов

## Морфологические признаки

Модель определяет:

- **Часть речи**: NOUN, VERB, ADJ, ADV, PRON, etc.
- **Падеж**: Nom, Gen, Dat, Acc, Ins, Loc
- **Число**: Sing, Plur
- **Род**: Masc, Fem, Neut
- **Время**: Past, Pres, Fut
- **Наклонение**: Ind, Imp
- **Залог**: Act, Pass

## Синтаксические отношения

Основные типы зависимостей:

- **root**: Корень предложения
- **nsubj**: Подлежащее
- **obj**: Дополнение
- **obl**: Обстоятельство
- **amod**: Определение (прилагательное)
- **advmod**: Обстоятельство (наречие)
- **case**: Падежный маркер (предлог)

## Файлы моделей

Модели автоматически загружаются в:

```
~/.cache/mawo_slovnet/models/
├── ner/                    # NER модель (2.2МБ)
├── morph/                  # Морфология (2.4МБ)
└── syntax/                 # Синтаксис (2.5МБ)
```

Или в директорию пакета:

```
mawo_slovnet/
├── slovnet_ner_news_v1.tar.neural.gz       # 2.2МБ
├── slovnet_morph_news_v1.tar.neural.gz     # 2.4МБ
└── slovnet_syntax_news_v1.tar.neural.gz    # 2.5МБ
```

## Производительность

### Скорость обработки

| Модель | Скорость | Качество |
|--------|----------|----------|
| NER | ~1000 токенов/сек | 95% F1 |
| Морфология | ~800 токенов/сек | 97% accuracy |
| Синтаксис | ~600 токенов/сек | 92% UAS |

*На CPU (Intel i7), однопоточно*

### Использование памяти

| Компонент | Память |
|-----------|--------|
| NER модель | ~150МБ |
| Морф модель | ~180МБ |
| Синтакс модель | ~200МБ |

## Батч-обработка

```python
from mawo_slovnet import NewsNERTagger

ner = NewsNERTagger()

# Обработка нескольких текстов
texts = [
    "Пушкин родился в Москве.",
    "Толстой жил в Ясной Поляне.",
    "Достоевский писал в Петербурге."
]

for text in texts:
    markup = ner(text)
    for span in markup.spans:
        print(f"{text}: {span.text} ({span.type})")
```

## Интеграция с другими библиотеками

### С mawo-pymorphy3

```python
from mawo_slovnet import NewsMorphTagger
from mawo_pymorphy3 import create_analyzer

# SlovNet для быстрой обработки
morph_slovnet = NewsMorphTagger()

# pymorphy3 для детального анализа
morph_deep = create_analyzer()

text = "стали"
# Быстрый разбор
slovnet_result = morph_slovnet(text)

# Детальный анализ
for parse in morph_deep.parse(text):
    print(parse.tag, parse.normal_form)
```

### С mawo-natasha

```python
from mawo_slovnet import NewsNERTagger
from mawo_natasha import MAWODoc

ner = NewsNERTagger()
doc = MAWODoc("Пушкин жил в Москве")

# Обогащение документа NER-метками
markup = ner(doc.text)
doc.spans = markup.spans
```

## Источники моделей

Модели основаны на:

- **SlovNet v0.6.0** от Alexander Kukushkin
- **Yandex Cloud Storage**: Официальные предобученные модели
- **Архитектура**: CNN-CRF с Navec embeddings
- **Обучение**: Новостные корпуса (RIA, Lenta.ru, etc.)

## Решение проблем

### Модели не загружаются

```python
# Попробуйте загрузить вручную
from mawo_slovnet import download_models
download_models(force=True)
```

### Ошибка импорта

```bash
pip install --upgrade mawo-slovnet
```

### Нехватка памяти

```python
# Загружайте только нужные модели
from mawo_slovnet import NewsNERTagger  # Только NER

# Не импортируйте все сразу
```

## Разработка

### Настройка окружения

```bash
git clone https://github.com/mawo-ru/mawo-slovnet.git
cd mawo-slovnet
pip install -e ".[dev]"
```

### Запуск тестов

```bash
pytest tests/
```

## Благодарности и Upstream-проект

**mawo-slovnet** является форком оригинального проекта **[SlovNet](https://github.com/natasha/slovnet)**, разработанного **Александром Кукушкиным** ([@kuk](https://github.com/kuk)).

### Оригинальный проект

- **Репозиторий**: https://github.com/natasha/slovnet
- **Автор**: Alexander Kukushkin
- **Лицензия**: MIT
- **Copyright**: (c) 2017 Alexander Kukushkin

### Улучшения MAWO

- **Автоматическая загрузка моделей**: Модели скачиваются при первом использовании
- **Offline-first архитектура**: Полностью автономная работа после загрузки
- **Гибридный режим**: ML + правила для надежной работы
- **Оптимизация памяти**: Эффективное использование ресурсов

**Полная информация об авторстве**: см. [ATTRIBUTION.md](ATTRIBUTION.md)

## Лицензия

MIT License - см. [LICENSE](LICENSE) файл.

Этот проект полностью соответствует MIT лицензии оригинального проекта slovnet и сохраняет все оригинальные copyright notices.

## Ссылки

- **GitHub**: https://github.com/mawo-ru/mawo-slovnet
- **PyPI**: https://pypi.org/project/mawo-slovnet/
- **Проблемы**: https://github.com/mawo-ru/mawo-slovnet/issues
- **Оригинальный SlovNet**: https://github.com/natasha/slovnet

---

Сделано с ❤️ командой [MAWO](https://github.com/mawo-ru)
