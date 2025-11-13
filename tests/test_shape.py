import pytest
from slovnet.shape import EN, NUM, OTHER, PUNCT, RU, XX, X, Xx, Xx_Xx, word_shape, x, xx
from slovnet.shape import format_shape as s
from slovnet.token import tokenize

TESTS = [
    [
        "В",
        [s(RU, X)],
    ],
    [
        "ИЛ-2",
        [s(RU, XX)],
    ],
    ["105г.", [NUM, s(RU, x), s(PUNCT, ".")]],
    ["Pal-Yz", [s(EN, Xx_Xx)]],
    ["и Я-ДаА", [s(RU, x), s(RU, OTHER)]],
    ["Прибыл на I@", [s(RU, Xx), s(RU, xx), s(EN, X), s(PUNCT, "@")]],
    ["и -‐", [s(RU, x), s(PUNCT, OTHER)]],
]


@pytest.mark.parametrize("test", TESTS)
def test_shape(test):
    text, etalon = test
    tokens = tokenize(text)
    guess = [word_shape(_.text) for _ in tokens]
    assert guess == etalon
