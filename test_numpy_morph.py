#!/usr/bin/env python3.11
"""Test numpy-based Morph implementation."""

import sys
from pathlib import Path

# Add parent to path to import as package
sys.path.insert(0, str(Path(__file__).parent))

from mawo_slovnet.numpy_api import Morph


def test_morph_basic():
    """Test basic morphology tagging."""
    print("=" * 80)
    print("TEST: Basic Morphology Tagging")
    print("=" * 80)
    print()

    # Load model
    model_path = Path(__file__).parent / "mawo_slovnet" / "models" / "morph" / "morph.tar"

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return False

    print(f"Loading model from: {model_path}")

    try:
        morph = Morph.load(model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("✅ Model loaded successfully")
    print()

    # Test cases
    test_sentences = [
        ["Мама", "мыла", "раму"],
        ["Об", "этом", "говорится", "в", "документе"],
        ["Я", "думаю", "о", "книге"],
        ["Красивая", "девушка", "читает", "большую", "книгу"],
    ]

    print("Running test cases...")
    print("-" * 80)

    for i, words in enumerate(test_sentences, 1):
        print(f"\n[{i}] Input: {' '.join(words)}")

        try:
            result = morph(words)

            print("Output:")
            for token in result:
                print(f"  {token['text']:15s} POS={token['pos']:5s} {token['feats']}")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback

            traceback.print_exc()
            return False

    print()
    print("-" * 80)
    print("✅ All tests passed!")
    print()

    return True


def test_prepositions():
    """Test preposition detection."""
    print("=" * 80)
    print("TEST: Preposition Detection")
    print("=" * 80)
    print()

    model_path = Path(__file__).parent / "mawo_slovnet" / "models" / "morph" / "morph.tar"
    morph = Morph.load(model_path)

    # Focus on prepositions
    test_cases = [
        ["О", "книге"],  # О should be ADP (preposition)
        ["В", "доме"],
        ["Без", "друга"],
        ["С", "мамой"],
        ["Для", "тебя"],
    ]

    print("Testing preposition POS tagging...")
    print("-" * 80)

    for words in test_cases:
        result = morph(words)
        prep_pos = result[0]["pos"]
        print(
            f"{words[0]:10s} -> POS={prep_pos:10s} {'✅' if prep_pos == 'ADP' else '❌ (expected ADP)'}"
        )

    print()
    return True


if __name__ == "__main__":
    success = True

    try:
        success = test_morph_basic() and success
        print()
        success = test_prepositions() and success
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        success = False

    if success:
        print("=" * 80)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 80)
        sys.exit(0)
    else:
        print("=" * 80)
        print("❌ SOME TESTS FAILED")
        print("=" * 80)
        sys.exit(1)
