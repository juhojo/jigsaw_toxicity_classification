def get_individual_words(col):
    assert hasattr(col, "str") and hasattr(col.str, "split")
    words = set()
    col.str.split().apply(words.update)

    return words, len(words)
