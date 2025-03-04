import os


def get_file_size(file_path):
    """Returns the file size in bytes."""
    return os.path.getsize(file_path) if os.path.exists(file_path) else 0


def count_terms(file_path):
    """Counts the number of unique terms in an index file."""
    if not os.path.exists(file_path):
        return 0
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)  # Each line is a term


def compare_indexes():
    """Compares single-word, biword, and positional indexes."""
    output_dir = "output"
    single_word_index_file = os.path.join(output_dir, "inverted_index.txt")
    biword_index_file = os.path.join(output_dir, "biword_index.txt")
    positional_index_file = os.path.join(output_dir, "positional_index.txt")

    # Compute sizes
    single_word_size = get_file_size(single_word_index_file)
    biword_size = get_file_size(biword_index_file)
    positional_size = get_file_size(positional_index_file)

    # Compute number of terms
    single_word_terms = count_terms(single_word_index_file)
    biword_terms = count_terms(biword_index_file)
    positional_terms = count_terms(positional_index_file)

    print("Index Comparison:")
    print(f"  - Single-word index: {single_word_terms} terms, {single_word_size} bytes")
    print(f"  - Biword index: {biword_terms} terms, {biword_size} bytes")
    print(f"  - Positional index: {positional_terms} terms, {positional_size} bytes")

    print("\nSummary:")
    print(f"  - Biword index is {biword_size / single_word_size:.2f} times the size of single-word index.")
    print(f"  - Positional index is {positional_size / single_word_size:.2f} times the size of single-word index.")
    print(f"  - Biword index has {biword_terms / single_word_terms:.2f} times the terms of single-word index.")
    print(f"  - Positional index has {positional_terms / single_word_terms:.2f} times the terms of single-word index.")


if __name__ == "__main__":
    compare_indexes()
