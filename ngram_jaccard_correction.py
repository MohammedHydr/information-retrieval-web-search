import os
import time
from collections import defaultdict
import itertools
import Levenshtein


def load_vocabulary_with_frequency(index_file):
    vocabulary = {}
    with open(index_file, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            term = parts[0]
            df = int(parts[1])
            vocabulary[term] = df
    return vocabulary


def load_inverted_index(index_file):
    inverted_index = {}
    with open(index_file, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            term = parts[0]
            postings = set(map(int, parts[2].split(','))) if len(parts) > 2 else set()
            inverted_index[term] = postings
    return inverted_index


def get_ngrams(word, n):
    return {word[i:i+n] for i in range(len(word) - n + 1)}


def combined_ngrams(word):
    return get_ngrams(word, 2).union(get_ngrams(word, 3))


def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0


def find_jaccard_candidates(word, vocabulary, top_n=7, min_similarity=0.3, min_df=20):
    word_ngrams = combined_ngrams(word)
    candidates = []
    for vocab_word, df in vocabulary.items():
        if df >= min_df:
            vocab_ngrams = combined_ngrams(vocab_word)
            similarity = jaccard_similarity(word_ngrams, vocab_ngrams)
            if similarity >= min_similarity:
                candidates.append((vocab_word, similarity, df))
    candidates.sort(key=lambda x: (-x[1], -x[2]))
    return [word for word, _, _ in candidates[:top_n]]


def find_edit_distance_candidates(word, vocabulary, max_distance=2, top_n=5):
    candidates = [(vocab_word, Levenshtein.distance(word, vocab_word))
                  for vocab_word in vocabulary
                  if Levenshtein.distance(word, vocab_word) <= max_distance]
    candidates.sort(key=lambda x: x[1])
    return [word for word, _ in candidates[:top_n]]


def get_best_candidates(word, vocabulary):
    candidates = find_jaccard_candidates(word, vocabulary)
    if not candidates:
        candidates = find_edit_distance_candidates(word, vocabulary)
    return candidates if candidates else [word]


def generate_alternative_queries(query, vocabulary, max_alternatives=10):
    words = query.split()
    corrected_terms = []
    for word in words:
        candidates = get_best_candidates(word, vocabulary)
        corrected_terms.append(candidates)
    all_combinations = list(itertools.product(*corrected_terms))
    return [" ".join(comb) for comb in all_combinations[:max_alternatives]]


def retrieve_documents_for_query(query, inverted_index):
    terms = query.split()
    postings = [inverted_index.get(term, set()) for term in terms]
    if not postings or not all(postings):
        return set()
    result = postings[0]
    for post in postings[1:]:
        result &= post
        if not result:
            break
    return result


def process_queries(queries, vocabulary, inverted_index):
    results = defaultdict(list)
    total_time = 0

    for query in queries:
        start_time = time.time()
        alternatives = generate_alternative_queries(query, vocabulary)

        for alt in alternatives:
            doc_count = len(retrieve_documents_for_query(alt, inverted_index))
            if doc_count > 0:
                results[query].append((alt, doc_count))

        elapsed = time.time() - start_time
        total_time += elapsed
        results[query] = sorted(results[query], key=lambda x: -x[1])[:3]
        print(f"Processed '{query}' in {elapsed:.4f} seconds with {len(alternatives)} alternatives.")

    avg_time = total_time / len(queries)
    print(f"\nTotal time: {total_time:.4f} seconds.")
    print(f"Average time per query: {avg_time:.4f} seconds.\n")

    return results


def main():
    index_file = 'output/inverted_index.txt'
    vocabulary = load_vocabulary_with_frequency(index_file)
    inverted_index = load_inverted_index(index_file)

    queries = [
        "ronaldo golas",
        "wourld cp",
        "real mdrid",
        "champios legue",
        "plarer award"
    ]

    print(f"Loaded vocabulary with {len(vocabulary)} terms.")
    print("\nStarting hybrid Jaccard + Edit Distance correction...\n")

    results = process_queries(queries, vocabulary, inverted_index)

    for query, alternatives in results.items():
        print(f"\nOriginal query: '{query}' - Top alternatives:")
        for alt, doc_count in alternatives:
            print(f"  {alt} ({doc_count} documents)")


if __name__ == "__main__":
    main()
