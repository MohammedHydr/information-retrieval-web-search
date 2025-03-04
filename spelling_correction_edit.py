import os
import time
import itertools
import Levenshtein
from collections import defaultdict


##############################
# 1. Load Vocabulary & DF    #
##############################

def load_vocabulary_with_frequency(index_file):
    """Load vocabulary with document frequencies."""
    vocabulary = {}
    with open(index_file, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            term = parts[0]
            df = int(parts[1])
            vocabulary[term] = df
    return vocabulary


##############################
# 2. Load Inverted Index     #
##############################

def load_inverted_index(index_file):
    """Load inverted index from file into a dictionary."""
    inverted_index = {}
    with open(index_file, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            term = parts[0]
            postings = set(map(int, parts[2].split(','))) if len(parts) > 2 else set()
            inverted_index[term] = postings
    return inverted_index


##############################
# 3. Edit Distance + Frequency#
##############################

def find_edit_distance_candidates(word, vocabulary, max_distance=2, top_n=5, min_df=20):
    """Find top_n vocabulary words within edit distance <= max_distance and DF >= min_df."""
    candidates = [
        (vocab_word, Levenshtein.distance(word, vocab_word), df)
        for vocab_word, df in vocabulary.items()
        if Levenshtein.distance(word, vocab_word) <= max_distance and df >= min_df
    ]
    # Sort: first by edit distance, then by highest DF
    candidates.sort(key=lambda x: (x[1], -x[2]))
    return [word for word, _, _ in candidates[:top_n]]


def generate_alternative_queries(query, vocabulary):
    """Generate alternative queries from edit distance corrections."""
    corrected_terms = []
    for word in query.split():
        candidates = find_edit_distance_candidates(word, vocabulary)
        corrected_terms.append(candidates if candidates else [word])
    alternative_queries = list(itertools.product(*corrected_terms))
    return [" ".join(alt_query) for alt_query in alternative_queries]


##############################
# 4. Retrieval Function      #
##############################

def retrieve_documents_for_query(query, inverted_index):
    """Retrieve documents using the intersection algorithm."""
    terms = query.split()
    postings = []
    for term in terms:
        postings.append(inverted_index.get(term, set()))
    if not postings:
        return set()
    result = postings[0]
    for post in postings[1:]:
        result = result.intersection(post)
        if not result:
            break
    return result


##############################
# 5. Process Queries         #
##############################

def process_edit_distance_corrections(queries, vocabulary, inverted_index):
    """Process edit distance correction for queries."""
    results = defaultdict(list)
    total_time = 0

    for query in queries:
        start_time = time.time()
        alternative_queries = generate_alternative_queries(query, vocabulary)

        for alt_query in alternative_queries:
            retrieved_docs = retrieve_documents_for_query(alt_query, inverted_index)
            results[query].append((alt_query, len(retrieved_docs)))

        elapsed = time.time() - start_time
        total_time += elapsed

        # Keep top-3 alternative queries with the most results
        results[query] = sorted(results[query], key=lambda x: x[1], reverse=True)[:3]

        print(f"Processed '{query}' in {elapsed:.4f} seconds with {len(alternative_queries)} alternatives.")

    avg_time = total_time / len(queries)
    print(f"\nTotal time: {total_time:.4f} seconds.")
    print(f"Average time per query: {avg_time:.4f} seconds.\n")

    return results


##############################
# 6. Main Execution          #
##############################

def main():
    index_file = 'output/inverted_index.txt'
    vocabulary = load_vocabulary_with_frequency(index_file)
    inverted_index = load_inverted_index(index_file)

    queries_with_mistakes = [
        "ronaldo golas",
        "wourld cp",
        "real mdrid",
        "champios legue",
        "plarer award"
    ]

    print(f"Loaded vocabulary with {len(vocabulary)} terms.")
    print("\nStarting improved edit distance correction...\n")

    results = process_edit_distance_corrections(queries_with_mistakes, vocabulary, inverted_index)

    # Display final top-3 alternatives for each query
    for original_query, alternatives in results.items():
        print(f"\nOriginal query: '{original_query}' - Top 3 alternatives:")
        for alt_query, doc_count in alternatives:
            print(f"  {alt_query} ({doc_count} documents)")


if __name__ == "__main__":
    main()
