import os
import time
from collections import defaultdict


def load_positional_index(index_file):
    positional_index = defaultdict(lambda: defaultdict(list))
    if not os.path.exists(index_file):
        print("Error: Positional index file not found!")
        return positional_index

    with open(index_file, 'r', encoding='utf-8') as f:
        for line in f:
            term, *postings = line.strip().split()
            for posting in postings:
                posting = posting.rstrip(';')  # Remove any trailing semicolons
                doc_id, positions = posting.split(':')
                positions = list(map(int, positions.split(',')))
                positional_index[term][int(doc_id)] = positions

    return positional_index


def phrase_query(phrase, positional_index):
    words = phrase.lower().split()
    if not words:
        return set()

    first_word = words[0]
    result_docs = set(positional_index.get(first_word, {}).keys())

    for i in range(1, len(words)):
        current_word = words[i]
        current_docs = set(positional_index.get(current_word, {}).keys())
        result_docs &= current_docs

    final_docs = set()
    for doc_id in result_docs:
        positions = [positional_index[word][doc_id] for word in words]
        for pos in positions[0]:
            if all((pos + i) in positions[i] for i in range(1, len(words))):
                final_docs.add(doc_id)
                break

    return final_docs


def measure_query_time(queries, positional_index):
    total_time = 0
    query_results = {}

    for query in queries:
        start_time = time.perf_counter()
        result_docs = phrase_query(query, positional_index)
        elapsed_time = time.perf_counter() - start_time
        total_time += elapsed_time
        query_results[query] = (len(result_docs), elapsed_time)

    avg_time = total_time / len(queries) if queries else 0
    return query_results, avg_time


def main():
    positional_index_file = "output/positional_index.txt"
    positional_index = load_positional_index(positional_index_file)

    if not positional_index:
        print("No positional index found. Make sure you have generated it.")
        return

    queries = [
        "real madrid win",
        "transfer market",
        "world cup",
        "cristiano ronaldo lionel messi",
        "copa del rey"
    ]

    print("\nRunning positional phrase queries and measuring performance...")
    query_results, avg_time = measure_query_time(queries, positional_index)

    for query, (doc_count, time_taken) in query_results.items():
        print(f"Query: '{query}' → Found in {doc_count} documents | Time taken: {time_taken:.6f} sec")

    print(f"\nAverage positional query retrieval time: {avg_time:.6f} sec")


if __name__ == "__main__":
    main()
