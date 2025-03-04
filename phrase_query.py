import os
import time

from nltk import PorterStemmer

stemmer = PorterStemmer()


def load_biword_index(index_file):
    """Loads the biword index from a text file."""
    biword_index = {}
    if not os.path.exists(index_file):
        print("Error: Biword index file not found!")
        return biword_index

    with open(index_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            biword = parts[0]  # Example: real_madrid
            doc_ids = set(map(int, parts[2].split(',')))
            biword_index[biword] = doc_ids

    return biword_index


def phrase_to_biwords(phrase):
    """Converts a phrase query into a list of correctly stemmed biwords."""
    words = phrase.lower().split()
    stemmed_words = [stemmer.stem(word) for word in words]  # Apply stemming to match the index
    return [f"{stemmed_words[i]}_{stemmed_words[i + 1]}" for i in range(len(stemmed_words) - 1)]


def retrieve_documents(phrase, biword_index):
    """Finds documents containing all biwords from the query."""
    biwords = phrase_to_biwords(phrase)

    if not biwords:
        return set()

    doc_lists = [biword_index.get(biword, set()) for biword in biwords]

    if not any(doc_lists):
        print(f"⚠️ No results found! The biwords {biwords} do not exist in the index.")

    return set.intersection(*doc_lists) if doc_lists else set()


def measure_query_time(queries, biword_index):
    total_time = 0
    query_results = {}

    for query in queries:
        start_time = time.perf_counter()
        result_docs = retrieve_documents(query, biword_index)
        elapsed_time = time.perf_counter() - start_time
        total_time += elapsed_time
        query_results[query] = (len(result_docs), elapsed_time)

    avg_time = total_time / len(queries) if queries else 0
    return query_results, avg_time

def main():
    biword_index_file = "output/biword_index.txt"
    biword_index = load_biword_index(biword_index_file)

    if not biword_index:
        print("No biword index found. Make sure you have generated it.")
        return

    # Example Queries
    queries = [
        "real madrid win",
        "transfer market",
        "world cup",
        "cristiano ronaldo lionel messi",
        "copa del rey"
    ]

    print("\nRunning phrase queries and measuring performance...")
    query_results, avg_time = measure_query_time(queries, biword_index)

    for query, (doc_count, time_taken) in query_results.items():
        print(f"Query: '{query}' → Found in {doc_count} documents | Time taken: {time_taken:.6f} sec")

    print(f"\nAverage query retrieval time: {avg_time:.6f} sec")


if __name__ == "__main__":
    main()
