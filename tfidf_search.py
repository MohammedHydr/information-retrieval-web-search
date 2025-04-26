import os
import time
import re
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')


###############################
# Preprocessing and Utilities
###############################
def preprocess_query(query, stop_words, stemmer):
    query = query.lower()
    tokens = re.findall(r'\b\w+\b', query)
    return [stemmer.stem(token) for token in tokens if token not in stop_words]


def load_tfidf_index(filepath):
    index = defaultdict(list)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            term = parts[0]
            postings_raw = " ".join(parts[1:])  # In case of spaces
            postings = postings_raw.split(',')
            for pair in postings:
                if ':' in pair:
                    try:
                        doc_id, score = pair.split(':')
                        index[term].append((int(doc_id), float(score)))
                    except ValueError:
                        continue  # Skip malformed pairs
    return index



def load_doc_id_map(filepath):
    doc_map = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            doc_id, filename = line.strip().split(maxsplit=1)
            doc_map[int(doc_id)] = filename
    return doc_map


#########################
# Query Scoring & Rank
#########################
def rank_documents(query, index, stop_words, stemmer):
    tokens = preprocess_query(query, stop_words, stemmer)
    doc_scores = defaultdict(float)
    for token in tokens:
        for doc_id, score in index.get(token, []):
            doc_scores[doc_id] += score
    # Sort documents by score descending
    ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs[:5]  # Top 5


#########################
# Main Search Loop
#########################
def main():
    index_file = os.path.join("output", "tfidf_index.txt")
    doc_map_file = os.path.join("output", "doc_id_map.txt")

    print("Loading tf-idf index...")
    tfidf_index = load_tfidf_index(index_file)
    doc_id_map = load_doc_id_map(doc_map_file)

    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    benchmark_queries = [
        "Real Madrid Champions League victories",
        "La Liga top scorers 2023",
        "Upcoming Real Madrid transfer rumors",
        "UEFA match results for Real Madrid",
        "Historical performance Real Madrid 2010-2020"
    ]

    for query in benchmark_queries:
        print(f"\nProcessing: '{query}'")
        start = time.time()
        results = rank_documents(query, tfidf_index, stop_words, stemmer)
        elapsed = time.time() - start
        print(f"Top-5 results (doc_id:score): {results}")
        for doc_id, score in results:
            print(f"  -> {doc_id_map[doc_id]}: {score:.4f}")
        print(f"Execution time: {elapsed:.6f} seconds")


if __name__ == "__main__":
    main()
