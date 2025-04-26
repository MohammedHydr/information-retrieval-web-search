import os
import math
import re
import json
from collections import defaultdict, Counter
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')


########################
# Preprocessing Helpers
########################
def extract_article_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    selectors = [('article', None), ('div', 'article-body'), ('div', 'article__content'), ('section', 'articleContent')]
    for tag, class_name in selectors:
        container = soup.find(tag, class_=class_name) if class_name else soup.find(tag)
        if container:
            return container.get_text(separator=' ', strip=True)
    return soup.get_text(separator=' ', strip=True)


def preprocess(text, stop_words, stemmer):
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return [stemmer.stem(token) for token in tokens if token not in stop_words]


########################
# TF-IDF Index Builder
########################
def build_tfidf_index(crawled_dir):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    doc_files = [f for f in os.listdir(crawled_dir) if f.endswith('.html')]
    N = len(doc_files)
    term_doc_freq = defaultdict(int)  # df_t
    doc_term_freq = {}  # tf_t,d
    doc_id_map = {}

    for doc_id, filename in enumerate(doc_files):
        filepath = os.path.join(crawled_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            html = f.read()
        text = extract_article_text(html)
        tokens = preprocess(text, stop_words, stemmer)
        term_counts = Counter(tokens)
        doc_term_freq[doc_id] = term_counts
        doc_id_map[doc_id] = filename

        for term in term_counts:
            term_doc_freq[term] += 1

    tfidf_index = defaultdict(list)
    for doc_id, term_counts in doc_term_freq.items():
        for term, tf in term_counts.items():
            df = term_doc_freq[term]
            idf = math.log(N / df)
            tfidf = (1 + math.log(tf)) * idf
            tfidf_index[term].append((doc_id, tfidf))

    for postings in tfidf_index.values():
        postings.sort(key=lambda x: x[1], reverse=True)  # Sort by tf-idf descending

    return tfidf_index, doc_id_map


########################
# Save Index & Mapping
########################
def save_tfidf_index(tfidf_index, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for term, postings in sorted(tfidf_index.items()):
            posting_str = ",".join(f"{doc}:{score:.6f}" for doc, score in postings)
            f.write(f"{term} {posting_str}\n")


def save_doc_id_map(doc_id_map, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc_id, filename in sorted(doc_id_map.items()):
            f.write(f"{doc_id} {filename}\n")


########################
# Main Entrypoint
########################
def main():
    crawled_dir = "crawled_pages"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    tfidf_index_file = os.path.join(output_dir, "tfidf_index.txt")
    doc_id_map_file = os.path.join(output_dir, "doc_id_map.txt")

    print("Building TF-IDF index...")
    tfidf_index, doc_id_map = build_tfidf_index(crawled_dir)

    print("Saving index...")
    save_tfidf_index(tfidf_index, tfidf_index_file)

    print("Saving document mapping...")
    save_doc_id_map(doc_id_map, doc_id_map_file)

    print(f"TF-IDF index saved to {tfidf_index_file}")
    print(f"Document ID mapping saved to {doc_id_map_file}")


if __name__ == "__main__":
    main()
