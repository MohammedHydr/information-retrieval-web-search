import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

nltk.download('stopwords')


def preprocess_text(text, stop_words, stemmer):
    text = text.lower()
    tokens = re.findall(r'\b[a-zA-Z]+\b', text)
    return [stemmer.stem(token) for token in tokens if token not in stop_words and len(token) > 1]


def extract_text_from_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text(separator=' ', strip=True)


def process_document(file_path, doc_id, stop_words, stemmer):
    with open(file_path, 'r', encoding='utf-8') as f:
        html = f.read()

    text = extract_text_from_html(html)
    tokens = preprocess_text(text, stop_words, stemmer)

    positions = defaultdict(list)
    for position, token in enumerate(tokens):
        positions[token].append(position)

    return doc_id, positions


def build_positional_index(crawled_dir):
    positional_index = defaultdict(lambda: defaultdict(list))
    doc_id_map = {}
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    doc_files = [f for f in os.listdir(crawled_dir) if f.endswith('.html')]

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(process_document, os.path.join(crawled_dir, filename), doc_id, stop_words, stemmer): (
            doc_id, filename)
            for doc_id, filename in enumerate(doc_files)}

        for future in futures:
            doc_id, positions = future.result()
            for token, pos_list in positions.items():
                positional_index[token][doc_id].extend(pos_list)
            doc_id_map[doc_id] = futures[future][1]

    return positional_index, doc_id_map


def save_positional_index(positional_index, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for term, postings in sorted(positional_index.items()):
            posting_str = "; ".join(
                f"{doc_id}:{','.join(map(str, positions))}"
                for doc_id, positions in sorted(postings.items())
            )
            f.write(f"{term} {posting_str}\n")


def save_doc_id_map(doc_id_map, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc_id, filename in sorted(doc_id_map.items()):
            f.write(f"{doc_id} {filename}\n")


def main():
    crawled_dir = "crawled_pages"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    positional_index_file = os.path.join(output_dir, "positional_index.txt")
    doc_id_map_file = os.path.join(output_dir, "doc_id_map.txt")

    print("Building positional index...")
    positional_index, doc_id_map = build_positional_index(crawled_dir)

    print("Saving positional index to file...")
    save_positional_index(positional_index, positional_index_file)

    print("Saving document ID mapping...")
    save_doc_id_map(doc_id_map, doc_id_map_file)

    print("Positional index construction completed.")


if __name__ == "__main__":
    main()
