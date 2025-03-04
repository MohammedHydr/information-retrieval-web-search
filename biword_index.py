import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')


def preprocess_text(text, stop_words, stemmer):
    """
    Tokenizes, normalizes, removes stopwords, and stems text.
    """
    text = text.lower()
    tokens = re.findall(r'\b[a-zA-Z]+\b', text)  # Keep only words, remove numbers/symbols
    processed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return processed_tokens


def extract_text_from_html(html):
    """
    Extracts visible text from an HTML document using BeautifulSoup.
    """
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text(separator=' ', strip=True)


def process_document(file_path, doc_id, stop_words, stemmer):
    """
    Processes a single document and extracts its biwords.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        html = f.read()

    text = extract_text_from_html(html)
    tokens = preprocess_text(text, stop_words, stemmer)

    # Generate biwords (two consecutive words)
    biwords = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1) if
               len(tokens[i]) > 1 and len(tokens[i + 1]) > 1]

    return doc_id, biwords


def build_biword_index(crawled_dir):
    """
    Constructs a biword inverted index from the HTML documents in `crawled_dir` using multithreading.
    """
    biword_index = defaultdict(set)
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
            doc_id, biwords = future.result()
            for biword in biwords:
                biword_index[biword].add(doc_id)
            doc_id_map[doc_id] = futures[future][1]

    return biword_index, doc_id_map


def save_biword_index(biword_index, output_file):
    """
    Saves the biword inverted index to a file, filtering out rare biwords.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for biword, doc_ids in sorted(biword_index.items()):
            if len(doc_ids) > 1:  # Ignore biwords appearing in only one document
                df = len(doc_ids)
                postings_list = ','.join(map(str, sorted(doc_ids)))
                f.write(f"{biword[0]}_{biword[1]} {df} {postings_list}\n")


def save_doc_id_map(doc_id_map, output_file):
    """
    Saves the document ID mapping to a file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc_id, filename in sorted(doc_id_map.items()):
            f.write(f"{doc_id} {filename}\n")


def main():
    crawled_dir = "crawled_pages"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    biword_index_file = os.path.join(output_dir, "biword_index.txt")
    doc_id_map_file = os.path.join(output_dir, "doc_id_map.txt")

    print("Building biword inverted index...")
    biword_index, doc_id_map = build_biword_index(crawled_dir)

    print("Saving biword index to file...")
    save_biword_index(biword_index, biword_index_file)

    print("Saving document ID mapping...")
    save_doc_id_map(doc_id_map, doc_id_map_file)

    print("Biword index construction completed.")


if __name__ == "__main__":
    main()