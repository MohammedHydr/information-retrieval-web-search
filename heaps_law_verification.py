import os
import re
import math
import random
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt

# Ensure stopwords are downloaded
nltk.download('stopwords')


def preprocess_text(text, stop_words, stemmer):
    """ Normalize, tokenize, remove stopwords, and stem. """
    text = text.lower()
    tokens = re.findall(r'\b[a-zA-Z]+\b', text)
    return [stemmer.stem(token) for token in tokens if token not in stop_words]


def extract_text_from_html(html):
    """ Extract visible text from HTML. """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text(separator=' ', strip=True)


def load_collection(crawled_dir):
    """ Load and preprocess all documents with shuffled order. """
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    doc_files = [f for f in os.listdir(crawled_dir) if f.endswith('.html')]

    # Shuffle document order to avoid topic clustering
    random.seed(42)  # Fixed seed for consistent results
    random.shuffle(doc_files)

    collection_tokens = []

    for filename in doc_files:
        with open(os.path.join(crawled_dir, filename), 'r', encoding='utf-8') as file:
            html = file.read()
        text = extract_text_from_html(html)
        tokens = preprocess_text(text, stop_words, stemmer)
        collection_tokens.extend(tokens)

    return collection_tokens


def compute_heap_data(tokens, parts=20):
    """ Split tokens into parts and compute cumulative tokens and unique terms. """
    part_size = len(tokens) // parts
    cumulative_tokens = []
    cumulative_vocab = set()
    token_counts = []
    unique_term_counts = []

    for i in range(parts):
        part_tokens = tokens[i * part_size: (i + 1) * part_size]
        cumulative_tokens.extend(part_tokens)
        cumulative_vocab.update(part_tokens)
        token_counts.append(len(cumulative_tokens))
        unique_term_counts.append(len(cumulative_vocab))

    return token_counts, unique_term_counts


def plot_heap_law(token_counts, unique_term_counts, output_file):
    """ Plot log-log relation and save the figure. """
    plt.figure(figsize=(10, 6))
    plt.plot([math.log10(n) for n in token_counts],
             [math.log10(m) for m in unique_term_counts],
             marker='o', label='Heap’s Law Data')

    plt.xlabel('Log(Number of Tokens)')
    plt.ylabel('Log(Number of Unique Terms)')
    plt.title('Heap’s Law Verification')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()
    print(f"Heap's Law plot saved as {output_file}")


def main():
    crawled_dir = "crawled_pages"
    output_plot = "output/heaps_law_plot.png"

    print("Loading collection and preprocessing...")
    tokens = load_collection(crawled_dir)
    print(f"Total tokens: {len(tokens)}")

    print("Computing Heap's Law data...")
    token_counts, unique_term_counts = compute_heap_data(tokens)

    print("Plotting Heap's Law...")
    plot_heap_law(token_counts, unique_term_counts, output_plot)

    # Optional: Print the data
    print("\nParts | Tokens | Unique Terms")
    for i, (n, m) in enumerate(zip(token_counts, unique_term_counts), 1):
        print(f"{i:2}    | {n}   | {m}")


if __name__ == "__main__":
    main()
