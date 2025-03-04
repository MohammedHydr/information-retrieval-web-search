import os
import re
import math
import collections
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup

nltk.download('stopwords')


def preprocess_text(text, stop_words, stemmer):
    """ Preprocess text: normalize, tokenize, remove stopwords, and stem. """
    text = text.lower()
    tokens = re.findall(r'\b[a-zA-Z]+\b', text)
    return [stemmer.stem(token) for token in tokens if token not in stop_words]


def extract_text_from_html(html):
    """ Extract visible text from HTML. """
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text(separator=' ', strip=True)


def load_collection_and_count_terms(crawled_dir):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    term_frequencies = collections.Counter()
    total_tokens = 0

    doc_files = sorted([f for f in os.listdir(crawled_dir) if f.endswith('.html')])

    for filename in doc_files:
        with open(os.path.join(crawled_dir, filename), 'r', encoding='utf-8') as file:
            html = file.read()
        text = extract_text_from_html(html)
        tokens = preprocess_text(text, stop_words, stemmer)
        term_frequencies.update(tokens)
        total_tokens += len(tokens)

    total_unique_terms = len(term_frequencies)

    return term_frequencies, total_tokens, total_unique_terms


def print_top_terms(term_frequencies, total_tokens, total_unique_terms, top_n=25):
    print(f"\nTotal Tokens: {total_tokens}")
    print(f"Total Unique Terms (Vocabulary Size): {total_unique_terms}")
    print(f"\nTop {top_n} Most Frequent Terms:")
    print(f"{'Rank':<5} {'Term':<15} {'Frequency':<10}")
    for rank, (term, freq) in enumerate(term_frequencies.most_common(top_n), start=1):
        print(f"{rank:<5} {term:<15} {freq:<10}")


def plot_zipf_law(term_frequencies, top_n=25, output_file="output/zipf_law_plot.png"):
    ranks = list(range(1, top_n + 1))
    frequencies = [freq for _, freq in term_frequencies.most_common(top_n)]

    plt.figure(figsize=(10, 6))
    plt.plot([math.log10(r) for r in ranks], [math.log10(f) for f in frequencies], marker='o', label='Top 25 Terms')

    plt.xlabel('Log(Rank)')
    plt.ylabel('Log(Frequency)')
    plt.title("Zipf's Law: Log-Log Plot of Term Frequencies")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_file)
    plt.close()
    print(f"\nZipf's Law plot saved as {output_file}")


def analyze_rare_terms(term_frequencies):
    rare_terms_count = sum(1 for freq in term_frequencies.values() if 1 <= freq <= 4)
    total_terms = len(term_frequencies)
    proportion_omitted = (rare_terms_count / total_terms) * 100

    print(f"\nTotal Unique Terms (Vocabulary): {total_terms}")
    print(f"Terms occurring 1-4 times: {rare_terms_count}")
    print(f"Proportion of omitted terms: {proportion_omitted:.2f}%")

    return rare_terms_count, proportion_omitted


def plot_zipf_with_omitted(term_frequencies, top_n=25, output_file="output/zipf_law_with_omission.png"):
    ranks = list(range(1, top_n + 1))
    frequencies = [freq for _, freq in term_frequencies.most_common(top_n)]

    K = frequencies[0]  # Highest frequency for Zipf line
    zipf_frequencies = [K / r for r in ranks]

    plt.figure(figsize=(10, 6))
    # Actual term data
    plt.plot([math.log10(r) for r in ranks], [math.log10(f) for f in frequencies],
             marker='o', label='Top 25 Terms')
    # Zipf’s Law line
    plt.plot([math.log10(r) for r in ranks], [math.log10(f) for f in zipf_frequencies],
             linestyle='--', color='red', label="Zipf's Law Line")

    plt.xlabel('Log(Rank)')
    plt.ylabel('Log(Frequency)')
    plt.title("Zipf's Law with Rare Term Omission")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()
    print(f"\nZipf's Law plot with omission saved as {output_file}")


def main():
    crawled_dir = "crawled_pages"

    print("Loading collection and counting term frequencies...")
    term_frequencies, total_tokens, total_unique_terms = load_collection_and_count_terms(crawled_dir)

    print_top_terms(term_frequencies, total_tokens, total_unique_terms)

    # Part 1 plot
    plot_zipf_law(term_frequencies)

    # Part 2 analysis
    rare_terms_count, proportion_omitted = analyze_rare_terms(term_frequencies)

    # Part 2 plot
    plot_zipf_with_omitted(term_frequencies)


if __name__ == "__main__":
    main()