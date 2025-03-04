import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

# Ensure necessary NLTK resources are available
nltk.download('stopwords')


##############################
# 1. Text Extraction Helpers #
##############################

def extract_article_text(html):
    """
    Extracts text from the HTML content using common selectors.
    """
    soup = BeautifulSoup(html, 'html.parser')
    article_selectors = [
        ('article', None),
        ('div', 'article-body'),
        ('div', 'article__content'),
        ('section', 'articleContent'),
    ]
    for tag, class_name in article_selectors:
        container = soup.find(tag, class_=class_name) if class_name else soup.find(tag)
        if container:
            return container.get_text(separator=' ', strip=True)
    return soup.get_text(separator=' ', strip=True)


###########################
# 2. Preprocessing Helper #
###########################

def preprocess_text(text, stop_words, stemmer):
    """
    Normalizes, tokenizes, removes stopwords, and stems the text.
    """
    text = text.lower()  # Normalize to lower-case
    # Tokenize: extract words using regular expression
    tokens = re.findall(r'\b\w+\b', text)
    # Remove stopwords and apply stemming
    processed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return processed_tokens


##############################
# 3. Inverted Index Builder  #
##############################

def build_inverted_index(crawled_dir):
    """
    Processes HTML files in `crawled_dir`, builds an inverted index, and returns:
      - inverted_index: dict mapping term -> set of doc IDs
      - doc_id_map: dict mapping doc ID -> filename
      - doc_distinct_word_counts: dict mapping doc ID -> number of distinct words in the document
    """
    inverted_index = {}
    doc_id_map = {}
    doc_distinct_word_counts = {}

    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    # Process only .html files
    doc_files = [f for f in os.listdir(crawled_dir) if f.endswith('.html')]
    doc_count = 0

    for filename in doc_files:
        file_path = os.path.join(crawled_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            html = f.read()

        text = extract_article_text(html)
        tokens = preprocess_text(text, stop_words, stemmer)
        unique_tokens = set(tokens)

        # Map document id to filename
        doc_id = doc_count
        doc_id_map[doc_id] = filename
        doc_distinct_word_counts[doc_id] = len(unique_tokens)

        # Update inverted index: for each unique token, add the document id to its posting list
        for token in unique_tokens:
            if token not in inverted_index:
                inverted_index[token] = set()
            inverted_index[token].add(doc_id)

        doc_count += 1

    return inverted_index, doc_id_map, doc_distinct_word_counts


######################################
# 4. Functions to Save the Results   #
######################################

def save_inverted_index(inverted_index, output_file):
    """
    Writes the inverted index to a text file.
    Each line: term, document frequency, and comma-separated posting list.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for term, postings in sorted(inverted_index.items()):
            df = len(postings)
            postings_list = ",".join(str(doc_id) for doc_id in sorted(postings))
            f.write(f"{term} {df} {postings_list}\n")


def save_doc_id_map(doc_id_map, output_file):
    """
    Saves the mapping from document id to filename.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc_id, filename in sorted(doc_id_map.items()):
            f.write(f"{doc_id} {filename}\n")


##############################
# 5. Compute Statistics      #
##############################

def compute_statistics(inverted_index, doc_distinct_word_counts, index_file, crawled_dir):
    """
    Computes:
      - Number of distinct terms
      - Number of documents
      - Average number of distinct words per document
      - The size of the inverted index file relative to the total collection size
      - Additional parameter: Maximum posting list length
    """
    num_distinct_terms = len(inverted_index)
    num_documents = len(doc_distinct_word_counts)
    avg_distinct_words = sum(doc_distinct_word_counts.values()) / num_documents if num_documents > 0 else 0

    # Get file sizes
    index_size = os.path.getsize(index_file)
    total_collection_size = 0
    for f in os.listdir(crawled_dir):
        if f.endswith('.html'):
            total_collection_size += os.path.getsize(os.path.join(crawled_dir, f))

    # Gather document frequencies for each term for the histogram
    doc_frequencies = [len(postings) for postings in inverted_index.values()]

    # Additional parameter: Maximum posting list length (i.e., the term that appears in the most documents)
    max_posting_length = max(doc_frequencies) if doc_frequencies else 0

    stats = {
        'num_distinct_terms': num_distinct_terms,
        'num_documents': num_documents,
        'avg_distinct_words': avg_distinct_words,
        'index_size': index_size,
        'collection_size': total_collection_size,
        'max_posting_length': max_posting_length
    }
    return stats, doc_frequencies


##############################
# 6. Plotting the Histogram  #
##############################

def plot_histogram(doc_frequencies, output_image):
    # Optionally filter out zeros
    filtered_frequencies = [freq for freq in doc_frequencies if freq > 0]

    plt.figure(figsize=(10, 6))

    # Use automatic binning; you can adjust or experiment with this parameter
    plt.hist(filtered_frequencies, bins='auto', color='blue', edgecolor='black')

    # Apply logarithmic scales if the data is highly skewed:
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('Document Frequency (Number of documents a term appears in) [log scale]')
    plt.ylabel('Number of Terms [log scale]')
    plt.title('Histogram of Document Frequencies')
    plt.tight_layout()  # Adjust layout to prevent clipping

    plt.savefig(output_image)
    plt.close()


##############################
# 7. Main Function           #
##############################

def main():
    # Define directories and file names
    crawled_dir = "crawled_pages"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    inverted_index_file = os.path.join(output_dir, "inverted_index.txt")
    doc_id_map_file = os.path.join(output_dir, "doc_id_map.txt")
    histogram_image = os.path.join(output_dir, "doc_freq_histogram.png")

    print("Building inverted index...")
    inverted_index, doc_id_map, doc_distinct_word_counts = build_inverted_index(crawled_dir)

    print("Saving inverted index to file...")
    save_inverted_index(inverted_index, inverted_index_file)

    print("Saving document id mapping...")
    save_doc_id_map(doc_id_map, doc_id_map_file)

    print("Computing statistics...")
    stats, doc_frequencies = compute_statistics(inverted_index, doc_distinct_word_counts, inverted_index_file,
                                                crawled_dir)

    print("\nStatistics:")
    print(f"  - Number of distinct terms: {stats['num_distinct_terms']}")
    print(f"  - Number of documents: {stats['num_documents']}")
    print(f"  - Average number of distinct words per document: {stats['avg_distinct_words']:.2f}")
    print(f"  - Inverted index size: {stats['index_size']} bytes")
    print(f"  - Total collection size: {stats['collection_size']} bytes")
    print(f"  - Maximum posting list length (most frequent term appears in): {stats['max_posting_length']}")

    print("Plotting histogram...")
    plot_histogram(doc_frequencies, histogram_image)
    print(f"Histogram saved as {histogram_image}")


if __name__ == "__main__":
    main()
