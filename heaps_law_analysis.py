import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import math

from heaps_law_verification import load_collection, compute_heap_data, main


def fit_heaps_law(token_counts, unique_term_counts):
    """ Fit a linear regression to log-log data to find K and b """
    log_tokens = np.log10(token_counts).reshape(-1, 1)
    log_terms = np.log10(unique_term_counts)

    model = LinearRegression()
    model.fit(log_tokens, log_terms)

    b = model.coef_[0]
    log_K = model.intercept_
    K = 10 ** log_K

    print(f"\nHeap's Law Fit Results:")
    print(f"  K = {K:.4f}")
    print(f"  b = {b:.4f}")

    return K, b, model


def calculate_required_tokens(K, b, total_vocab, percentage):
    """ Calculate number of tokens required for a percentage of the vocabulary """
    target_vocab = total_vocab * percentage
    required_tokens = (target_vocab / K) ** (1 / b)
    return required_tokens


def plot_with_fit(token_counts, unique_term_counts, model, output_file):
    """ Plot the Heap's Law data with the best-fit line """
    log_tokens = np.log10(token_counts)
    log_terms = np.log10(unique_term_counts)

    plt.figure(figsize=(10, 6))
    plt.plot(log_tokens, log_terms, 'o', label="Heap's Law Data")

    # Best-fit line
    fit_line = model.predict(log_tokens.reshape(-1, 1))
    plt.plot(log_tokens, fit_line, 'r-', label="Best-fit line")

    plt.xlabel('Log(Number of Tokens)')
    plt.ylabel('Log(Number of Unique Terms)')
    plt.title("Heap's Law Verification with Best-fit Line")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()
    print(f"Heap's Law plot with best-fit line saved as {output_file}")


def step_2(token_counts, unique_term_counts):
    K, b, model = fit_heaps_law(token_counts, unique_term_counts)

    total_vocab = unique_term_counts[-1]
    total_tokens = token_counts[-1]

    required_tokens = calculate_required_tokens(K, b, total_vocab, 0.3)
    actual_30_percent_tokens = total_tokens * 0.3

    print(f"\nTotal vocabulary size: {total_vocab}")
    print(f"Total tokens: {total_tokens}")
    print(f"Tokens required for 30% vocabulary (Heap's Law): {required_tokens:.0f}")
    print(f"Actual 30% of tokens in collection: {actual_30_percent_tokens:.0f}")

    plot_with_fit(token_counts, unique_term_counts, model, "output/heaps_law_fit.png")


# 🚀 Run this after your main Heap's Law steps:

if __name__ == "__main__":
    main()
    # Re-load data
    crawled_dir = "crawled_pages"
    tokens = load_collection(crawled_dir)
    token_counts, unique_term_counts = compute_heap_data(tokens)
    step_2(token_counts, unique_term_counts)
