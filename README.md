# 📚 Information Retrieval and Web Search Project

## Overview
This project implements a complete Information Retrieval (IR) system for processing and searching a collection of documents. It features multiple indexing techniques, advanced query processing, spelling correction, and evaluation of classic IR laws such as Heap’s and Zipf’s Law.

---

## 📂 Project Structure
```
information-retrieval-web-search/
│
├── crawled_pages/                  # HTML document collection.
├── output/                         # Generated indexes, logs, and plots.
├── biword_index.py                 # Builds biword inverted index.
├── boolean.py                      # Boolean retrieval methods.
├── compare_indexes.py              # Compares index sizes and statistics.
├── heaps_law_analysis.py           # Heap’s Law analysis and curve fitting.
├── heaps_law_verification.py       # Heap’s Law data preparation.
├── inverted_index.py               # Builds single-term inverted index.
├── lists.py                        # Helper functions for lists.
├── main.py                         # Main driver script.
├── ngram_jaccard_correction.py     # N-gram Jaccard spelling correction.
├── phrase_query.py                 # Phrase queries using biword index.
├── positional_index.py             # Builds positional index.
├── positional_phrase_query.py      # Phrase queries with positional index.
├── spelling_correction_edit.py     # Edit distance spelling correction.
├── zipf_law_analysis.py            # Zipf’s Law analysis and plotting.
└── medium_data_engineering.json    # Example configuration.
```

---

## 🔑 Features
- Single-term inverted index
- Biword indexing for phrase queries
- Positional index for exact phrase matching
- Boolean retrieval
- Spelling correction (Edit Distance & N-gram Jaccard)
- Heap’s Law verification and analysis
- Zipf’s Law analysis with omitted terms impact
- Index comparison and benchmarking

---

## ⚙️ Technologies
- Python 3
- NLTK (text preprocessing)
- BeautifulSoup (HTML parsing)
- Matplotlib (visualizations)
- NumPy (numerical analysis)

---

## 🚀 How to Run
Make sure your HTML files are in `crawled_pages/`. Outputs will be stored in `output/`.

Example executions:
```bash
python inverted_index.py
python phrase_query.py
python heaps_law_analysis.py
python zipf_law_analysis.py
```

---

## 📊 Purpose
This project was created to explore and apply core concepts of Information Retrieval (IR), including efficient search, indexing, spelling correction, and theoretical validation through real datasets.

---

## 📝 Author
I developed as part of a university-level Information Retrieval course.

---

## 📌 Notes
- For large datasets, ensure sufficient memory (16GB+ recommended).
- Performance and accuracy align with real-world IR expectations.

---

## 📃 License
This project is for educational purposes.

