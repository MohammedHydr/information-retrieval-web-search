import os
import json
import requests
from bs4 import BeautifulSoup
import re

# Elasticsearch setup
ELASTIC_HOST = "https://localhost:9200"
USERNAME = "elastic"
PASSWORD = "xlj6B8WV*imCI94DjrrA"  # Use your real password here!

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text

# Improved article extractor
def extract_article_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()

    article = soup.find('article')
    if article:
        content = article.get_text(separator=' ', strip=True)
    else:
        main = soup.find('main')
        if main:
            content = main.get_text(separator=' ', strip=True)
        else:
            divs = soup.find_all('div')
            largest_div = max(divs, key=lambda d: len(d.get_text()), default=None)
            if largest_div:
                content = largest_div.get_text(separator=' ', strip=True)
            else:
                content = soup.body.get_text(separator=' ', strip=True)

    content = re.sub(r'\s+', ' ', content)
    return content

# Delete index if it exists
def delete_index(index_name):
    response = requests.delete(f"{ELASTIC_HOST}/{index_name}", auth=(USERNAME, PASSWORD), verify=False)
    if response.status_code in [200, 404]:
        print(f"Deleted (or not found): {index_name}")
    else:
        print(f"Error deleting {index_name}: {response.text}")

# Create index with good settings
def create_index(index_name):
    settings_body = {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            }
        },
        "mappings": {
            "properties": {
                "content": {"type": "text"},
                "doc_length": {"type": "integer"}
            }
        }
    }
    response = requests.put(f"{ELASTIC_HOST}/{index_name}", auth=(USERNAME, PASSWORD),
                            headers={"Content-Type": "application/json"},
                            data=json.dumps(settings_body), verify=False)
    print(f"Created index {index_name}: {response.text}")

# Index documents in batches
def index_documents(index_name, crawled_dir):
    headers = {"Content-Type": "application/x-ndjson"}
    doc_files = [f for f in os.listdir(crawled_dir) if f.endswith(".html")]
    batch_size = 500

    for i in range(0, len(doc_files), batch_size):
        bulk_lines = []
        for filename in doc_files[i:i+batch_size]:
            with open(os.path.join(crawled_dir, filename), 'r', encoding='utf-8') as f:
                html = f.read()
            content = extract_article_text(html)
            content = preprocess_text(content)
            doc_length = len(content.split())

            bulk_lines.append(json.dumps({"index": {"_index": index_name}}))
            bulk_lines.append(json.dumps({"content": content, "doc_length": doc_length}))

        bulk_payload = "\n".join(bulk_lines) + "\n"
        response = requests.post(f"{ELASTIC_HOST}/_bulk", auth=(USERNAME, PASSWORD),
                                 headers=headers, data=bulk_payload.encode('utf-8'), verify=False)
        print(f"Indexed batch {i // batch_size + 1} into {index_name} | Response: {response.status_code}")

if __name__ == "__main__":
    os.environ['CURL_CA_BUNDLE'] = ''  # Ignore SSL warnings

    # Delete old indexes first
    delete_index("collection_vector")
    delete_index("collection_bm25")

    # Recreate indexes
    create_index("collection_vector")
    create_index("collection_bm25")

    # Start indexing
    crawled_dir = "../crawled_pages"
    index_documents("collection_vector", crawled_dir)
    index_documents("collection_bm25", crawled_dir)

    print("✅ All indexing completed successfully!")
