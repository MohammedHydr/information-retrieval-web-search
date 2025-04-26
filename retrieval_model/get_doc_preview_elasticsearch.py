import requests
import json
import os


ELASTIC_HOST = "https://localhost:9200"
USERNAME = "elastic"
PASSWORD = "xlj6B8WV*imCI94DjrrA"

RESULTS_DIR = "../retrieval_results_dismax"  # Location of dismax results
OUTPUT_DIR = "../top10_dismax_previews"  # Previews saved here
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_doc_preview(doc_id, index_name="collection_bm25"):
    # Tries both bm25 and vector index if needed
    for idx_name in ["collection_bm25", "collection_vector"]:
        url = f"{ELASTIC_HOST}/{idx_name}/_doc/{doc_id}"
        response = requests.get(url, auth=(USERNAME, PASSWORD), verify=False)
        doc = response.json()
        content = doc.get("_source", {}).get("content", "")
        if content.strip():
            preview = content[:700].replace('\n', ' ') + "..."
            return preview
    return "No content found."


def fetch_previews_for_dismax_files():
    json_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")]

    for file_name in json_files:
        query_id = file_name.split("_")[2].replace(".json", "")
        model_name = file_name.split("_")[-1].replace(".json", "")

        output_file = os.path.join(OUTPUT_DIR, f"{file_name.replace('.json', '')}_previews.txt")

        with open(os.path.join(RESULTS_DIR, file_name), 'r') as f:
            doc_ids = json.load(f)

        with open(output_file, 'w', encoding='utf-8') as out:
            out.write(f"Previews for Query {query_id} - Model {model_name}\n\n")
            for doc_id in doc_ids:
                preview = get_doc_preview(doc_id)
                out.write(f"Document ID: {doc_id}\n")
                out.write(f"Preview: {preview}\n")
                out.write("\n" + "=" * 80 + "\n\n")

        print(f"✅ Previews saved in {output_file}")


if __name__ == "__main__":
    fetch_previews_for_dismax_files()
    print("✅ All previews from dismax results generated!")
