import os
import json

RESULTS_DIR = "retrieval_results"


def summarize_pools():
    print("\n📊 Pool Summary for Each Query:\n")

    for file in os.listdir(RESULTS_DIR):
        if file.startswith("pool_query_") and file.endswith(".json"):
            pool_path = os.path.join(RESULTS_DIR, file)
            with open(pool_path, 'r') as f:
                doc_ids = json.load(f)

            query_number = file.split('_')[-1].split('.')[0]
            unique_count = len(doc_ids)

            print(f"➡️ Query {query_number}: {unique_count} unique documents in the pool")
            print("Top 5 document IDs:")
            for doc_id in doc_ids[:5]:
                print(f"  - {doc_id}")
            print("...\n")


if __name__ == "__main__":
    summarize_pools()
