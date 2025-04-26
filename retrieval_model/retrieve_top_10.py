import requests
import json
import os

ELASTIC_HOST = "https://localhost:9200"
USERNAME = "elastic"
PASSWORD = "xlj6B8WV*imCI94DjrrA"

QUERIES_FILE = "../output/query_benchmark.txt"
RESULTS_DIR = "../retrieval_results_dismax"
os.makedirs(RESULTS_DIR, exist_ok=True)

def search_elasticsearch(index_name, query_body):
    response = requests.get(
        f"{ELASTIC_HOST}/{index_name}/_search",
        auth=(USERNAME, PASSWORD),
        headers={"Content-Type": "application/json"},
        data=json.dumps(query_body),
        verify=False
    )
    return response.json()

def retrieve_top10_dismax():
    with open(QUERIES_FILE, 'r') as f:
        queries = [line.strip() for line in f.readlines() if line.strip()]

    for idx, query_text in enumerate(queries, start=1):
        print(f"\n🔎 Retrieving results for Query {idx}: {query_text}")

        # BM25
        bm25_query = {"query": {"match": {"content": query_text}}, "size": 10}

        # Vector (regular boosted match)
        vector_query = {"query": {"match": {"content": {"query": query_text, "boost": 1.2}}}, "size": 10}

        # Dirichlet simulation using dis_max to combine strict and fuzzy matches
        dismax_query = {
            "query": {
                "dis_max": {
                    "queries": [
                        {"match": {"content": {"query": query_text, "boost": 1.0}}},
                        {"match": {"content": {"query": query_text, "fuzziness": "AUTO", "boost": 0.7}}}
                    ],
                    "tie_breaker": 0.3
                }
            },
            "size": 10
        }

        bm25_res = search_elasticsearch("collection_bm25", bm25_query)
        vector_res = search_elasticsearch("collection_vector", vector_query)
        dirichlet_res = search_elasticsearch("collection_vector", dismax_query)

        bm25_docs = [hit['_id'] for hit in bm25_res.get("hits", {}).get("hits", []) if hit['_source'].get("content")]
        vector_docs = [hit['_id'] for hit in vector_res.get("hits", {}).get("hits", []) if hit['_source'].get("content")]
        dirichlet_docs = [hit['_id'] for hit in dirichlet_res.get("hits", {}).get("hits", []) if hit['_source'].get("content")]

        with open(os.path.join(RESULTS_DIR, f"top10_query_{idx}_bm25.json"), 'w') as f:
            json.dump(bm25_docs, f, indent=2)

        with open(os.path.join(RESULTS_DIR, f"top10_query_{idx}_vector.json"), 'w') as f:
            json.dump(vector_docs, f, indent=2)

        with open(os.path.join(RESULTS_DIR, f"top10_query_{idx}_dirichlet.json"), 'w') as f:
            json.dump(dirichlet_docs, f, indent=2)

        print(f"BM25: {len(bm25_docs)} | VECTOR: {len(vector_docs)} | DIRICHLET (dis_max): {len(dirichlet_docs)}")

if __name__ == "__main__":
    os.environ['CURL_CA_BUNDLE'] = ''
    retrieve_top10_dismax()
    print("✅ Distinct retrieval with dis_max complete!")
