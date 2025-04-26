import json
import os

RESULTS_DIR = "retrieval_results"


# Assuming each pooled file is the union of 3 models
# And each file is named pool_query_X.json
def load_results(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def generate_overlap_report():
    report_lines = []

    for query_num in range(1, 6):  # Assuming 5 queries
        pool_file = os.path.join(RESULTS_DIR, f"pool_query_{query_num}.json")
        pool_docs = load_results(pool_file)

        report_lines.append(f"\nQuery {query_num} Pool Summary:")
        report_lines.append(f"Total pooled documents: {len(pool_docs)}")

        # (Optional advanced) If you store individual model results separately, you could compare
        # If not, just note overlap between models requires storing separate hits.
        # For now, just show total pool length (should be around 20)

    with open(os.path.join(RESULTS_DIR, "overlap_report.txt"), 'w') as out_file:
        out_file.write("\n".join(report_lines))

    print("✅ Overlap report generated at retrieval_results/overlap_report.txt")


if __name__ == "__main__":
    generate_overlap_report()
