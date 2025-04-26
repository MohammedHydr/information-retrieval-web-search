import pandas as pd
import math
import os
import glob

# Load relevance assessments
relevance_df = pd.read_csv("../relevant_assesment_query.csv")

# Ask user for retrieval model input
model_choice = input("Enter the model (bm25 / vector / dirichlet): ").lower()
model_map = {
    "bm25": "bm25",
    "vector": "vector",
    "dirichlet": "dirichlet"
}

if model_choice not in model_map:
    print("Invalid model name!")
    exit()

# Get list of result files
files = glob.glob(f"top10_dismax_previews/top10_query_*_{model_map[model_choice]}_previews.txt")

# Function to compute DCG

def compute_dcg(relevances):
    return sum([(2 ** rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(relevances)])

# Store results
results = []

for file in files:
    query_num = int(file.split("_query_")[1].split("_")[0])
    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        doc_ids = [line.strip() for line in f.readlines() if line.strip()]

    relevances = []
    for doc_id in doc_ids:
        row = relevance_df[(relevance_df['query_number'] == query_num) & (relevance_df['document_id'] == doc_id)]
        if not row.empty:
            rel = row.iloc[0]['relevance_score']
            # As per exercise: highly relevant(3), somewhat(1), irrelevant(0)
            rel = 3 if rel == 3 else (1 if rel == 1 else (2 if rel == 2 else 0))
        else:
            rel = 0
        relevances.append(rel)

    # Metrics calculation
    precision_at_10 = sum(r > 0 for r in relevances) / 10
    average_precision = 0
    relevant_count = 0
    for i, rel in enumerate(relevances):
        if rel > 0:
            relevant_count += 1
            average_precision += relevant_count / (i + 1)
    average_precision = average_precision / relevant_count if relevant_count > 0 else 0

    reciprocal_rank = 0
    for i, rel in enumerate(relevances):
        if rel > 0:
            reciprocal_rank = 1 / (i + 1)
            break

    ideal_sorted = sorted(relevances, reverse=True)
    dcg = compute_dcg(relevances)
    idcg = compute_dcg(ideal_sorted)
    ndcg = dcg / idcg if idcg > 0 else 0

    results.append({
        'Query': query_num,
        'NDCG': ndcg,
        'Precision@10': precision_at_10,
        'AP': average_precision,
        'MRR': reciprocal_rank
    })

# Create DataFrame
metrics_df = pd.DataFrame(results)

# Add mean row
avg_row = metrics_df.mean()
avg_row['Query'] = 'Mean'
metrics_df = pd.concat([metrics_df, pd.DataFrame([avg_row])], ignore_index=True)

# Display results
print("\nMetrics Table:")
print(metrics_df)

# Save to CSV
output_file = f"metrics_summary_{model_map[model_choice]}.csv"
metrics_df.to_csv(output_file, index=False)
print(f"\n✅ Metrics saved to {output_file}")
