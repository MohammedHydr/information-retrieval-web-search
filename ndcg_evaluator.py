import math

def dcg_at_k(relevance_grades, k=5):
    """Compute Discounted Cumulative Gain (DCG) at rank k"""
    return sum(
        (2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevance_grades[:k])
    )

def ndcg_at_k(relevance_grades, k=5):
    """Compute Normalized DCG (NDCG) at rank k"""
    dcg = dcg_at_k(relevance_grades, k)
    ideal = sorted(relevance_grades, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0

##############################
# Example: Simulated Results
##############################
queries = {
    "Real Madrid Champions League victories":         [3, 3, 1, 0, 0],
    "La Liga top scorers 2023":                       [3, 1, 0, 0, 0],
    "Upcoming Real Madrid transfer rumors":           [1, 1, 1, 0, 0],
    "UEFA match results for Real Madrid":             [3, 1, 0, 0, 0],
    "Historical performance Real Madrid 2010-2020":   [3, 3, 3, 3, 3],
}

print("Query-wise NDCG@5 Scores:\n")
total_ndcg = 0

for query, grades in queries.items():
    score = ndcg_at_k(grades, 5)
    total_ndcg += score
    print(f"{query}\n  -> NDCG@5: {score:.3f}\n")

average = total_ndcg / len(queries)
print(f"Average NDCG@5: {average:.3f}")
