from collections import defaultdict
from itertools import combinations

def sard_algorithm(sequences, min_support, min_confidence):
    # Step 1: Initialize candidate itemsets
    candidate_itemsets = {frozenset([item]): 1 for sequence in sequences for item in sequence}

    # Step 2: Scan the database and count support
    itemset_counts = defaultdict(int)
    for sequence in sequences:
        for itemset in candidate_itemsets:
            if itemset.issubset(sequence):
                itemset_counts[itemset] += 1

    # Step 3: Filter candidate itemsets based on min_support
    freq_itemsets = {itemset: count for itemset, count in itemset_counts.items() if count >= min_support}

    # Step 4: Generate new candidate itemsets
    def generate_candidates(itemsets, length):
        return {i.union(j) for i in itemsets for j in itemsets if len(i.union(j)) == length}

    k = 2
    current_itemsets = freq_itemsets
    while current_itemsets:
        candidate_itemsets = generate_candidates(current_itemsets, k)
        current_itemsets = {itemset: sum(1 for sequence in sequences if itemset.issubset(sequence)) for itemset in candidate_itemsets}
        freq_itemsets.update({itemset: count for itemset, count in current_itemsets.items() if count >= min_support})
        k += 1

    # Step 6: Generate association rules
    rules = []
    for itemset in freq_itemsets:
        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                if consequent:
                    antecedent_support = freq_itemsets[antecedent]
                    itemset_support = freq_itemsets[itemset]
                    confidence = itemset_support / antecedent_support
                    if confidence >= min_confidence:
                        rules.append((antecedent, consequent, confidence))

    return freq_itemsets, rules

# Example usage
sequences = [
    ['A', 'B', 'C'],
    ['A', 'B', 'D'],
    ['B', 'C', 'D'],
    ['A', 'C', 'D'],
    ['A', 'B', 'C', 'D']
]
min_support = 2
min_confidence = 0.5

freq_itemsets, rules = sard_algorithm(sequences, min_support, min_confidence)
print("Frequent Itemsets:")
for itemset, count in freq_itemsets.items():
    print(f"{set(itemset)}: {count}")

print("\nAssociation Rules:")
for antecedent, consequent, confidence in rules:
    print(f"Rule: {set(antecedent)} -> {set(consequent)} (Confidence: {confidence:.2f})")