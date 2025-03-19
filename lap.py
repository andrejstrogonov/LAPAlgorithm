from itertools import combinations
from collections import defaultdict

def lap_algorithm(sequences, min_support, min_confidence):
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

    # Step 4: Linear approximation for larger itemsets
    def linear_approximation(itemsets, length):
        approximated_itemsets = {}
        for itemset in itemsets:
            for other_itemset in itemsets:
                if len(itemset.union(other_itemset)) == length:
                    combined = itemset.union(other_itemset)
                    approximated_support = min(itemsets[itemset], itemsets[other_itemset])
                    approximated_itemsets[combined] = approximated_support
        return approximated_itemsets

    k = 2
    current_itemsets = freq_itemsets
    while current_itemsets:
        approximated_itemsets = linear_approximation(current_itemsets, k)
        current_itemsets = {itemset: sum(1 for sequence in sequences if itemset.issubset(sequence)) for itemset in approximated_itemsets}
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
    ['milk', 'bread', 'butter'],
    ['beer', 'bread'],
    ['milk', 'bread', 'butter', 'beer'],
    ['bread', 'butter']
]
min_support = 2
min_confidence = 0.5

freq_itemsets, rules = lap_algorithm(sequences, min_support, min_confidence)
print("Frequent Itemsets:")
for itemset, count in freq_itemsets.items():
    print(f"{set(itemset)}: {count}")

print("\nAssociation Rules:")
for antecedent, consequent, confidence in rules:
    print(f"Rule: {set(antecedent)} -> {set(consequent)} (Confidence: {confidence:.2f})")