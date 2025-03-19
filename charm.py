from collections import defaultdict
from itertools import combinations


def charm_algorithm(sequences, min_support, min_confidence):
    # Step 1: Initialize single itemsets
    itemset_counts = defaultdict(int)
    for sequence in sequences:
        for item in sequence:
            itemset_counts[frozenset([item])] += 1

    # Step 2: Filter itemsets based on min_support
    freq_itemsets = {itemset: count for itemset, count in itemset_counts.items() if count >= min_support}

    # Step 3: Recursive exploration to find closed itemsets
    def charm_extend(itemsets, prefix):
        closed_itemsets = {}
        for itemset in itemsets:
            new_prefix = prefix.union(itemset)
            new_support = sum(1 for sequence in sequences if new_prefix.issubset(sequence))
            if new_support >= min_support:
                closed_itemsets[new_prefix] = new_support
                # Recursive call
                new_itemsets = {i for i in itemsets if i != itemset}
                closed_itemsets.update(charm_extend(new_itemsets, new_prefix))
        return closed_itemsets

    closed_itemsets = charm_extend(set(freq_itemsets.keys()), frozenset())

    # Step 4: Generate association rules from closed itemsets
    rules = []
    for itemset in closed_itemsets:
        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                if consequent:
                    antecedent_support = closed_itemsets[antecedent]
                    itemset_support = closed_itemsets[itemset]
                    confidence = itemset_support / antecedent_support
                    if confidence >= min_confidence:
                        rules.append((antecedent, consequent, confidence))

    return closed_itemsets, rules

# Example usage
sequences = [
    ['milk', 'bread', 'butter'],
    ['beer', 'bread'],
    ['milk', 'bread', 'butter', 'beer'],
    ['bread', 'butter']
]
min_support = 2
min_confidence = 0.5

closed_itemsets, rules = charm_algorithm(sequences, min_support, min_confidence)
print("Closed Frequent Itemsets:")
for itemset, count in closed_itemsets.items():
    print(f"{set(itemset)}: {count}")

print("\nAssociation Rules:")
for antecedent, consequent, confidence in rules:
    print(f"Rule: {set(antecedent)} -> {set(consequent)} (Confidence: {confidence:.2f})")