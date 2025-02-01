from collections import defaultdict
from itertools import combinations

def apriori(transactions, min_support, min_confidence):
    """Apriori algorithm for frequent itemset generation and rule mining."""
    itemset_counts = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            itemset_counts[frozenset([item])] += 1

    # Filter itemsets by min_support
    freq_itemsets = {itemset for itemset, count in itemset_counts.items() if count >= min_support}

    def get_support(itemset):
        return sum(1 for transaction in transactions if itemset.issubset(transaction))

    def generate_candidates(itemsets, length):
        return {i.union(j) for i in itemsets for j in itemsets if len(i.union(j)) == length}

    k = 2
    current_itemsets = freq_itemsets
    while current_itemsets:
        candidate_itemsets = generate_candidates(current_itemsets, k)
        current_itemsets = {itemset for itemset in candidate_itemsets if get_support(itemset) >= min_support}
        freq_itemsets.update(current_itemsets)
        k += 1

    rules = []
    for itemset in freq_itemsets:
        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                if consequent:
                    antecedent_support = get_support(antecedent)
                    itemset_support = get_support(itemset)
                    confidence = itemset_support / antecedent_support
                    if confidence >= min_confidence:
                        rules.append((antecedent, consequent, confidence))

    return freq_itemsets, rules

def main():
    transactions = [
        frozenset(['milk', 'bread', 'butter']),
        frozenset(['bread', 'diaper', 'beer', 'milk']),
        frozenset(['milk', 'bread', 'diaper']),
        frozenset(['diaper', 'milk', 'bread']),
        frozenset(['bread', 'diaper', 'milk', 'beer'])
    ]

    min_support = 2
    min_confidence = 0.5

    freq_itemsets, rules = apriori(transactions, min_support, min_confidence)
    print("Frequent Itemsets:")
    for itemset in freq_itemsets:
        print(itemset)

    print("\nAssociation Rules:")
    for rule in rules:
        print(f"Rule: {set(rule[0])} -> {set(rule[1])} (Confidence: {rule[2]:.2f})")

if __name__ == "__main__":
    main()