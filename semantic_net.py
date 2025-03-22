from collections import defaultdict
from itertools import combinations
import random

class LPStructure:
    def __init__(self, eLengthItems, eItemBitSize):
        self.eLengthItems = eLengthItems
        self.eItemBitSize = eItemBitSize

    def EQ(self, ls, rs):
        for i in range(self.eLengthItems):
            if ls[i] != rs[i]:
                return False
        return True

    def EZ(self, ls):
        for i in range(self.eLengthItems):
            if ls[i]:
                return False
        return True

    def LE(self, ls, rs):
        for i in range(self.eLengthItems):
            if (ls[i] | rs[i]) != rs[i]:
                return False
        return True

    def LT(self, ls, rs):
        bExistLT = False
        for i in range(self.eLengthItems):
            if (ls[i] | rs[i]) == rs[i]:
                if ls[i] != rs[i]:
                    bExistLT = True
            else:
                return False
        return bExistLT

    def lJoin(self, ls, rs):
        for i in range(self.eLengthItems):
            ls[i] |= rs[i]

    def lMeet(self, ls, rs):
        for i in range(self.eLengthItems):
            ls[i] &= rs[i]

    def lDiff(self, ls, rs):
        res = False
        for i in range(self.eLengthItems):
            if ls[i] & rs[i]:
                ls[i] &= ~rs[i]
                res = True
        return res

    def isMeet(self, ls, rs):
        for i in range(self.eLengthItems):
            if ls[i] & rs[i]:
                return True
        return False

    def isON(self, eTest, nAtom):
        nItem = nAtom // self.eItemBitSize
        nBit = nAtom % self.eItemBitSize
        nMask = 1 << (self.eItemBitSize - 1 - nBit)
        return bool(eTest[nItem] & nMask)


class SCPAlgorithm:
    def __init__(self, lp_structure, population_size=100, generations=100):
        self.lp_structure = lp_structure
        self.population_size = population_size
        self.generations = generations

    # ... (keep the existing SCPAlgorithm methods as is)

    def run(self, sequences, min_support, min_confidence):
        best_individual = self.evolve()
        freq_itemsets = self.get_frequent_itemsets(best_individual, sequences, min_support)
        rules = self.generate_rules(freq_itemsets, sequences, min_confidence)
        return freq_itemsets, rules

    def get_frequent_itemsets(self, individual, sequences, min_support):
        freq_itemsets = defaultdict(int)
        for sequence in sequences:
            for i, item in enumerate(sequence):
                if self.lp_structure.is_on(individual, i):
                    freq_itemsets[frozenset([item])] += 1
        return {itemset: count for itemset, count in freq_itemsets.items() if count >= min_support}

    def generate_rules(self, freq_itemsets, sequences, min_confidence):
        rules = []
        for itemset in freq_itemsets:
            if len(itemset) > 1:
                for i in range(1, len(itemset)):
                    for antecedent in combinations(itemset, i):
                        antecedent = frozenset(antecedent)
                        consequent = itemset - antecedent
                        confidence = freq_itemsets[itemset] / freq_itemsets[antecedent]
                        if confidence >= min_confidence:
                            rules.append((antecedent, consequent, confidence))
        return rules

def scp_algorithm(sequences, min_support, min_confidence):
    unique_items = set(item for sequence in sequences for item in sequence)
    lp_structure = LPStructure(len(unique_items), 1)  # Assuming 1 bit per item for simplicity
    scp = SCPAlgorithm(lp_structure)
    return scp.run(sequences, min_support, min_confidence)

def inverse_inference(rules, target_attribute):
    inverse_rules = []
    for antecedent, consequent, confidence in rules:
        if any(item.startswith(target_attribute) for item in consequent):
            inverse_rules.append((antecedent, consequent, confidence))
    return inverse_rules

def optimize_decision_process(sequences, min_support, min_confidence, target_attributes):
    # Step 1: Use SCP algorithm to identify key factors and patterns
    freq_itemsets, rules = scp_algorithm(sequences, min_support, min_confidence)

    # Step 2: Use relevant inverse inference to check factors against objectives and constraints
    relevant_rules = []
    for target in target_attributes:
        relevant_rules.extend(inverse_inference(rules, target))

    # Step 3: Optimize the decision-making process based on findings and patterns
    decision_process = {}
    for antecedent, consequent, confidence in relevant_rules:
        decision = list(consequent)[0].split('=')[1]
        conditions = [item.split('=') for item in antecedent]
        decision_process[frozenset(conditions)] = (decision, confidence)

    return decision_process

def make_decision(optimized_process, input_data):
    for conditions, (decision, confidence) in optimized_process.items():
        if all(attr in input_data and input_data[attr] == val for attr, val in conditions):
            return decision, confidence
    return "Unknown", 0.0

# Example usage
if __name__ == "__main__":
    sequences = [
        ['относительный_вес=нормальный', 'зн=да'],
        ['относительный_вес=недостаточный', 'зн=да'],
        ['коронарный_риск=ниже_среднего', 'сердзаб=низкий'],
        # ... (rest of the sequences)
    ]

    min_support = 2
    min_confidence = 0.5
    target_attributes = ['продолжительность', 'перспектива', 'риск']

    optimized_decision_process = optimize_decision_process(sequences, min_support, min_confidence, target_attributes)

    print("Optimized Decision Process:")
    for conditions, (decision, confidence) in optimized_decision_process.items():
        print(f"If {', '.join([f'{attr}={val}' for attr, val in conditions])}")
        print(f"Then {decision} (Confidence: {confidence:.2f})")
        print()

    # Example decision-making
    input_data = {
        'основная_продолжительность': '72',
        'фактор': 'плюс_12',
        'перспектива': 'отличная',
        'риск': 'неизвестен',
        'дополн': 'хорошо'
    }

    decision, confidence = make_decision(optimized_decision_process, input_data)
    print(f"Decision for input data: {decision} (Confidence: {confidence:.2f})")
# Add more insertions as needed