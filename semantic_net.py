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
        self.population = self.initialize_population()

    def initialize_population(self):
        # Initialize a random population
        return [self.random_individual() for _ in range(self.population_size)]

    def random_individual(self):
        # Create a random individual
        return [random.randint(0, 1) for _ in range(self.lp_structure.eLengthItems)]

    def run(self, sequences, min_support, min_confidence):
        # Initialize and evolve the population
        best_individual = self.evolve()

        # Placeholder for generating frequent itemsets and rules
        # This should be replaced with actual logic to generate itemsets and rules
        freq_itemsets = []  # Example: list of frequent itemsets
        rules = []  # Example: list of rules with confidence

        # Return the best individual as a placeholder for the actual results
        return freq_itemsets, rules

    def evolve(self):
        for generation in range(self.generations):
            # Evaluate fitness of each individual
            fitness_scores = [self.fitness(individual) for individual in self.population]

            # Select individuals for the next generation
            selected_individuals = self.selection(fitness_scores)

            # Create the next generation through crossover and mutation
            next_generation = self.crossover_and_mutate(selected_individuals)

            # Update the population
            self.population = next_generation

        # Return the best individual from the final generation
        best_individual = max(self.population, key=self.fitness)
        return best_individual

    def fitness(self, individual):
        # Define a fitness function for individuals
        return sum(individual)  # Example: maximize the number of '1's

    def selection(self, fitness_scores):
        # Select individuals based on fitness scores
        selected = random.choices(self.population, weights=fitness_scores, k=self.population_size)
        return selected

    def crossover_and_mutate(self, selected_individuals):
        # Perform crossover and mutation to create the next generation
        next_generation = []
        for i in range(0, len(selected_individuals), 2):
            parent1 = selected_individuals[i]
            parent2 = selected_individuals[i + 1] if i + 1 < len(selected_individuals) else selected_individuals[0]
            child1, child2 = self.crossover(parent1, parent2)
            next_generation.extend([self.mutate(child1), self.mutate(child2)])
        return next_generation

    def crossover(self, parent1, parent2):
        # Perform crossover between two parents
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def mutate(self, individual):
        # Mutate an individual
        mutation_rate = 0.01
        return [gene if random.random() > mutation_rate else 1 - gene for gene in individual]

    def fitness(self, individual):
        # Define a fitness function for individuals
        return sum(individual)  # Example: maximize the number of '1's

    def selection(self, fitness_scores):
        # Select individuals based on fitness scores
        selected = random.choices(self.population, weights=fitness_scores, k=self.population_size)
        return selected

    def crossover_and_mutate(self, selected_individuals):
        # Perform crossover and mutation to create the next generation
        next_generation = []
        for i in range(0, len(selected_individuals), 2):
            parent1 = selected_individuals[i]
            parent2 = selected_individuals[i + 1] if i + 1 < len(selected_individuals) else selected_individuals[0]
            child1, child2 = self.crossover(parent1, parent2)
            next_generation.extend([self.mutate(child1), self.mutate(child2)])
        return next_generation

    def crossover(self, parent1, parent2):
        # Perform crossover between two parents
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def mutate(self, individual):
        # Mutate an individual
        mutation_rate = 0.01
        return [gene if random.random() > mutation_rate else 1 - gene for gene in individual]

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
        ['коронарный_риск=средний', 'сердзаб=низкий'],
        ['старт=да', 'сердзаб=иной'],
        ['возраст=25_или_меньше', 'пол=ж', 'основная_продолжительность=72'],
        ['возраст=25_или_меньше', 'пол=м', 'основная_продолжительность=67'],
        ['возраст=25-55', 'пол=ж', 'основная_продолжительность=67'],
        ['возраст=25-55', 'пол=м', 'основная_продолжительность=62'],
        ['возраст=55_или_больше', 'пол=ж', 'основная_продолжительность=64'],
        ['возраст=55_или_больше', 'пол=м', 'основная_продолжительность=60'],
        ['вес=55_или_меньше', 'сложение=мелкое', 'пол=ж', 'относительный_вес=нормальный'],
        ['вес=85_или_больше', 'сложение=мелкое', 'пол=ж', 'относительный_вес=излишний'],
        ['вес=55-85', 'сложение=мелкое', 'пол=ж', 'относительный_вес=излишний'],
        ['вес=55_или_меньше', 'сложение=крупное', 'относительный_вес=недостаточный'],
        ['вес=55_или_меньше', 'сложение=крупное', 'пол=м', 'относительный_вес=недостаточный'],
        ['вес=55-85', 'сложение=мелкое', 'пол=ж', 'относительный_вес=излишний'],
        ['вес=55-85', 'сложение=крупное', 'пол=ж', 'относительный_вес=нормальный'],
        ['вес=55-85', 'сложение=мелкое', 'пол=м', 'относительный_вес=нормальный'],
        ['вес=55-85', 'сложение=крупное', 'пол=м', 'относительный_вес=недостаточный'],
        ['вес=85_или_больше', 'пол=ж', 'относительный_вес=излишний'],
        ['вес=85_или_больше', 'пол=м', 'сложение=мелкое', 'относительный_вес=излишний'],
        ['вес=85_или_больше', 'пол=м', 'сложение=крупное', 'относительный_вес=нормальный'],
        ['холестерин=низкий', 'жиры=излишек', 'коронарный_риск=ниже_среднего'],
        ['холестерин=низкий', 'жиры=норма', 'коронарный_риск=средний'],
        ['холестерин=излишек', 'коронарный_риск=выше_среднего'],
        ['холестерин=норма', 'коронарный_риск=средний'],
        ['соль=излишек', 'кровяное_давление=выше_среднего'],
        ['соль=норма', 'кровяное_давление=среднее'],
        ['кальций=излишек', 'риск_остеопороза=ниже_среднего'],
        ['кальций=норма', 'риск_остеопороза=средний'],
        ['кальций=низкий', 'риск_остеопороза=выше_среднего'],
        ['относительный_вес=излишний', 'коронарный_риск=выше_среднего', 'кровяное_давление=выше_среднего', 'курение=да',
         'перспектива=унылая'],
        ['относительный_вес=излишний', 'коронарный_риск=выше_среднего', 'кровяное_давление=выше_среднего',
         'курение=нет', 'перспектива=плохая'],
        ['относительный_вес=излишний', 'коронарный_риск=выше_среднего', 'кровяное_давление=среднее', 'курение=нет',
         'перспектива=хорошая'],
        ['относительный_вес=излишний', 'сердзаб=низкий', 'кровяное_давление=среднее', 'курение=нет',
         'перспектива=хорошая'],
        ['зн=да', 'сердзаб=низкий', 'кровяное_давление=среднее', 'курение=нет', 'перспектива=отличная'],
        ['зн=да', 'коронарный_риск=выше_среднего', 'кровяное_давление=среднее', 'курение=да',
         'перспектива=посредственная'],
        ['зн=да', 'сердзаб=низкий', 'кровяное_давление=среднее', 'курение=да', 'перспектива=посредственная'],
        ['зн=да', 'сердзаб=низкий', 'кровяное_давление=среднее', 'курение=нет', 'перспектива=хорошая'],
        ['старт=да', 'перспектива=неизвестна'],
        ['раса=негроидная', 'происхождение=средиземноморское', 'риск=высокий'],
        ['характер=агрессивный', 'тип_личности=тип_a'],
        ['характер=мягкий', 'тип_личности=тип_b'],
        ['тип_личности=тип_a', 'риск=высокий'],
        ['старт=да', 'риск=неизвестен'],
        ['потребление_алкоголя=умеренное', 'дополн=посредственно'],
        ['потребление_алкоголя=не_употребляет', 'дополн=хорошо'],
        ['потребление_алкоголя=чрезмерное', 'дополн=плохо'],
        ['перспектива=унылая', 'риск=высокий', 'дополн=плохо', 'фактор=минус_12'],
        ['перспектива=отличная', 'риск=неизвестен', 'дополн=посредственно', 'фактор=плюс_12'],
        ['перспектива=отличная', 'риск=высокий', 'дополн=посредственно', 'фактор=ноль'],
        ['перспектива=хорошая', 'риск=высокий', 'дополн=посредственно', 'фактор=ноль'],
        ['перспектива=хорошая', 'риск=высокий', 'фактор=минус_12'],
        ['перспектива=посредственная', 'риск=высокий', 'фактор=минус_12'],
        ['перспектива=отличная', 'риск=неизвестен', 'дополн=хорошо', 'фактор=плюс_12'],
        ['перспектива=отличная', 'риск=неизвестен', 'дополн=плохо', 'фактор=плюс_12'],
        ['перспектива=посредственная', 'риск=неизвестен', 'фактор=плюс_12'],
        ['перспектива=хорошая', 'риск=неизвестен', 'фактор=ноль'],
        ['перспектива=унылая', 'риск=неизвестен', 'фактор=ноль'],
        ['старт=да', 'фактор=неизвестен'],
        ['старт=выход', 'фактор=выход'],
        ['основная_продолжительность=72', 'фактор=ноль', 'продолжительность=72'],
        ['основная_продолжительность=67', 'фактор=ноль', 'продолжительность=67'],
        ['основная_продолжительность=64', 'фактор=ноль', 'продолжительность=64'],
        ['основная_продолжительность=62', 'фактор=ноль', 'продолжительность=62'],
        ['основная_продолжительность=60', 'фактор=ноль', 'продолжительность=60'],
        ['основная_продолжительность=72', 'фактор=плюс_12', 'продолжительность=84'],
        ['основная_продолжительность=67', 'фактор=плюс_12', 'продолжительность=79'],
        ['основная_продолжительность=64', 'фактор=плюс_12', 'продолжительность=76'],
        ['основная_продолжительность=62', 'фактор=плюс_12', 'продолжительность=74'],
        ['основная_продолжительность=60', 'фактор=плюс_12', 'продолжительность=72'],
        ['основная_продолжительность=72', 'фактор=минус_12', 'продолжительность=60'],
        ['основная_продолжительность=67', 'фактор=минус_12', 'продолжительность=55'],
        ['основная_продолжительность=64', 'фактор=минус_12', 'продолжительность=52'],
        ['основная_продолжительность=62', 'фактор=минус_12', 'продолжительность=50'],
        ['основная_продолжительность=60', 'фактор=минус_12', 'продолжительность=48']
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