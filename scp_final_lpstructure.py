import random


class Rule:
    def __init__(self, premises, conclusion):
        self.premises = premises
        self.conclusion = conclusion


class ExpertSystem:
    def __init__(self):
        self.knowledge_base = []
        self.facts = set()

    def add_rule(self, premises, conclusion):
        self.knowledge_base.append(Rule(premises, conclusion))

    def add_fact(self, fact):
        self.facts.add(fact)

    def backward_inference(self, goal):
        if goal in self.facts:
            return True

        for rule in self.knowledge_base:
            if rule.conclusion == goal and all(self.backward_inference(premise) for premise in rule.premises):
                return True

        return False


class LPStructure:
    def __init__(self, eLengthItems, eItemBitSize):
        self.eLengthItems = eLengthItems
        self.eItemBitSize = eItemBitSize

    def EQ(self, ls, rs):
        return all(ls[i] == rs[i] for i in range(self.eLengthItems))

    def EZ(self, ls):
        return all(not ls[i] for i in range(self.eLengthItems))

    def LE(self, ls, rs):
        return all((ls[i] | rs[i]) == rs[i] for i in range(self.eLengthItems))

    def LT(self, ls, rs):
        return any((ls[i] | rs[i]) == rs[i] and ls[i] != rs[i] for i in range(self.eLengthItems))

    def lJoin(self, ls, rs):
        for i in range(self.eLengthItems):
            ls[i] |= rs[i]

    def lMeet(self, ls, rs):
        for i in range(self.eLengthItems):
            ls[i] &= rs[i]

    def lDiff(self, ls, rs):
        return any(ls[i] & rs[i] and not (ls[i] & ~rs[i]) for i in range(self.eLengthItems))

    def isMeet(self, ls, rs):
        return any(ls[i] & rs[i] for i in range(self.eLengthItems))

    def isON(self, eTest, nAtom):
        nItem = nAtom // self.eItemBitSize
        nBit = nAtom % self.eItemBitSize
        nMask = 1 << (self.eItemBitSize - 1 - nBit)
        return bool(eTest[nItem] & nMask)

    def to_binary_string(self, individual):
        return [bin(item)[2:].zfill(self.eItemBitSize) for item in individual]


class SCPAlgorithm:
    def __init__(self, lp_structure, population_size=100, generations=100):
        self.lp_structure = lp_structure
        self.population_size = population_size
        self.generations = generations

    def generate_individual(self):
        return [random.randint(0, (1 << self.lp_structure.eItemBitSize) - 1) for _ in range(self.lp_structure.eLengthItems)]

    def generate_population(self):
        return [self.generate_individual() for _ in range(self.population_size)]

    def fitness(self, individual):
        return sum(1 for i in range(len(individual) * self.lp_structure.eItemBitSize) if self.lp_structure.isON(individual, i))

    def select(self, population):
        return sorted(population, key=self.fitness, reverse=True)[:int(0.2 * len(population))]

    def crossover(self, parent1, parent2):
        child1, child2 = parent1.copy(), parent2.copy()
        self.lp_structure.lJoin(child1, parent2)
        self.lp_structure.lJoin(child2, parent1)
        return child1, child2

    def mutate(self, individual):
        return [item ^ (1 << random.randint(0, self.lp_structure.eItemBitSize - 1)) if random.random() < 0.01 else item for item in individual]

    def evolve(self):
        population = self.generate_population()
        for _ in range(self.generations):
            selected = self.select(population)
            next_generation = selected.copy()

            while len(next_generation) < self.population_size:
                if len(selected) >= 2:
                    parent1, parent2 = random.sample(selected, 2)
                    next_generation.extend(self.crossover(parent1, parent2))
                else:
                    next_generation.append(self.generate_individual())

            population = [self.mutate(individual) for individual in next_generation]

        return self.select(population)[0]


# Example usage
lp_structure = LPStructure(10, 8)
scp_algorithm = SCPAlgorithm(lp_structure)
best_individual = scp_algorithm.evolve()

# Print the best individual in decimal format
print("Best Individual (decimal):", best_individual)

# Convert and print the best individual in binary format
binary_representation = lp_structure.to_binary_string(best_individual)
print("Best Individual (binary):")
for i, binary in enumerate(binary_representation):
    print(f"Item {i}: {binary}")

# Print a more visual representation
print("\nBinary Matrix Representation:")
for binary in binary_representation:
    print(' '.join(binary))

# Calculate and print the fitness of the best individual
fitness_value = scp_algorithm.fitness(best_individual)
print(f"\nFitness value: {fitness_value} (number of bits set to 1)")

# Expert System Example
expert_system = ExpertSystem()
expert_system.add_fact("A")
expert_system.add_rule(["A"], "B")
expert_system.add_rule(["B"], "C")

goal = "C"
result = expert_system.backward_inference(goal)
print(f"Goal '{goal}' is {'proven' if result else 'not proven'}")