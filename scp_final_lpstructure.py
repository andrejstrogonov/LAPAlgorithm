
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
        rule = Rule(premises, conclusion)
        self.knowledge_base.append(rule)

    def add_fact(self, fact):
        self.facts.add(fact)

    def backward_inference(self, goal):
        # Example usage of SCPAlgorithm
        lp_structure = LPStructure(10, 8)
        scp_algorithm = SCPAlgorithm(lp_structure)
        best_individual = scp_algorithm.evolve()

        if goal in self.facts:
            return True

        for rule in self.knowledge_base:
            if rule.conclusion == goal:
                all_premises_true = True
                for premise in rule.premises:
                    if not self.backward_inference(premise):
                        all_premises_true = False
                        break
                if all_premises_true:
                    return True

        return False

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

    def to_binary_string(self, individual):
        binary_representation = []
        for item in individual:
            binary = bin(item)[2:]
            padded_binary = binary.zfill(self.eItemBitSize)
            binary_representation.append(padded_binary)
        return binary_representation


class SCPAlgorithm:
    def __init__(self, lp_structure, population_size=100, generations=100):
        self.lp_structure = lp_structure
        self.population_size = population_size
        self.generations = generations

    def generate_individual(self):
        individual = []
        for _ in range(self.lp_structure.eLengthItems):
            item = random.randint(0, (1 << self.lp_structure.eItemBitSize) - 1)
            individual.append(item)
        return individual

    def generate_population(self):
        return [self.generate_individual() for _ in range(self.population_size)]

    def fitness(self, individual):
        score = 0
        for i in range(len(individual) * self.lp_structure.eItemBitSize):
            if self.lp_structure.isON(individual, i):
                score += 1
        return score

    def select(self, population):
        scored = [(self.fitness(individual), individual) for individual in population]
        scored.sort(reverse=True)
        ranked = [individual for (score, individual) in scored]
        return ranked[:int(0.2 * len(ranked))]

    def crossover(self, parent1, parent2):
        child1 = parent1.copy()
        child2 = parent2.copy()
        self.lp_structure.lJoin(child1, parent2)
        self.lp_structure.lJoin(child2, parent1)
        return child1, child2

    def mutate(self, individual):
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < 0.01:
                bit_position = random.randint(0, self.lp_structure.eItemBitSize - 1)
                mutated[i] ^= (1 << bit_position)
        return mutated

    def evolve(self):
        population = self.generate_population()
        for _ in range(self.generations):
            selected = self.select(population)
            next_generation = selected.copy()

            while len(next_generation) < self.population_size:
                if len(selected) >= 2:
                    parent1, parent2 = random.sample(selected, 2)
                    child1, child2 = self.crossover(parent1, parent2)
                    next_generation.append(child1)
                    if len(next_generation) < self.population_size:
                        next_generation.append(child2)
                else:
                    next_generation.append(self.generate_individual())

            population = [self.mutate(individual) for individual in next_generation]

        best_individual = self.select(population)[0]
        return best_individual


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