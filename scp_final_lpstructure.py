
import random  # Import the entire random module instead of just the random function


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

    def generate_individual(self):
        # Generate an individual as a list of integers
        individual = []
        for _ in range(self.lp_structure.eLengthItems):
            # Generate a random integer with appropriate bit size
            item = random.randint(0, (1 << self.lp_structure.eItemBitSize) - 1)
            individual.append(item)
        return individual

    def generate_population(self):
        population = [self.generate_individual() for _ in range(self.population_size)]
        return population

    def fitness(self, individual):
        # Calculate fitness based on the number of bits set to 1
        score = 0
        for i in range(len(individual) * self.lp_structure.eItemBitSize):
            if self.lp_structure.isON(individual, i):
                score += 1
        return score

    def select(self, population):
        scored = [(self.fitness(individual), individual) for individual in population]
        scored.sort(reverse=True)  # Higher fitness is better
        ranked = [individual for (score, individual) in scored]
        return ranked[:int(0.2 * len(ranked))]  # Select top 20%

    def crossover(self, parent1, parent2):
        child1 = parent1.copy()  # Copy parent1
        child2 = parent2.copy()  # Copy parent2

        # Create temporary lists for lJoin operation
        temp1 = child1.copy()
        temp2 = child2.copy()

        # Perform crossover using lJoin
        self.lp_structure.lJoin(child1, parent2)
        self.lp_structure.lJoin(child2, parent1)

        return child1, child2

    def mutate(self, individual):
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < 0.01:  # 1% chance of mutation
                # Flip a random bit
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
                    # If not enough selected individuals, add a new random one
                    next_generation.append(self.generate_individual())

            population = [self.mutate(individual) for individual in next_generation]

        best_individual = self.select(population)[0]
        return best_individual


# Example usage
lp_structure = LPStructure(10, 8)
scp_algorithm = SCPAlgorithm(lp_structure)
best_individual = scp_algorithm.evolve()
print("Best Individual:", best_individual)
