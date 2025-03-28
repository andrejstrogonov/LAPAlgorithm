import random
import unittest


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
        individual = []
        for _ in range(self.lp_structure.eLengthItems):
            item = [random.choice([0, 1]) for _ in range(self.lp_structure.eItemBitSize)]
            individual.append(item)
        return individual

    def generate_population(self):
        population = [self.generate_individual() for _ in range(self.population_size)]
        return population

    def fitness(self, individual):
        # Flatten the individual list
        flat_individual = [bit for item in individual for bit in item]
        score = 0
        for i in range(self.lp_structure.eLengthItems):
            if self.lp_structure.isON(flat_individual, i):
                score += 1
        return score

    def select(self, population):
        scored = [(self.fitness(individual), individual) for individual in population]
        scored.sort()
        ranked = [individual for (score, individual) in scored]
        return ranked[:int(0.2 * len(ranked))]  # Select top 20%

    def crossover(self, parent1, parent2):
        child1 = [item[:] for item in parent1]  # Deep copy of parent1
        child2 = [item[:] for item in parent2]  # Deep copy of parent2
        for i in range(self.lp_structure.eLengthItems):
            self.lp_structure.lJoin(child1[i], parent2[i])
            self.lp_structure.lJoin(child2[i], parent1[i])
        return child1, child2

    def mutate(self, individual):
        for i in range(len(individual)):
            if random.random() < 0.01:  # 1% chance of mutation
                individual[i] = [random.choice([0, 1]) for _ in range(self.lp_structure.eItemBitSize)]
        return individual

    def evolve(self):
        population = self.generate_population()
        for _ in range(self.generations):
            selected = self.select(population)
            next_generation = selected[:]
            while len(next_generation) < self.population_size:
                parents = random.sample(selected, 2)
                children = self.crossover(parents[0], parents[1])
                next_generation.extend(children)
            population = [self.mutate(individual) for individual in next_generation]
        best_individual = self.select(population)[0]
        return best_individual

# Пример использования
lp_structure = LPStructure(10, 8)
scp_algorithm = SCPAlgorithm(lp_structure)
best_individual = scp_algorithm.evolve()
print("Best Individual:", best_individual)

class TestSCPAlgorithm(unittest.TestCase):
    def setUp(self):
        self.lp = LPStructure(3, 8)
        self.scp = SCPAlgorithm(self.lp, population_size=10, generations=5)

    def test_generate_individual(self):
        individual = self.scp.generate_individual()
        self.assertEqual(len(individual), 3)
        for item in individual:
            self.assertEqual(len(item), 8)

    def test_generate_population(self):
        population = self.scp.generate_population()
        self.assertEqual(len(population), 10)

    def test_fitness(self):
        individual = [[0b10000000, 0, 0], [0, 0, 0], [0, 0, 0]]
        score = self.scp.fitness(individual)
        self.assertEqual(score, 1)

    def test_select(self):
        population = self.scp.generate_population()
        selected = self.scp.select(population)
        self.assertEqual(len(selected), 2)

    def test_crossover(self):
        parent1 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        parent2 = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        child1, child2 = self.scp.crossover(parent1, parent2)
        self.assertEqual(len(child1), 3)
        self.assertEqual(len(child2), 3)

    def test_mutate(self):
        individual = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        mutated = self.scp.mutate(individual)
        self.assertEqual(len(mutated), 3)

    def test_evolve(self):
        best_individual = self.scp.evolve()
        self.assertEqual(len(best_individual), 3)

if __name__ == '__main__':
    unittest.main()