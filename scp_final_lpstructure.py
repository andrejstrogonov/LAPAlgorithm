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

class LPStructureReduction(LPStructure):
    def __init__(self, eLengthItems, eItemBitSize, prContainer):
        super().__init__(eLengthItems, eItemBitSize)
        self.prContainer = prContainer

    def lReductionLC(self, onEvent, dwUser):
        # Удаление логически связанных пар
        self.prContainer = [pair for pair in self.prContainer if not self.isLConnected(pair, onEvent, dwUser)]
        return len(self.prContainer)

    def isLConnected(self, aPair, onEvent, dwUser):
        eRight = self.right(aPair)  # Элемент, который требуется получить при выводе
        eRes = self.left(aPair)  # Текущий результат полного прямого вывода

        if not eRight or not eRes:
            return False

        # Проверка на "подчинённую" пару
        if self.LE(eRight, eRes):
            return True

        # Память для результата логического вывода
        bRes = bytearray(eRes)
        eRes = bRes

        # Вектор для возможного запоминания цепочки вывода
        prSecVector = []

        res = False
        wasAdded = True

        # Копия исходного множества пар
        tmpContainer = list(self.prContainer)
        tmpContainer.remove(aPair)  # Будем искать логическую связь в остальных парах

        while not res and wasAdded:
            wasAdded = False
            i = 0
            while i < len(tmpContainer):
                prCurr = tmpContainer[i]
                if self.LE(self.left(prCurr), eRes):  # left(prCurr) <= eRes
                    # Задан режим запоминания вывода - сохраняем пары, вносящие вклад
                    if onEvent and not self.LE(self.right(prCurr), eRes):
                        prSecVector.append(prCurr)

                    self.lJoin(eRes, self.right(prCurr))  # eRes |= right(prCurr)
                    wasAdded = True  # Результат увеличился

                    tmpContainer.pop(i)  # Каждая пара используется единственный раз

                    if self.LE(eRight, eRes):
                        res = True
                        break  # eRight <= eRes
                else:
                    i += 1

        # Не зря запоминали вывод - сообщаем о результатах
        if res and onEvent:
            onEvent(aPair, "etRedundant", dwUser)  # Лишняя пара
            # Её вывод
            for k in prSecVector:
                onEvent(k, "etInference", dwUser)

        return res

    # Вспомогательные методы для работы с парами
    def left(self, pair):
        return pair[0]

    def right(self, pair):
        return pair[1]

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
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
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

class TestLPStructureReduction(unittest.TestCase):
    def setUp(self):
        self.lp_reduction = LPStructureReduction(3, 8, [([1, 2, 3], [4, 5, 6])])

    def test_lReductionLC(self):
        def dummy_event(pair, event_type, user_data):
            pass

        result = self.lp_reduction.lReductionLC(dummy_event, None)
        self.assertEqual(result, 1)

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