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

def generate_initial_population(size, num_subsets):
    population = []
    for _ in range(size):
        chromosome = [random.choice([0, 1]) for _ in range(num_subsets)]
        population.append(chromosome)
    return population

def fitness(chromosome, subsets, lattice):
    covered = set()
    cost = 0
    subset_keys = list(subsets.keys())  # Get the keys once to avoid repeated calls
    for i, bit in enumerate(chromosome):
        if bit == 1:
            if i < len(subset_keys):  # Ensure the index is within bounds
                point = subset_keys[i]
                covered.update(subsets[point])
                cost += lattice.calculate_distance(point, point)
    if covered == set(lattice.points):
        return 1 / cost
    else:
        return 0

def crossover(parent1, parent2, crossover_rate, lattice):
    if random.random() < crossover_rate:
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    else:
        return parent1, parent2

def mutate(chromosome, mutation_rate, lattice):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]
    return chromosome

def solve_scp_fca(formal_context, pop_size=100, max_gens=100, crossover_rate=0.8, mutation_rate=0.1):
    num_points = 10
    lattice = LPStructure(points)
    subsets = {point: formal_context.relations[point] for point in lattice.points}  # Initialize subsets here
    population = generate_initial_population(pop_size, len(subsets))  # Use the length of subsets
    for gen in range(max_gens):
        fitnesses = [fitness(chromosome, subsets, lattice) for chromosome in population]
        # Add a small positive value to the fitnesses list
        fitnesses = [f + 1e-6 for f in fitnesses]
        best_chromosome = max(zip(fitnesses, population))[1]
        print(f"Generation {gen}: Best fitness = {fitness(best_chromosome, subsets, lattice)}")
        new_population = []
        while len(new_population) < pop_size:
            parents = random.choices(population, weights=fitnesses, k=2)
            children = crossover(parents[0], parents[1], crossover_rate, lattice)
            children = [mutate(child, mutation_rate, lattice) for child in children]
            new_population.extend(children)
        population = new_population
    best_solution = max(zip([fitness(chromosome, subsets, lattice) for chromosome in population], population))[1]
    return best_solution

# Пример использования
points = [(x, y) for x in range(10) for y in range(10)]
relations = {point: set([f'{x}{y}' for x in range(3) for y in range(3)]) for point in points}
formal_context = sequences = [
    ['A', 'B', 'C'],
    ['A', 'B', 'D'],
    ['B', 'C', 'D'],
    ['A', 'C', 'D'],
    ['A', 'B', 'C', 'D']
]  # Import the formal_context module and define it here
lattice = LPStructure(points)
solution = solve_scp_fca(formal_context)
print("Best solution:", solution)
