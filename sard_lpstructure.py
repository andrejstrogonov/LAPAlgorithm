import random
import math

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
class SARDAlgorithm:
    def __init__(self, lp_structure, initial_temperature=1000, cooling_rate=0.95, max_iterations=1000):
        self.lp_structure = lp_structure
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations

    def generate_initial_solution(self):
        return [random.choice([0, 1]) for _ in range(self.lp_structure.eLengthItems * self.lp_structure.eItemBitSize)]

    def fitness(self, solution):
        score = 0
        for i in range(self.lp_structure.eLengthItems):
            if self.lp_structure.isON(solution, i):
                score += 1
        return score

    def generate_neighbor(self, solution):
        neighbor = solution[:]
        index = random.randint(0, len(neighbor) - 1)
        neighbor[index] = 1 - neighbor[index]  # Flip the bit
        return neighbor

    def anneal(self):
        current_solution = self.generate_initial_solution()
        current_fitness = self.fitness(current_solution)
        best_solution = current_solution[:]
        best_fitness = current_fitness

        for _ in range(self.max_iterations):
            neighbor = self.generate_neighbor(current_solution)
            neighbor_fitness = self.fitness(neighbor)

            if neighbor_fitness > current_fitness or random.random() < math.exp((neighbor_fitness - current_fitness) / self.temperature):
                current_solution = neighbor
                current_fitness = neighbor_fitness

                if current_fitness > best_fitness:
                    best_solution = current_solution[:]
                    best_fitness = current_fitness

            self.temperature *= self.cooling_rate

        return best_solution

# Example usage
lp_structure = LPStructure(10, 8)
sard_algorithm = SARDAlgorithm(lp_structure)
best_solution = sard_algorithm.anneal()
print("Best Solution:", best_solution)