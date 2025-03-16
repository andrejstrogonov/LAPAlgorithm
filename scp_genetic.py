import random


class FormalContext:
    def __init__(self, objects, attributes, relations):
        self.objects = objects
        self.attributes = attributes
        self.relations = relations  # Dictionary where keys are objects and values are sets of attributes

    def derive_concepts(self):
        concepts = []
        for object_set in self.generate_power_set(self.objects):
            intent = self.derive_intent(object_set)
            if intent:
                extent = self.derive_extent(intent)
                if extent:
                    concepts.append((extent, intent))
        return concepts

    def derive_intent(self, object_set):
        intent = set(self.attributes)
        for obj in object_set:
            intent.intersection_update(self.relations[obj])
        return intent

    def derive_extent(self, intent):
        extent = set(self.objects)
        for attr in intent:
            extent.intersection_update([obj for obj, attrs in self.relations.items() if attr in attrs])
        return extent

    def generate_power_set(self, s):
        power_set = [[]]
        for elem in s:
            power_set.extend([subset + [elem] for subset in power_set])
        return power_set

def generate_initial_population(size, num_subsets):
    population = []
    for _ in range(size):
        chromosome = [random.choice([0, 1]) for _ in range(num_subsets)]
        population.append(chromosome)
    return population

def fitness(chromosome, subsets, elements):
    covered = set()
    cost = 0
    for i, bit in enumerate(chromosome):
        if bit == 1:
            covered.update(list(subsets)[i])  # Convert dict_values to list
            cost += 1  # Assuming unit cost for simplicity
    if covered == elements:
        return 1 / cost
    else:
        return 0

def crossover(parent1, parent2, crossover_rate):
    if random.random() < crossover_rate:
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    else:
        return parent1, parent2

def mutate(chromosome, mutation_rate):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]
    return chromosome

def solve_scp_fca(formal_context, pop_size=100, max_gens=100, crossover_rate=0.8, mutation_rate=0.1):
    population = generate_initial_population(pop_size, len(formal_context.relations))
    for gen in range(max_gens):
        fitnesses = [fitness(chromosome, list(formal_context.relations.values()), formal_context.attributes) for chromosome in population]
        # Add a small positive value to the fitnesses list
        fitnesses = [f + 1e-6 for f in fitnesses]
        best_chromosome = max(zip(fitnesses, population))[1]
        print(f"Generation {gen}: Best fitness = {fitness(best_chromosome, list(formal_context.relations.values()), formal_context.attributes)}")
        new_population = []
        while len(new_population) < pop_size:
            parents = random.choices(population, weights=fitnesses, k=2)
            children = crossover(parents[0], parents[1], crossover_rate)
            children = [mutate(child, mutation_rate) for child in children]
            new_population.extend(children)
        population = new_population
    best_solution = max(zip([fitness(chromosome, list(formal_context.relations.values()), formal_context.attributes) for chromosome in population], population))[1]
    return best_solution

# Пример использования
objects = ['A', 'B', 'C', 'D']
attributes = ['a', 'b', 'c', 'd']
relations = {
    'A': {'a', 'b'},
    'B': {'b', 'c'},
    'C': {'c', 'd'},
    'D': {'a', 'd'}
}
formal_context = FormalContext(objects, attributes, relations)
solution = solve_scp_fca(formal_context)
print("Best solution:", solution)

