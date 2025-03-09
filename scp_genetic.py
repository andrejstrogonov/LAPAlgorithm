import random

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
            covered.update(subsets[i])
            cost += 1  # Assuming unit cost for simplicity
    if covered == elements:
        return 1 / cost
    else:
        return 0

def select(population, fitnesses):
    total_fitness = sum(fitnesses)
    roulette_wheel = [fitness / total_fitness for fitness in fitnesses]
    selected = random.choices(population, weights=roulette_wheel, k=2)
    return selected

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

def solve_scp_ga(subsets, elements, pop_size=100, max_gens=100, crossover_rate=0.8, mutation_rate=0.1):
    population = generate_initial_population(pop_size, len(subsets))
    for gen in range(max_gens):
        fitnesses = [fitness(chromosome, subsets, elements) for chromosome in population]
        best_chromosome = max(zip(fitnesses, population))[1]
        print(f"Generation {gen}: Best fitness = {fitness(best_chromosome, subsets, elements)}")
        new_population = []
        while len(new_population) < pop_size:
            parents = select(population, fitnesses)
            children = crossover(parents[0], parents[1], crossover_rate)
            children = [mutate(child, mutation_rate) for child in children]
            new_population.extend(children)
        population = new_population
    best_solution = max(zip([fitness(chromosome, subsets, elements) for chromosome in population], population))[1]
    return best_solution

# Пример использования
subsets = [set([1, 2]), set([2, 3]), set([3, 4]), set([4, 5]), set([1, 5])]
elements = set([1, 2, 3, 4, 5])
solution = solve_scp_ga(subsets, elements)
print("Best solution:", solution)