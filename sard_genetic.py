import random

def generate_initial_population(size, gene_length):
    population = []
    for _ in range(size):
        chromosome = [random.choice([0, 1]) for _ in range(gene_length)]
        population.append(chromosome)
    return population

def fitness(chromosome, target):
    score = 0
    for gene, target_gene in zip(chromosome, target):
        if gene == target_gene:
            score += 1
    return score

def selection(population, fitnesses, num_parents):
    total_fitness = sum(fitnesses)
    roulette_wheel = [fitness / total_fitness for fitness in fitnesses]
    parents = random.choices(population, weights=roulette_wheel, k=num_parents)
    return parents

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

def solve_sard_ga(target, pop_size=100, max_gens=100, crossover_rate=0.8, mutation_rate=0.1):
    gene_length = len(target)
    population = generate_initial_population(pop_size, gene_length)
    for gen in range(max_gens):
        fitnesses = [fitness(chromosome, target) for chromosome in population]
        best_chromosome = max(zip(fitnesses, population), key=lambda x: x[0])[1]
        print(f"Generation {gen}: Best fitness = {fitness(best_chromosome, target)}")
        parents = selection(population, fitnesses, pop_size // 2)
        new_population = []
        while len(new_population) < pop_size:
            parent1, parent2 = random.choice(parents), random.choice(parents)
            children = crossover(parent1, parent2, crossover_rate)
            children = [mutate(child, mutation_rate) for child in children]
            new_population.extend(children)
        population = new_population
    best_solution = max(zip([fitness(chromosome, target) for chromosome in population], population), key=lambda x: x[0])[1]
    return best_solution

# Пример использования
target = [1, 0, 1, 1, 0, 0, 1, 0]
solution = solve_sard_ga(target)
print("Best solution:", solution)