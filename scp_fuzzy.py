import numpy as np


def fuzzy_membership(element, subset, membership_function):
    # Пример функции принадлежности
    return membership_function(element, subset)


def solve_scp_fuzzy(U, S, weights, membership_function):
    n = len(S)
    m = len(U)

    # Инициализация матриц принадлежности
    membership_matrix = np.zeros((n, m))
    for i, subset in enumerate(S):
        for j, element in enumerate(U):
            membership_matrix[i, j] = fuzzy_membership(element, subset, membership_function)

    # Расчёт нечёткой стоимости для каждого подмножества
    fuzzy_costs = np.dot(membership_matrix, weights)

    # Выбор подмножеств с минимальной нечёткой стоимостью
    selected_subsets = []
    total_cost = 0
    while len(selected_subsets) < n:
        min_cost_index = np.argmin(fuzzy_costs)
        selected_subsets.append(min_cost_index)
        total_cost += weights[min_cost_index]
        fuzzy_costs[min_cost_index] = np.inf  # Исключаем выбранное подмножество из дальнейшего рассмотрения

    return selected_subsets, total_cost


# Пример использования
U = 1, 2, 3, 4, 5
S = set(1, 2), set(2, 3), set(3, 4), set(4, 5), set(1, 5)
weights = 1, 1, 1, 1, 1

# Исправленное создание множеств
S = {1, 2}, {2, 3}, {3, 4}, {4, 5}, {1, 5}


def example_membership_function(element, subset):
    if element in subset:
        return 1.0
    else:
        return 0.0


selected_subsets, total_cost = solve_scp_fuzzy(U, S, weights, example_membership_function)
print("Selected subsets:", selected_subsets)
print("Total cost:", total_cost)