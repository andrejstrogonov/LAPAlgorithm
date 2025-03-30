import timeit
from collections import defaultdict


class Rule:
    def __init__(self, premises, conclusion):
        self.premises = premises
        self.conclusion = conclusion

    def __repr__(self):
        return f"Rule({self.premises} -> {self.conclusion})"


class OptimizedExpertSystem:
    def __init__(self):
        self.knowledge_base = []
        self.facts = set()
        # Index rules by conclusion for faster lookup
        self.conclusion_to_rules = defaultdict(list)
        # Cache for memoization
        self.inference_cache = {}

    def add_rule(self, premises, conclusion):
        rule = Rule(premises, conclusion)
        self.knowledge_base.append(rule)
        self.conclusion_to_rules[conclusion].append(rule)

    def add_fact(self, fact):
        self.facts.add(fact)
        # Clear cache when facts change
        self.inference_cache = {}

    def backward_inference(self, goal, path=None, depth=0, max_depth=100):
        """
        Optimized backward chaining algorithm with:
        - Memoization to avoid redundant computations
        - Path tracking to avoid cycles
        - Depth limiting to prevent infinite recursion
        - Rule indexing for faster lookups
        """
        # Initialize path tracking if not provided
        if path is None:
            path = set()
            self.used_rules = []
            self.inference_cache = {}  # Reset cache for new inference

        # Check depth limit to prevent infinite recursion
        if depth > max_depth:
            return False

        # Check if goal is already a known fact
        if goal in self.facts:
            return True

        # Check if we've already computed this goal
        if goal in self.inference_cache:
            return self.inference_cache[goal]

        # Check for cycles in the inference path
        if goal in path:
            return False

        # Add current goal to path
        path.add(goal)

        # Get rules that can derive this goal
        relevant_rules = self.conclusion_to_rules.get(goal, [])

        # Sort rules by number of premises (prefer simpler rules first)
        relevant_rules.sort(key=lambda r: len(r.premises))

        for rule in relevant_rules:
            all_premises_true = True

            # Check each premise
            for premise in rule.premises:
                if not self.backward_inference(premise, path, depth + 1, max_depth):
                    all_premises_true = False
                    break

            if all_premises_true:
                # Add this rule to the used rules list
                if rule not in self.used_rules:
                    self.used_rules.append(rule)

                # Remove goal from path before returning
                path.remove(goal)

                # Cache the result
                self.inference_cache[goal] = True
                return True

        # Remove goal from path before returning
        path.remove(goal)

        # Cache the negative result
        self.inference_cache[goal] = False
        return False


def measure_optimized_inference_time(expert_system, goal):
    result = expert_system.backward_inference(goal)
    used_rules = expert_system.used_rules if hasattr(expert_system, 'used_rules') else []
    return result, used_rules


# Example usage
if __name__ == "__main__":
    # Original system for comparison
    original_system = OptimizedExpertSystem()
    original_system.add_fact("A")
    original_system.add_fact("D")
    original_system.add_rule(["A"], "B")
    original_system.add_rule(["B"], "C")
    original_system.add_rule(["D"], "E")
    original_system.add_rule(["E", "B"], "F")

    # Create a more complex knowledge base
    complex_system = OptimizedExpertSystem()

    # Add base facts
    complex_system.add_fact("A")
    complex_system.add_fact("D")
    complex_system.add_fact("X")

    # Add rules with various complexity
    complex_system.add_rule(["A"], "B")
    complex_system.add_rule(["B"], "C")
    complex_system.add_rule(["D"], "E")
    complex_system.add_rule(["E", "B"], "F")
    complex_system.add_rule(["X"], "Y")
    complex_system.add_rule(["Y"], "Z")
    complex_system.add_rule(["Z", "C"], "G")
    complex_system.add_rule(["F", "G"], "H")
    complex_system.add_rule(["A", "B", "C", "D", "E"], "F")  # Redundant complex rule

    print("=== Original System ===")
    goal = "F"
    time_taken = timeit.timeit(lambda: measure_optimized_inference_time(original_system, goal), number=1000)
    result, used_rules = measure_optimized_inference_time(original_system, goal)
    print(f"Goal '{goal}' is {'proven' if result else 'not proven'}")
    print(f"Time taken for 1000 runs: {time_taken:.6f} seconds")
    print(f"Rules used: {len(used_rules)}")
    print("Rules used to achieve the goal:")
    for rule in used_rules:
        print(f"Premises: {rule.premises} -> Conclusion: {rule.conclusion}")

    print("\n=== Complex System ===")
    goal = "H"
    time_taken = timeit.timeit(lambda: measure_optimized_inference_time(complex_system, goal), number=1000)
    result, used_rules = measure_optimized_inference_time(complex_system, goal)
    print(f"Goal '{goal}' is {'proven' if result else 'not proven'}")
    print(f"Time taken for 1000 runs: {time_taken:.6f} seconds")
    print(f"Rules used: {len(used_rules)}")
    print("Rules used to achieve the goal:")
    for rule in used_rules:
        print(f"Premises: {rule.premises} -> Conclusion: {rule.conclusion}")