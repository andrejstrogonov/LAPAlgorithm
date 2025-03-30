
import timeit

class Rule:
    def __init__(self, premises, conclusion):
        self.premises = premises
        self.conclusion = conclusion

class ExpertSystem:
    def __init__(self):
        self.knowledge_base = []
        self.facts = set()

    def add_rule(self, premises, conclusion):
        rule = Rule(premises, conclusion)
        self.knowledge_base.append(rule)

    def add_fact(self, fact):
        self.facts.add(fact)

    def backward_inference(self, goal, used_rules=None):
        if used_rules is None:
            used_rules = []

        if goal in self.facts:
            return True

        for rule in self.knowledge_base:
            if rule.conclusion == goal:
                all_premises_true = True
                for premise in rule.premises:
                    if not self.backward_inference(premise, used_rules):
                        all_premises_true = False
                        break
                if all_premises_true:
                    used_rules.append(rule)
                    return True

        return False

def measure_inference_time(expert_system, goal):
    used_rules = []
    result = expert_system.backward_inference(goal, used_rules)
    return result, used_rules

# Example usage
if __name__ == "__main__":
    expert_system = ExpertSystem()
    expert_system.add_fact("A")
    expert_system.add_fact("D")

    # Adding rules
    expert_system.add_rule(["A"], "B")
    expert_system.add_rule(["B"], "C")
    expert_system.add_rule(["D"], "E")
    expert_system.add_rule(["E", "B"], "F")

    # Measure time for goal "F"
    goal = "F"
    time_taken = timeit.timeit(lambda: measure_inference_time(expert_system, goal), number=1)
    result, used_rules = measure_inference_time(expert_system, goal)
    print(f"Goal '{goal}' is {'proven' if result else 'not proven'}")
    print(f"Time taken: {time_taken:.6f} seconds")

    # Print the rules used to achieve the goal
    if result:
        print("Rules used to achieve the goal:")
        for rule in used_rules:
            print(f"Premises: {rule.premises} -> Conclusion: {rule.conclusion}")

    # Measure time for another goal "C"
    goal = "C"
    time_taken = timeit.timeit(lambda: measure_inference_time(expert_system, goal), number=1)
    result, used_rules = measure_inference_time(expert_system, goal)
    print(f"Goal '{goal}' is {'proven' if result else 'not proven'}")
    print(f"Time taken: {time_taken:.6f} seconds")

    # Print the rules used to achieve the goal
    if result:
        print("Rules used to achieve the goal:")
        for rule in used_rules:
            print(f"Premises: {rule.premises} -> Conclusion: {rule.conclusion}")
