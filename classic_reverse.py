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

    def backward_inference(self, goal):
        if goal in self.facts:
            return True

        for rule in self.knowledge_base:
            if rule.conclusion == goal:
                all_premises_true = True
                for premise in rule.premises:
                    if not self.backward_inference(premise):
                        all_premises_true = False
                        break
                if all_premises_true:
                    return True

        return False

# Example usage
if __name__ == "__main__":
    es = ExpertSystem()
    es.add_fact("A")
    es.add_rule(["A"], "B")
    es.add_rule(["B"], "C")

    goal = "C"
    result = es.backward_inference(goal)
    print(f"Goal '{goal}' is {'proven' if result else 'not proven'}")
