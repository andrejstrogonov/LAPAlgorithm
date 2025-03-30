import timeit
from scp_final_lpstructure import ExpertSystem, SCPAlgorithm, LPStructure, Rule


class ClassicExpertSystem(ExpertSystem):
    """
    A classic expert system that uses simple backward chaining without SCP optimization
    """

    def backward_inference_classic(self, goal, used_rules=None):
        if used_rules is None:
            used_rules = []

        if goal in self.facts:
            return True

        # Classic backward chaining - try all rules in the order they were added
        for rule in self.knowledge_base:
            if rule.conclusion == goal:
                if all(self.backward_inference_classic(premise, used_rules) for premise in rule.premises):
                    used_rules.append(rule)
                    return True

        return False


def create_complex_ruleset():
    """
    Create a complex ruleset with many redundant and irrelevant rules
    that will challenge the classic algorithm but be handled efficiently by SCP
    """
    expert_system = ExpertSystem()

    # Add base facts
    base_facts = ["A", "B", "C", "D", "E"]
    for fact in base_facts:
        expert_system.add_fact(fact)

    # Create a network of rules with many paths to the same conclusions
    # This creates a situation where SCP can prioritize better

    # Layer 1 - Multiple ways to derive the same facts
    for i in range(10):
        # Create multiple rules that lead to the same conclusions
        expert_system.add_rule(["A"], f"L1_{i}")
        expert_system.add_rule(["B"], f"L1_{i}")
        expert_system.add_rule(["C"], f"L1_{i}")

    # Layer 2 - Build on layer 1
    for i in range(5):
        for j in range(10):
            expert_system.add_rule([f"L1_{j}"], f"L2_{i}")

    # Layer 3 - Multiple paths to the goal
    for i in range(5):
        expert_system.add_rule([f"L2_{i}"], "GOAL")

    # Add some irrelevant rules that classic algorithm will waste time on
    for i in range(50):
        expert_system.add_rule(["D", "E"], f"IRRELEVANT_{i}")
        expert_system.add_rule([f"IRRELEVANT_{i}"], f"IRRELEVANT_BRANCH_{i}")

    return expert_system


def run_comparison():
    # Create the expert system with our complex ruleset
    expert_system = create_complex_ruleset()

    # Create a copy for classic algorithm
    classic_system = ClassicExpertSystem()
    classic_system.facts = expert_system.facts.copy()
    classic_system.knowledge_base = expert_system.knowledge_base.copy()

    # Setup SCP algorithm
    lp_structure = LPStructure(10, 8)
    scp_algorithm = SCPAlgorithm(lp_structure)

    # Goal to prove
    goal = "GOAL"

    # Measure SCP performance
    scp_used_rules = []
    scp_time = timeit.timeit(
        lambda: expert_system.backward_inference(goal, scp_algorithm, scp_used_rules),
        number=5
    ) / 5  # Average of 5 runs

    # Measure classic performance
    classic_used_rules = []
    classic_time = timeit.timeit(
        lambda: classic_system.backward_inference_classic(goal, classic_used_rules),
        number=5
    ) / 5  # Average of 5 runs

    # Print results
    print("\nComparison Results:")
    print("===================")
    print(f"Total rules in knowledge base: {len(expert_system.knowledge_base)}")
    print(f"Goal to prove: {goal}")
    print("\nSCP Algorithm:")
    print(f"  Time: {scp_time:.6f} seconds")
    print(f"  Rules used: {len(scp_used_rules)}")
    print("\nClassic Algorithm:")
    print(f"  Time: {classic_time:.6f} seconds")
    print(f"  Rules used: {len(classic_used_rules)}")

    print("\nImprovement:")
    print(f"  Speed improvement: {(classic_time / scp_time):.2f}x faster")
    print(f"  Rule usage reduction: {(len(classic_used_rules) / len(scp_used_rules)):.2f}x fewer rules")

    # Print the rules used by each algorithm
    print("\nRules used by SCP algorithm:")
    for i, rule in enumerate(scp_used_rules):
        print(f"  {i + 1}. {rule.premises} -> {rule.conclusion}")

    print("\nRules used by Classic algorithm:")
    for i, rule in enumerate(classic_used_rules[:10]):  # Show only first 10 to avoid overwhelming output
        print(f"  {i + 1}. {rule.premises} -> {rule.conclusion}")

    if len(classic_used_rules) > 10:
        print(f"  ... and {len(classic_used_rules) - 10} more rules")


if __name__ == "__main__":
    run_comparison()