import random
import timeit
import matplotlib.pyplot as plt
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


def generate_complex_ruleset(num_facts, num_rules, branching_factor, depth):
    """
    Generate a complex ruleset with many rules and facts

    Parameters:
    - num_facts: Number of base facts to add
    - num_rules: Total number of rules to generate
    - branching_factor: How many rules can lead to the same conclusion
    - depth: Maximum inference chain length
    """
    expert_system = ExpertSystem()

    # Generate base facts (A, B, C, ...)
    facts = [chr(65 + i) for i in range(num_facts)]
    for fact in facts:
        expert_system.add_fact(fact)

    # Generate intermediate facts
    intermediate_facts = [f"F{i}" for i in range(1, num_rules + 1)]

    # Generate rules with varying depths
    all_possible_premises = facts.copy()

    for i in range(num_rules):
        # Determine conclusion
        if i < len(intermediate_facts):
            conclusion = intermediate_facts[i]
        else:
            # Reuse some intermediate facts for more complex chains
            conclusion = random.choice(intermediate_facts)

        # Select premises
        num_premises = random.randint(1, min(branching_factor, len(all_possible_premises)))
        premises = random.sample(all_possible_premises, num_premises)

        # Add the rule
        expert_system.add_rule(premises, conclusion)

        # Add conclusion to possible premises for future rules (creates depth)
        if len(all_possible_premises) < depth * num_facts:
            all_possible_premises.append(conclusion)

    # Add a final goal that requires multiple inference steps
    final_goal = "GOAL"
    premises_for_goal = random.sample(intermediate_facts, min(branching_factor, len(intermediate_facts)))
    expert_system.add_rule(premises_for_goal, final_goal)

    return expert_system, final_goal


def benchmark_comparison(num_facts_list, num_rules_list):
    """
    Run benchmarks comparing SCP vs classic backward chaining
    """
    scp_times = []
    classic_times = []
    scp_rules_used = []
    classic_rules_used = []

    for num_facts, num_rules in zip(num_facts_list, num_rules_list):
        print(f"Testing with {num_facts} facts and {num_rules} rules...")

        # Generate test case
        expert_system, goal = generate_complex_ruleset(
            num_facts=num_facts,
            num_rules=num_rules,
            branching_factor=3,
            depth=5
        )

        # Create a copy for classic algorithm
        classic_system = ClassicExpertSystem()
        classic_system.facts = expert_system.facts.copy()
        classic_system.knowledge_base = expert_system.knowledge_base.copy()

        # Setup SCP algorithm
        lp_structure = LPStructure(10, 8)
        scp_algorithm = SCPAlgorithm(lp_structure)

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

        # Record results
        scp_times.append(scp_time)
        classic_times.append(classic_time)
        scp_rules_used.append(len(scp_used_rules))
        classic_rules_used.append(len(classic_used_rules))

        print(f"  SCP: {scp_time:.6f}s, Rules used: {len(scp_used_rules)}")
        print(f"  Classic: {classic_time:.6f}s, Rules used: {len(classic_used_rules)}")
        print(
            f"  Improvement: {(classic_time / scp_time):.2f}x faster, {(classic_rules_used[-1] / scp_rules_used[-1]):.2f}x fewer rules")
        print()

    return scp_times, classic_times, scp_rules_used, classic_rules_used


def plot_results(num_rules_list, scp_times, classic_times, scp_rules, classic_rules):
    """
    Plot the benchmark results
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Time comparison
    ax1.plot(num_rules_list, scp_times, 'o-', label='SCP Algorithm')
    ax1.plot(num_rules_list, classic_times, 's-', label='Classic Algorithm')
    ax1.set_xlabel('Number of Rules')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Inference Time Comparison')
    ax1.legend()
    ax1.grid(True)

    # Rules used comparison
    ax2.plot(num_rules_list, scp_rules, 'o-', label='SCP Algorithm')
    ax2.plot(num_rules_list, classic_rules, 's-', label='Classic Algorithm')
    ax2.set_xlabel('Number of Rules')
    ax2.set_ylabel('Rules Used')
    ax2.set_title('Rules Used Comparison')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('scp_vs_classic_comparison.png')
    plt.show()


if __name__ == "__main__":
    # Define test cases with increasing complexity
    num_facts_list = [5, 10, 15, 20, 25]
    num_rules_list = [20, 50, 100, 200, 300]

    # Run benchmarks
    scp_times, classic_times, scp_rules, classic_rules = benchmark_comparison(
        num_facts_list, num_rules_list
    )

    # Plot results
    plot_results(num_rules_list, scp_times, classic_times, scp_rules, classic_rules)

    # Print summary
    print("\nSummary:")
    print("========")
    print(f"Average speedup: {sum(c / s for c, s in zip(classic_times, scp_times)) / len(classic_times):.2f}x")
    print(f"Average rule reduction: {sum(c / s for c, s in zip(classic_rules, scp_rules)) / len(classic_rules):.2f}x")