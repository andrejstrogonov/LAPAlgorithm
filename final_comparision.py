import random
import timeit
import matplotlib.pyplot as plt
import numpy as np
import tracemalloc
import statistics
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


def measure_memory_usage(func, *args, **kwargs):
    """
    Measure peak memory usage of a function
    Returns (result, peak_memory_kb)
    """
    tracemalloc.start()
    result = func(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak / 1024  # Convert to KB


def benchmark_comparison(num_facts_list, num_rules_list, num_runs=5):
    """
    Run benchmarks comparing SCP vs classic backward chaining
    """
    results = {
        'scp': {
            'times': [],
            'memory': [],
            'rules_used': [],
            'time_stats': [],
            'memory_stats': [],
            'rules_stats': []
        },
        'classic': {
            'times': [],
            'memory': [],
            'rules_used': [],
            'time_stats': [],
            'memory_stats': [],
            'rules_stats': []
        }
    }

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

        # Run multiple times to get statistics
        scp_run_times = []
        scp_run_memory = []
        scp_run_rules = []
        classic_run_times = []
        classic_run_memory = []
        classic_run_rules = []

        for _ in range(num_runs):
            # Measure SCP performance
            scp_used_rules = []

            # Measure time
            start_time = timeit.default_timer()
            # Measure memory
            _, peak_memory = measure_memory_usage(
                expert_system.backward_inference,
                goal,
                scp_algorithm,
                scp_used_rules
            )
            end_time = timeit.default_timer()

            scp_run_times.append(end_time - start_time)
            scp_run_memory.append(peak_memory)
            scp_run_rules.append(len(scp_used_rules))

            # Measure classic performance
            classic_used_rules = []

            # Measure time
            start_time = timeit.default_timer()
            # Measure memory
            _, peak_memory = measure_memory_usage(
                classic_system.backward_inference_classic,
                goal,
                classic_used_rules
            )
            end_time = timeit.default_timer()

            classic_run_times.append(end_time - start_time)
            classic_run_memory.append(peak_memory)
            classic_run_rules.append(len(classic_used_rules))

        # Calculate statistics for this test case
        scp_time_stats = {
            'mean': statistics.mean(scp_run_times),
            'median': statistics.median(scp_run_times),
            'stdev': statistics.stdev(scp_run_times) if len(scp_run_times) > 1 else 0,
            'min': min(scp_run_times),
            'max': max(scp_run_times)
        }

        scp_memory_stats = {
            'mean': statistics.mean(scp_run_memory),
            'median': statistics.median(scp_run_memory),
            'stdev': statistics.stdev(scp_run_memory) if len(scp_run_memory) > 1 else 0,
            'min': min(scp_run_memory),
            'max': max(scp_run_memory)
        }

        scp_rules_stats = {
            'mean': statistics.mean(scp_run_rules),
            'median': statistics.median(scp_run_rules),
            'stdev': statistics.stdev(scp_run_rules) if len(scp_run_rules) > 1 else 0,
            'min': min(scp_run_rules),
            'max': max(scp_run_rules)
        }

        classic_time_stats = {
            'mean': statistics.mean(classic_run_times),
            'median': statistics.median(classic_run_times),
            'stdev': statistics.stdev(classic_run_times) if len(classic_run_times) > 1 else 0,
            'min': min(classic_run_times),
            'max': max(classic_run_times)
        }

        classic_memory_stats = {
            'mean': statistics.mean(classic_run_memory),
            'median': statistics.median(classic_run_memory),
            'stdev': statistics.stdev(classic_run_memory) if len(classic_run_memory) > 1 else 0,
            'min': min(classic_run_memory),
            'max': max(classic_run_memory)
        }

        classic_rules_stats = {
            'mean': statistics.mean(classic_run_rules),
            'median': statistics.median(classic_run_rules),
            'stdev': statistics.stdev(classic_run_rules) if len(classic_run_rules) > 1 else 0,
            'min': min(classic_run_rules),
            'max': max(classic_run_rules)
        }

        # Store results
        results['scp']['times'].append(scp_time_stats['mean'])
        results['scp']['memory'].append(scp_memory_stats['mean'])
        results['scp']['rules_used'].append(scp_rules_stats['mean'])
        results['scp']['time_stats'].append(scp_time_stats)
        results['scp']['memory_stats'].append(scp_memory_stats)
        results['scp']['rules_stats'].append(scp_rules_stats)

        results['classic']['times'].append(classic_time_stats['mean'])
        results['classic']['memory'].append(classic_memory_stats['mean'])
        results['classic']['rules_used'].append(classic_rules_stats['mean'])
        results['classic']['time_stats'].append(classic_time_stats)
        results['classic']['memory_stats'].append(classic_memory_stats)
        results['classic']['rules_stats'].append(classic_rules_stats)

        # Print results for this test case
        print(f"  SCP: {scp_time_stats['mean']:.6f}s (±{scp_time_stats['stdev']:.6f}), "
              f"Memory: {scp_memory_stats['mean']:.2f}KB, "
              f"Rules used: {scp_rules_stats['mean']:.2f}")

        print(f"  Classic: {classic_time_stats['mean']:.6f}s (±{classic_time_stats['stdev']:.6f}), "
              f"Memory: {classic_memory_stats['mean']:.2f}KB, "
              f"Rules used: {classic_rules_stats['mean']:.2f}")

        print(f"  Improvement: {(classic_time_stats['mean'] / scp_time_stats['mean']):.2f}x faster, "
              f"{(classic_memory_stats['mean'] / scp_memory_stats['mean']):.2f}x less memory, "
              f"{(classic_rules_stats['mean'] / scp_rules_stats['mean']):.2f}x fewer rules")
        print()

    return results


def plot_results(num_rules_list, results):
    """
    Plot the benchmark results with error bars
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Time comparison
    scp_times = results['scp']['times']
    classic_times = results['classic']['times']
    scp_time_errors = [stats['stdev'] for stats in results['scp']['time_stats']]
    classic_time_errors = [stats['stdev'] for stats in results['classic']['time_stats']]

    ax1.errorbar(num_rules_list, scp_times, yerr=scp_time_errors, fmt='o-', label='SCP Algorithm')
    ax1.errorbar(num_rules_list, classic_times, yerr=classic_time_errors, fmt='s-', label='Classic Algorithm')
    ax1.set_xlabel('Number of Rules')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Inference Time Comparison')
    ax1.legend()
    ax1.grid(True)

    # Memory usage comparison
    scp_memory = results['scp']['memory']
    classic_memory = results['classic']['memory']
    scp_memory_errors = [stats['stdev'] for stats in results['scp']['memory_stats']]
    classic_memory_errors = [stats['stdev'] for stats in results['classic']['memory_stats']]

    ax2.errorbar(num_rules_list, scp_memory, yerr=scp_memory_errors, fmt='o-', label='SCP Algorithm')
    ax2.errorbar(num_rules_list, classic_memory, yerr=classic_memory_errors, fmt='s-', label='Classic Algorithm')
    ax2.set_xlabel('Number of Rules')
    ax2.set_ylabel('Memory Usage (KB)')
    ax2.set_title('Memory Usage Comparison')
    ax2.legend()
    ax2.grid(True)

    # Rules used comparison
    scp_rules = results['scp']['rules_used']
    classic_rules = results['classic']['rules_used']
    scp_rules_errors = [stats['stdev'] for stats in results['scp']['rules_stats']]
    classic_rules_errors = [stats['stdev'] for stats in results['classic']['rules_stats']]

    ax3.errorbar(num_rules_list, scp_rules, yerr=scp_rules_errors, fmt='o-', label='SCP Algorithm')
    ax3.errorbar(num_rules_list, classic_rules, yerr=classic_rules_errors, fmt='s-', label='Classic Algorithm')
    ax3.set_xlabel('Number of Rules')
    ax3.set_ylabel('Rules Used')
    ax3.set_title('Rules Used Comparison')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig('scp_vs_classic_comparison.png')
    plt.show()


def print_detailed_statistics(results, num_rules_list):
    """
    Print detailed statistics for all test cases
    """
    print("\n=== DETAILED STATISTICS ===")

    # Table headers
    headers = ["Rules", "Algorithm", "Time (s)", "Memory (KB)", "Rules Used"]
    row_format = "{:>8} | {:<10} | {:>10} | {:>12} | {:>10}"

    print(row_format.format(*headers))
    print("-" * 60)

    # Print data for each test case
    for i, num_rules in enumerate(num_rules_list):
        # SCP row
        scp_time = f"{results['scp']['times'][i]:.6f} ±{results['scp']['time_stats'][i]['stdev']:.6f}"
        scp_memory = f"{results['scp']['memory'][i]:.2f} ±{results['scp']['memory_stats'][i]['stdev']:.2f}"
        scp_rules = f"{results['scp']['rules_used'][i]:.2f} ±{results['scp']['rules_stats'][i]['stdev']:.2f}"
        print(row_format.format(num_rules, "SCP", scp_time, scp_memory, scp_rules))

        # Classic row
        classic_time = f"{results['classic']['times'][i]:.6f} ±{results['classic']['time_stats'][i]['stdev']:.6f}"
        classic_memory = f"{results['classic']['memory'][i]:.2f} ±{results['classic']['memory_stats'][i]['stdev']:.2f}"
        classic_rules = f"{results['classic']['rules_used'][i]:.2f} ±{results['classic']['rules_stats'][i]['stdev']:.2f}"