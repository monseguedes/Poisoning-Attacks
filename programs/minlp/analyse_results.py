"""
File to check results make sense
"""

import numpy as np

seed = 3
poisoning_rate = 0.2

# Import gradient results
gradient_dict = np.load(
    f"programs/benchmark/manip-ml-master/poisoning/results/{seed}_{int(300 * poisoning_rate)}_gradient_results.npy",
    allow_pickle=True,
)

print(gradient_dict)

# Import bilevel results
bilevel_dict = np.load(
    f"programs/minlp/results/{seed}_{int(poisoning_rate * 100)}_bilevel_results.npy",
    allow_pickle=True,
)


# Print table with comparison, gradient as first column, benchmark as second, and our method as third, each row is type of data
# print("Score & Gradient & Benchmark & Our Method \\\\")
# print(
#     f"Poisoned mse validation & {gradient_dict.item()['poisoned_validation_mse']} & {bilevel_dict.item()['benchmark_validation_mse']} & {bilevel_dict.item()['flippin_validation_mse']} \\\\"
# )
# print(
#     f"Poisoned mse test & {gradient_dict.item()['poisoned_test_mse']} & {bilevel_dict.item()['benchmark_test_mse']} & {bilevel_dict.item()['flipping_test_mse']} \\\\"
# )

# Print formatted table with all columns aligned, including headings
print(
    f"{'Score':<25}  {'Gradient':<25}  {'Benchmark':<25}  {'Our Method':<25} \\"
)
print("-" * 100)
print(
    f"{'Poisoned mse validation':<25}  {gradient_dict.item()['poisoned_validation_mse']:<25}  {bilevel_dict.item()['benchmark_validation_mse']:<25}  {bilevel_dict.item()['flippin_validation_mse']:<25} \\"
)
print(
    f"{'Poisoned mse test':<25}  {gradient_dict.item()['poisoned_test_mse']:<25}  {bilevel_dict.item()['benchmark_test_mse']:<25}  {bilevel_dict.item()['flipping_test_mse']:<25} \\"
)


