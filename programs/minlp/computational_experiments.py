"""
File to run all computational experiments of bilevel-based poisoning attacks.
"""

import itertools
import yaml
import flipping_attack
import instance_data_class

if __name__ == "__main__":
    with open("programs/minlp/config.yml", "r") as config_file:
        config = yaml.safe_load(config_file)

    seeds = [1, 2, 3, 4, 5]
    poiscts = [12, 24, 36, 48, 60]
    for seed, poisct in itertools.product(seeds, poiscts):
        instance_data = instance_data_class.InstanceData(
        config, benchmark_data=True, seed=seed, poisoning_samples=poisct
        )
        numerical_model = None

        _, instance_data, regression_parameters = flipping_attack.run(
            config, instance_data, numerical_model
        )
            
