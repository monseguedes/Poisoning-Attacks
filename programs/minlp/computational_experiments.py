"""
File to run all computational experiments of bilevel-based poisoning attacks.
"""

import itertools
import yaml
import flipping_attack
import instance_data_class
import copy

if __name__ == "__main__":
    with open("programs/minlp/config.yml", "r") as config_file:
        init_config = yaml.safe_load(config_file)

    seeds = [1, 2, 3, 4, 5]
    # poison_rates = [0.06, 0.08, 0.12, 0.16, 0.2]
    poison_rates = [0.2]
    batch_size = [1, 6, 'all']
    config = copy.deepcopy(init_config)
    for seed, rate in itertools.product(seeds, poison_rates):
        # Update configs
        config['seed'] = seed
        config['poison_rate'] = rate
        config['numerical_attack_mini_batch_size'] = 6
        instance_data = instance_data_class.InstanceData(
        config, benchmark_data=True, seed=seed
        )
        numerical_model = None

        _, instance_data, regression_parameters = flipping_attack.run(
            config, instance_data, numerical_model
        )
            
