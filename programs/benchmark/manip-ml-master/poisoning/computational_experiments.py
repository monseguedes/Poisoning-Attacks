"""
File to run all computational experiments of gradient-based poisoning attacks.
"""

from my_args import setup_argparse
from poison import main
import itertools

if __name__ == "__main__":
    seeds = [1, 2, 3, 4, 5]
    poiscts = [12, 24, 36, 48, 60]
    for seed, poisct in itertools.product(seeds, poiscts):
        print("-----------------------------------------------------------")
        print("starting poison ...\n")
        parser = setup_argparse(poisct=poisct, seed=seed)
        args = parser.parse_args()

        print("-----------------------------------------------------------")
        print(args)
        print("-----------------------------------------------------------")
        main(args)