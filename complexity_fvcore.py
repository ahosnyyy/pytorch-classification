import argparse

import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str

from models import build_model


def parse_args():
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True, help="Model name")

    return parser.parse_args()


def main():
    args = parse_args()

    model, _ = build_model(model_name=args.model)

    model.eval()
    sample_input = torch.randn(1, 3, 224, 224)
    flop = FlopCountAnalysis(model, sample_input)

    # Print a table of FLOP counts with a maximum depth of 4
    print("1. FLOP count table (max depth = 4):\n")
    print(flop_count_table(flop, max_depth=4))

    # Print a string representation of the FLOP counts
    # print("\nFLOP count string:\n")
    # print(flop_count_str(flop))

    # Print the total number of FLOPs
    print("\n2. Total FLOP count:", flop.total())

    # Print the FLOP count breakdown by operator
    print("\n3. FLOP count by operator:")
    print(flop.by_operator())


if __name__ == '__main__':
    main()
