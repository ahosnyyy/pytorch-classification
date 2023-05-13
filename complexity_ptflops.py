import argparse

import torch
from ptflops import get_model_complexity_info
from models import build_model


def parse_args():
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True, help="Model name")

    return parser.parse_args()


def main():
    args = parse_args()

    with torch.cuda.device(0):
        model, _ = build_model(model_name=args.model)
        macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                                 print_per_layer_stat=True, verbose=False)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


if __name__ == "__main__":
    main()
