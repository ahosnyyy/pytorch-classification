import argparse

import torch
from torchscan import summary, crawl_module

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
        model.eval().cuda()
        summary(model, (3, 224, 224), receptive_field=True, max_depth=2)


if __name__ == '__main__':
    main()
