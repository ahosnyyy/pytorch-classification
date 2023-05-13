import argparse

import torch
from thop import profile

from models import build_model


def parse_args():
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True, help="Model name")

    return parser.parse_args()


def main():
    args = parse_args()
    model, _ = build_model(model_name=args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(1, 3, 224, 224).to(device)
    model.eval().to(device)

    flops, params = profile(model, inputs=(x,))
    print('FLOPs : ', flops / 1e9, ' B')
    print('Params : ', params / 1e6, ' M')


if __name__ == '__main__':
    main()
