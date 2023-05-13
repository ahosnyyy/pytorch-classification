import argparse
import os

import torch

from models import build_model


def parse_args():
    # create argument parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('-m', '--model', type=str, default='resnet18',
                        help='model name to evaluate (default: resnet18)')
    parser.add_argument('-w', '--weights', type=str, required=True,
                        help='path to model weights file')

    return parser.parse_args()


def quantize_model(model_name='resnet18', weights=None):    
    model, _ = build_model(model_name=model_name)

    checkpoint = torch.load(weights, map_location='cpu')
    # checkpoint state dict
    checkpoint_state_dict = checkpoint.pop("model")
    model.load_state_dict(checkpoint_state_dict)

    model.half()
    torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.float16, inplace=False)

    dir_path = os.path.join(os.path.dirname(weights), 'converted_models')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    # file_name = os.path.splitext(os.path.basename(weights))[0]
    file_path = os.path.join(dir_path, f'{model_name}_quantized.pth')

    torch.save({'model': model.state_dict()}, file_path)

    print(f'Model exported to {file_path}')


def main():
    args = parse_args()
    quantize_model(args.model, args.weights)


if __name__ == "__main__":
    main()
