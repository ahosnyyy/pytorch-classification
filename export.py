import argparse
import os

import onnx
import torch

from models import build_model


def parse_args():
    # create argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', type=str, required=True, help="Model name")
    parser.add_argument('-w', '--weights', type=str, required=True, help="Path to pt weights")
    parser.add_argument('-e', '--engine', type=str, required=True, help="Inference engine [onnx/torchscript]")
    parser.add_argument('-q', '--quantized', action="store_true", default=False, help="If exporting a quantized model")

    return parser.parse_args()


def read_model(model_name='resnet18', weights_path=None):
    model, _ = build_model(model_name=model_name)
    checkpoint = torch.load(weights_path, map_location='cpu')
    # checkpoint state dict
    checkpoint_state_dict = checkpoint.pop("model")
    model.load_state_dict(checkpoint_state_dict)

    return model


def construct_file_path(weights_path, engine, file_name):
    dir_path = os.path.join(os.path.dirname(weights_path), 'converted_models')

    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    # file_name = os.path.splitext(os.path.basename(weights_path))[0]

    if engine == 'torchscript':
        file_path = os.path.join(dir_path, f'{file_name}_{engine}.pt')

    elif engine == 'onnx':
        file_path = os.path.join(dir_path, f'{file_name}_{engine}.onnx')

    return file_path


def torchScript_export(model, weights_path, engine, is_quantized, file_name):

    if is_quantized:
        model.half()
        model = torch.quantization.convert(model, inplace=False)

    model.eval()
    file_path = construct_file_path(weights_path, engine, file_name)

    # Save for deployment
    try:
        model_scripted = torch.jit.script(model)  # Export to TorchScript
        model_scripted.save(file_path)  # Save
        print(f'Model exported to {file_path}')
    except Exception as e:
        print("TorchScript export failed:", e)


def onnx_export(model, weights_path, engine, file_name):

    dummy_input = torch.randn(1, 3, 224, 224)  # Create a dummy input tensor

    # Set the model to inference mode
    model.eval()
    file_path = construct_file_path(weights_path, engine, file_name)

    # Export the model to ONNX format
    input_names = ["input"]
    output_names = ["output"]
    torch.onnx.export(model, dummy_input, file_path, input_names=input_names, output_names=output_names)

    try:
        # Verify that the exported ONNX model is valid
        onnx_model = onnx.load(file_path)
        onnx.checker.check_model(onnx_model)
        print(f'Model exported to {file_path}')

    except onnx.checker.ValidationError as e:
        print("Model is invalid:", e)


def main():
    args = parse_args()
    model = read_model(args.model, args.weights)

    if args.engine == 'torchscript':
        torchScript_export(model, args.weights, args.engine, args.quantized, args.model)

    elif args.engine == 'onnx':
        onnx_export(model, args.weights, args.engine, args.model)

    else:
        print("Please choose the inference engine '--engine [onnx/torchscript]]'")
        return


if __name__ == "__main__":
    main()
