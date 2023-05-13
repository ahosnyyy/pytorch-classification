import argparse
import time

import cv2
import numpy as np
import onnxruntime
import torch
import torchvision.transforms as transforms


def parse_args():
    # create argument parser
    parser = argparse.ArgumentParser(description="Image Prediction using PyTorch")
    parser.add_argument("-i", "--image", type=str, required=True, help="path to input image")
    parser.add_argument("-m", "--model", type=str, required=True, help="path to model")
    parser.add_argument("-c", "--classes", type=str, required=True, help="path to classes file")
    parser.add_argument("-e", "--engine", type=str, choices=["onnx", "torchscript"], default="torchscript",
                        help="choose the inference engine: ONNX or TorchScript")

    return parser.parse_args()


def read_classes(classes_file):
    # Open the text file for reading
    with open(classes_file, 'r') as file:
        # Read the contents of the file
        contents = file.read()
    # Split the contents into lines and assign to a list
    classes = contents.split('\n')

    return classes


def read_transform_image(image_path):
    # Load the image using OpenCV
    img = cv2.imread(image_path)

    # Define the transforms to be applied
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert numpy array to PIL image
        transforms.Resize(256),
        transforms.CenterCrop(224),  # Resize the image
        transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # Normalize the tensor
    ])

    # Apply the transforms to the image
    img = transform(img)
    # Reshape the tensor to match the expected input shape of ResNet50
    img = img.unsqueeze(0)

    return img


def torchScript_inference(model_path, image, classes, device):
    img = image

    model = torch.jit.load(model_path, map_location='cpu')
    model.float()  # Convert to FP32 to work on FP32 input if the passed model is quantized
    model.eval()

    start_time = time.time()
    with torch.no_grad():
        output = model(img)
    end_time = time.time()

    predicted_class = classes[torch.argmax(output)]
    inference_time = end_time - start_time

    return predicted_class, inference_time


def onnx_inference(model_path, img, classes):
    session = onnxruntime.InferenceSession(model_path)

    # Run inference on the input tensor
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    start_time = time.time()
    outputs = session.run([output_name], {input_name: img.numpy()})
    end_time = time.time()

    predictions = np.argmax(outputs[0], axis=1)
    predicted_class = classes[predictions[0]]
    inference_time = end_time - start_time

    return predicted_class, inference_time


def main():
    args = parse_args()

    img = read_transform_image(args.image)
    classes = read_classes(args.classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.engine == 'torchscript':
        out_class, inference_time = torchScript_inference(args.model, img, classes, device)

    elif args.engine == 'onnx':
        out_class, inference_time = onnx_inference(args.model, img, classes)

    else:
        print("Please choose the inference engine '--engine [onnx/torchscript]]'")
        return

    print(f"Predicted class: {out_class} in {inference_time:.4f} seconds.")


if __name__ == "__main__":
    main()
