import argparse
import time

import cv2
import torch
import torchvision.transforms as transforms

from models import build_model


def parse_args():
    # create argument parser
    parser = argparse.ArgumentParser(description="Image Prediction using PyTorch")
    parser.add_argument("-i", "--image", type=str, required=True, help="path to input image")
    parser.add_argument('-m', '--model', type=str, default='resnet18',
                        help='model name to evaluate (default: resnet18)')
    parser.add_argument("-w", "--weights", type=str, required=True, help="path to weights checkpoint")
    parser.add_argument("-c", "--classes", type=str, required=True, help="path to classes file")

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


def read_model(model_name='resnet18', weights_path=None):
    model, _ = build_model(model_name=model_name)
    checkpoint = torch.load(weights_path, map_location='cpu')
    # checkpoint state dict
    checkpoint_state_dict = checkpoint.pop("model")
    model.load_state_dict(checkpoint_state_dict)

    return model


def checkpoint_inference(model_name, weights_path, image, classes):
    model = read_model(model_name, weights_path)
    model.float()  # Convert to FP32 to work on FP32 input if the passed model is quantized
    model.eval()

    # Perform inference
    start_time = time.time()
    with torch.no_grad():
        output = model(image)
    end_time = time.time()

    predicted_class = classes[torch.argmax(output)]
    inference_time = end_time - start_time

    return predicted_class, inference_time


def main():
    args = parse_args()

    img = read_transform_image(args.image)
    classes = read_classes(args.classes)

    out_class, inference_time = checkpoint_inference(args.model, args.weights, img, classes)

    print(f"Predicted class: {out_class} in {inference_time:.4f} seconds.")


if __name__ == "__main__":
    main()