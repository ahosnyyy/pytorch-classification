from torch import nn
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights

resnet_dict = {'resnet18': {'model': resnet18, 'weights': ResNet18_Weights.DEFAULT},
               'resnet50': {'model': resnet50, 'weights': ResNet50_Weights.DEFAULT}}


def build_resnet(model_name, num_classes, pretrained):

    weights = resnet_dict[model_name]['weights']
    model = resnet_dict[model_name]['model'](weights=weights)

    if pretrained:
        # Freeze model parameters
        for param in model.parameters():
            param.requires_grad = False

        # Change the final layer of ResNet50 Model for Transfer Learning
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),  # Since 10 possible outputs
    )

    return model
