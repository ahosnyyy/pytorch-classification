from torch import nn
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large, MobileNet_V3_Large_Weights, \
    MobileNet_V3_Small_Weights

from models.model_utils import get_num_features

mobilenet_dict = {'mobilenet_v3_small': {'model': mobilenet_v3_small, 'weights': MobileNet_V3_Small_Weights.DEFAULT},
                  'mobilenet_v3_large': {'model': mobilenet_v3_large, 'weights': MobileNet_V3_Large_Weights.DEFAULT}}


def build_mobilenet(model_name, num_classes, pretrained):
    weights = mobilenet_dict[model_name]['weights']
    model = mobilenet_dict[model_name]['model'](weights=weights)

    if pretrained:
        # Freeze model parameters
        for param in model.parameters():
            param.requires_grad = False

    num_features = get_num_features(model)

    classifier = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(256, num_classes)
    )

    model.classifier = classifier

    return model
