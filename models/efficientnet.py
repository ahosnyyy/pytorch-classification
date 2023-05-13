from torch import nn
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s, efficientnet_v2_m, \
    EfficientNet_V2_M_Weights, EfficientNet_V2_L_Weights, efficientnet_v2_l

from models.model_utils import get_num_features

efficientnet_dict = {'efficientnet_v2_s': {'model': efficientnet_v2_s, 'weights': EfficientNet_V2_S_Weights.DEFAULT},
                     'efficientnet_v2_m': {'model': efficientnet_v2_m, 'weights': EfficientNet_V2_M_Weights.DEFAULT},
                     'efficientnet_v2_l': {'model': efficientnet_v2_l, 'weights': EfficientNet_V2_L_Weights.DEFAULT}}


def build_efficientnet(model_name, num_classes, pretrained):
    weights = efficientnet_dict[model_name]['weights']
    model = efficientnet_dict[model_name]['model'](weights=weights)

    if pretrained:
        # Freeze model parameters
        for param in model.parameters():
            param.requires_grad = False

    num_features = get_num_features(model)

    model.classifier = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4),
        nn.Linear(512, num_classes),
    )

    return model


