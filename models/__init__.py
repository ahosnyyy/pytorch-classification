import torch

from .custom import FashionCNN
from .mobilenet import build_mobilenet
from .resnet import build_resnet
from .efficientnet import build_efficientnet


def build_model(model_name='resnet18', pretrained=False, num_classes=10, resume=None):
    if model_name in ['resnet18', 'resnet50']:
        model = build_resnet(model_name, num_classes, pretrained)

    elif model_name in ['efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l']:
        model = build_efficientnet(model_name, num_classes, pretrained)

    elif model_name in ['mobilenet_v3_small', 'mobilenet_v3_large']:
        model = build_mobilenet(model_name, num_classes, pretrained)

    else:
        model = FashionCNN()

    epoch = 0
    if resume is not None:
        print('Resume training: ', resume)
        checkpoint = torch.load(resume, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        model.load_state_dict(checkpoint_state_dict)
        epoch = checkpoint.pop("epoch")

    return model, epoch
