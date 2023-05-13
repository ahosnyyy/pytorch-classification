from torch import nn


def get_num_features(model):
    for name, layer in model.classifier.named_children():
        if isinstance(layer, nn.Linear):
            num_features = model.classifier[int(name)].in_features
            break

    return num_features
