import os
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torchvision import datasets, transforms

from models import build_model


def get_batch_size(
        args,
        num_classes,
        device: torch.device,
        dataset_size: int,
        num_iterations: int = 5
) -> int:
    model, _ = build_model(model_name=args.model, pretrained=args.pretrained,
                           num_classes=num_classes, resume=args.resume)
    model.to(device)
    model.train(True)
    optimizer = torch.optim.Adam(model.parameters())

    batch_size = args.batch_size
    print(f"\nChecking batch size ...")
    while True:
        if batch_size >= dataset_size:
            batch_size = batch_size // 2
            break
        try:
            for _ in range(num_iterations):
                # dummy inputs and targets
                inputs = torch.rand(*(batch_size, *(3, 224, 224)), device=device)
                targets = torch.rand(*(batch_size, *(num_classes,)), device=device)
                outputs = model(inputs)
                loss = F.mse_loss(targets, outputs)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            if batch_size < args.batch_size:
                print(f"\tAdjusted batch size from {args.batch_size} to {batch_size} due to CUDA OOM error.")
            else:
                print(f"\tGood to go with batch size of {args.batch_size}.")
            break
            # batch_size *= 2
        except RuntimeError:
            print(f"\tCUDA out of memory at batch size {batch_size}, trying {batch_size // 2} ...")
            batch_size //= 2
            if batch_size < 2:
                print("\tCan't train at batch size < 2.")
                break
    del model, optimizer
    torch.cuda.empty_cache()
    return batch_size


def save_classes(classes, dir_path):
    full_path = os.path.join(dir_path, 'classes.txt')
    with open(full_path, 'w') as f:
        for item in classes:
            f.write("%s\n" % item)


def get_mean_std(dataset_root):
    train_data_root = os.path.join(dataset_root, 'train')
    train_dataset = datasets.ImageFolder(root=train_data_root, transform=transforms.ToTensor())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64)

    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(train_dataloader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    dataset_mean = channels_sum / num_batches
    dataset_std = (channels_sqrd_sum / num_batches - dataset_mean ** 2) ** 0.5

    return dataset_mean, dataset_std
