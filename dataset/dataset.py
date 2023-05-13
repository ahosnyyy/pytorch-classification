import os

import torch
from torchvision import transforms, datasets

from dataset.dataset_utils import get_batch_size, save_classes

pixel_mean = [0.485, 0.456, 0.406]
pixel_std = [0.229, 0.224, 0.225]

image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(pixel_mean,
                             pixel_std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(pixel_mean,
                             pixel_std)
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(pixel_mean,
                             pixel_std)
    ])
}

fashion_mnist_transforms = {
    'train': transforms.Compose([
        transforms.Grayscale(3),
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(pixel_mean,
                             pixel_std)
    ]),
    'val': transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(pixel_mean,
                             pixel_std)
    ]),
    'test': transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(pixel_mean,
                             pixel_std)
    ])
}


def read_dataset(dataset_root, is_eval):
    if not is_eval:
        train_data_root = os.path.join(dataset_root, 'train')
        train_dataset = datasets.ImageFolder(
            root=train_data_root,
            transform=image_transforms['train'])

    val_data_root = os.path.join(dataset_root, 'val')
    val_dataset = datasets.ImageFolder(
        root=val_data_root,
        transform=image_transforms['val'])

    if not is_eval:
        num_classes = len(train_dataset.classes)
        print(f'\nval set of {len(val_dataset)}, examples.')
        save_classes(train_dataset.classes, dataset_root)
        return train_dataset, val_dataset, num_classes, len(train_dataset)

    else:
        num_classes = len(val_dataset.classes)
        return val_dataset, num_classes, len(val_dataset)


def build_dataloader(args, device, is_eval=False):

    if not is_eval:
        train_dataset, val_dataset, num_classes, dataset_size = read_dataset(args.data_path, is_eval)
        calculated_batch_size = get_batch_size(args, num_classes, device, dataset_size)
        batch_size = min(calculated_batch_size, args.batch_size)
    else:
        val_dataset, num_classes, dataset_size = read_dataset(args.data_path, is_eval)
        batch_size = args.batch_size

    if not is_eval:
        sampler = torch.utils.data.RandomSampler(train_dataset)

        batch_sampler_train = torch.utils.data.BatchSampler(sampler,
                                                            batch_size,
                                                            drop_last=True)

        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_sampler=batch_sampler_train,
                                                       num_workers=args.num_workers,
                                                       pin_memory=True)

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)

    if not is_eval:
        return train_dataloader, val_dataloader, num_classes, dataset_size

    else:
        return val_dataloader, num_classes, dataset_size


def load_fashionMNIST(args, device, is_eval=False):
    train_dataset = datasets.FashionMNIST(root='./data', train=True,
                                          download=True, transform=fashion_mnist_transforms['train'])
    num_classes = len(train_dataset.classes)
    dataset_size = len(train_dataset)

    if not is_eval:
        calculated_batch_size = get_batch_size(args, num_classes, device, dataset_size)
        batch_size = min(calculated_batch_size, args.batch_size)
        save_classes(train_dataset.classes, './data/FashionMNIST')
    else:
        batch_size = args.batch_size

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=2)

    val_dataset = datasets.FashionMNIST(root='./data', train=False,
                                        download=True, transform=fashion_mnist_transforms['val'])

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    if not is_eval:
        return train_loader, val_loader, num_classes, dataset_size

    else:
        return train_loader, val_loader, num_classes, len(val_dataset)
