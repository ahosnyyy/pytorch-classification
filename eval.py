import json
import os

import torch


import argparse
from tqdm import tqdm

from models import build_model
from utils.misc import calc_accuracy

from dataset.dataset import build_dataloader, load_fashionMNIST


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=False, help='Use CUDA for inference if available.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for inference (default: 128)')
    parser.add_argument('--num_workers', type=int, default=8, help='number of worker threads for data loading.')

    parser.add_argument('-m', '--model', type=str, default='resnet18',
                        help='model name to evaluate (default: resnet18)')

    parser.add_argument('-w', '--weights', type=str, required=True,
                        help='path to model weights file')

    parser.add_argument('-q', '--quantized', action='store_true', default=False,
                        help='whether the model is quantized (default: False)')

    parser.add_argument('-d', '--dataset', type=str, default=None,
                        help='Path to the root directory of the dataset')

    return parser.parse_args()


def dump_json(result, model, is_quantized):
    # Dump the dictionary to a JSON file
    dir_name = os.path.join("results")
    os.makedirs(dir_name, exist_ok=True)

    if is_quantized:
        file_name = os.path.join(dir_name, f"{model}_quantized.json")
    else:
        file_name = os.path.join(dir_name, f"{model}.json")

    with open(file_name, "w") as f:
        json.dump(result, f, indent=4, separators=(',', ':'))


def main():
    args = parse_args()

    # cuda
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print('Using CUDA.')
    else:
        device = torch.device("cpu")
        print('CUDA is not available, using cpu instead.')

    if args.dataset is not None:
        val_loader, num_classes, dataset_size = build_dataloader(args, device, is_eval=True)
    else:
        _, val_loader, num_classes, dataset_size = load_fashionMNIST(args, device, is_eval=True)

    print('Total validation data size: ', dataset_size)

    print('Loading model ...\n')
    # model
    model, _ = build_model(
        model_name=args.model,
        num_classes=num_classes,
    )

    # load checkpoint
    model.load_state_dict(torch.load(args.weights, map_location='cpu')["model"], strict=False)

    model.to(device)
    # loss
    criterion = torch.nn.CrossEntropyLoss().to(device)

    print("Start evaluating ...")
    model.eval()
    acc_num_pos1 = 0.
    acc_num_pos5 = 0.
    acc_loss = 0.
    count = 0.
    with torch.no_grad():
        val_pbar = tqdm(total=len(val_loader))
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # inference
            output = model(images)

            # loss
            loss = criterion(output, target)

            # accuracy
            cur_acc1, cur_acc5 = calc_accuracy(output, target, num_classes, device)

            val_pbar.update(1)
            if num_classes >= 5:
                val_pbar.set_description(f"Loss {loss:.4f}, Acc@1 {cur_acc1:.4f}, Acc@5 {cur_acc5:.4f}")
            else:
                val_pbar.set_description(f"Loss {loss:.4f}, Acc@1 {cur_acc1:.4f}")

            # Count the number of positive samples
            bs = images.shape[0]
            count += bs
            acc_num_pos1 += cur_acc1 * bs
            acc_num_pos5 += cur_acc5 * bs
            acc_loss += loss * bs

        # top1 acc
        acc1 = acc_num_pos1 / count
        acc5 = acc_num_pos5 / count
        loss_ = acc_loss / count
    val_pbar.close()

    if num_classes >= 5:
        print(f'\nEvaluation results: Loss {loss_:.4f}, Acc@1 {acc1:.4f}, Acc@5 {acc5:.4f}')
        res_dict = {"loss": loss_.item(), "Acc@1": acc1.item(), "Acc@5": acc5.item()}
        dump_json(res_dict, args.model, args.quantized)
    else:
        print(f'\nEvaluation results: Loss {loss_:.4f}, Acc@1 {acc1:.4f}')
        res_dict = {"loss": loss_.item(), "Acc@1": acc1.item()}
        dump_json(res_dict, args.model, args.quantized)


if __name__ == "__main__":
    main()
