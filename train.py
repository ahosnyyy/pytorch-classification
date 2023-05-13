import os
import time
import math
import argparse

import torch
import torch.optim as optim
import torch.cuda.amp as amp
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset.dataset import build_dataloader, load_fashionMNIST
from models import build_model
from utils.misc import ModelEMA, calc_accuracy


def parse_args():
    parser = argparse.ArgumentParser()
    # basic
    parser.add_argument('--cuda', action='store_true', default=False, help='Use CUDA for inference if available.')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for inference (default: 128)')

    parser.add_argument('--wp_epoch', type=int, default=20,
                        help='Number of warmup epochs (default: 20)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='Starting epoch for training (default: 0)')
    parser.add_argument('--max_epoch', type=int, default=300,
                        help='Maximum number of training epochs (default: 300)')
    parser.add_argument('--eval_epoch', type=int, default=1,
                        help='Evaluate the model every eval_epoch epochs during training (default: 1)')

    parser.add_argument('--num_workers', type=int, default=8, help='number of worker threads for data loading.')

    parser.add_argument('--base_lr', type=float,
                        default=4e-3, help='Base learning rate for training the model (default: 4e-3)')
    parser.add_argument('--min_lr', type=float,
                        default=1e-6, help='Minimum learning rate for training the model (default: 1e-6)')
    parser.add_argument('--path_to_save', type=str,
                        default='weights/', help='Path to save the trained model weights (default: weights/)')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='Enable tensorboard visualization during training (default: False)')

    parser.add_argument('--fp16', dest="fp16", action="store_true", default=False,
                        help="Use mixed precision training to reduce memory usage and speed up training.")

    # optimization
    parser.add_argument('--optimizer', type=str, default='adamw',
                        help='Choose optimizer: sgd, adam (default: adamw)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay value (default: 0.05)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum value for SGD (default: 0.9)')
    parser.add_argument('--accumulation', type=int, default=1,
                        help='Number of steps for gradient accumulation (default: 1)')

    # Model
    parser.add_argument('-m', '--model', type=str, default='resnet18',
                        help='model architecture to evaluate (default: resnet18)')

    parser.add_argument('-p', '--pretrained', action='store_true', default=False,
                        help='Use ImageNet pretrained weights.')
    parser.add_argument('--norm_type', type=str, default='BN',
                        help='Type of normalization layer to use.')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the checkpoint to resume training from.')
    parser.add_argument('--ema', action='store_true', default=False,
                        help='Use Exponential Moving Average (EMA) during training to stabilize the training process.')

    # dataset
    parser.add_argument('--dataset', type=str, default=None,
                        help='path to dataset directory. The directory should contain subdirectories for each class '
                             'and the images for each class should be located in their corresponding subdirectories.')

    return parser.parse_args()


def main():
    args = parse_args()
    print(f'\n{args}')

    if args.dataset is not None:
        dataset_name = os.path.basename(os.path.normpath(args.dataset))
    else:
        dataset_name = "FashionMNIST"

    if args.fp16:
        path_to_save = os.path.join(args.path_to_save, dataset_name, args.model + '_fp16')
    else:
        path_to_save = os.path.join(args.path_to_save, dataset_name, args.model)

    os.makedirs(path_to_save, exist_ok=True)

    # amp
    scaler = amp.GradScaler(enabled=args.fp16)

    # use gpu
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print('CUDA is not available, using cpu instead.')

    # tensorboard
    if args.tensorboard:
        c_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
        if args.fp16:
            log_path = os.path.join('logs/', dataset_name, args.model + '_fp16', c_time)
        else:
            log_path = os.path.join('logs/', dataset_name, args.model, c_time)

        print(f'\nUsing TensorBoard, you can run tensorboard --logdir={log_path} to see logs.')
        os.makedirs(log_path, exist_ok=True)
        tb_logger = SummaryWriter(log_path)

    if args.dataset is not None:
        train_loader, val_loader, num_classes, dataset_size = build_dataloader(args, device)
    else:
        train_loader, val_loader, num_classes, dataset_size = load_fashionMNIST(args, device)

    # model
    model, loaded_epoch = build_model(
        model_name=args.model,
        pretrained=args.pretrained,
        num_classes=num_classes,
        resume=args.resume
    )

    model.train().to(device)

    # basic config
    best_acc = -1.
    base_lr = args.base_lr
    min_lr = args.min_lr
    tmp_lr = base_lr
    epoch_size = len(train_loader)
    wp_iter = len(train_loader) * args.wp_epoch
    total_epochs = args.max_epoch + args.wp_epoch
    lr_schedule = True
    warmup = True

    # optimizer
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=base_lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            momentum=args.momentum,
            lr=base_lr,
            weight_decay=args.weight_decay
        )

    # EMA
    if args.ema:
        ema = ModelEMA(model, args.start_epoch * epoch_size)
    else:
        ema = None

    # loss
    criterion = torch.nn.CrossEntropyLoss().to(device)

    if args.resume:
        start_epoch = loaded_epoch
    else:
        start_epoch = args.start_epoch

    print(f"\nStart training from epoch {start_epoch} ...")
    for epoch in range(start_epoch, total_epochs):
        if not warmup:
            # use cos lr decay
            T_max = total_epochs - 15
            if epoch + 1 > T_max and lr_schedule:
                print('\tCosine annealing is over.')
                lr_schedule = False
                set_lr(optimizer, min_lr)

            if lr_schedule:
                # Cosine Annealing Scheduler
                tmp_lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * epoch / T_max))
                set_lr(optimizer, tmp_lr)

        train_pbar = tqdm(total=len(train_loader))

        # train one epoch
        for iter_i, (images, target) in enumerate(train_loader):
            ni = iter_i + epoch * epoch_size

            # warmup
            if ni < wp_iter and warmup:
                alpha = ni / wp_iter
                warmup_factor = 0.00066667 * (1 - alpha) + alpha
                tmp_lr = base_lr * warmup_factor
                set_lr(optimizer, tmp_lr)
            elif ni >= wp_iter and warmup:
                print('\tWarmup is over.')
                warmup = False
                set_lr(optimizer, base_lr)

            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # inference
            if args.fp16:
                with torch.cuda.amp.autocast(enabled=args.fp16):
                    output = model(images)
            else:
                output = model(images)

            # loss
            loss = criterion(output, target)

            # check loss
            if torch.isnan(loss):
                continue

            # accuracy
            acc1, acc5 = calc_accuracy(output, target, num_classes, device)

            # bp
            if args.fp16:
                # Backward and Optimize
                loss /= args.accumulation
                scaler.scale(loss).backward()

                # Optimize
                if ni % args.accumulation == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss /= args.accumulation
                loss.backward()

                if ni % args.accumulation == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            # ema
            if args.ema:
                ema.update(model)

            train_pbar.update(1)
            if num_classes >= 5:
                train_pbar.set_description(f"Training: Epoch {epoch}/{total_epochs}, Loss {loss:.4f}, "
                                           f"Acc@1 {acc1:.4f}, Acc@5 {acc5:.4f}")
            else:
                train_pbar.set_description(f"Training: Epoch {epoch}/{total_epochs}, Loss {loss:.4f}, "
                                           f"Acc@1 {acc1:.4f}")

        if args.tensorboard:
            # viz loss
            tb_logger.add_scalar('loss', loss.item() * args.accumulation, epoch)
            tb_logger.add_scalar('Acc@1', acc1.item(), epoch)
            if num_classes >= 5:
                tb_logger.add_scalar('Acc@5', acc1.item(), epoch)

        train_pbar.close()

        # evaluate
        if (epoch % args.eval_epoch) == 0 or (epoch == total_epochs - 1):
            model_eval = ema.ema if args.ema else model

            val_loss, val_acc1, val_acc5 = validate(device=device, val_loader=val_loader, model=model_eval,
                                                    criterion=criterion, num_classes=num_classes)

            if num_classes >= 5:
                print(f'Epoch {epoch} summary: training loss:{loss:.4f}, Acc@1 {acc1:.4f}, '
                      f' Acc@5 {acc5:.4f}, validation loss {val_loss:.4f},'
                      f' Acc@1 {val_acc1:.4f}, Acc@5 {acc5:.4f}')
            else:
                print(f'Epoch {epoch} summary: training loss:{loss:.4f}, Acc@1 {acc1:.4f}, '
                      f'validation loss {val_loss:.4f}, Acc@1 {val_acc1:.4f}')

            is_best = val_acc1 > best_acc
            best_acc = max(val_acc1, best_acc)

            if epoch % 10 == 0:
                print(f'Saving checkpoint at epoch {epoch} ...')
                weight_name = '{}_epoch_{}.pth'.format(args.model, epoch)
                checkpoint_path = os.path.join(path_to_save, weight_name)
                torch.save({'model': model_eval.state_dict()},
                           checkpoint_path)
            if is_best:
                print(f'New best Acc@1 {acc1:.4f}, saving the model ...')
                checkpoint_path = os.path.join(path_to_save, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model': model_eval.state_dict(),
                    'loss': val_loss,
                }, checkpoint_path)


def validate(device, val_loader, model, criterion, num_classes):
    # switch to evaluate mode
    model.eval()
    acc_num_pos1 = 0.
    acc_num_pos5 = 0.
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
                val_pbar.set_description(f"Validation: Loss {loss:.4f}, Acc@1 {cur_acc1:.4f}, Acc@5 {cur_acc5:.4f}")
            else:
                val_pbar.set_description(f"Validation: Loss {loss:.4f}, Acc@1 {cur_acc1:.4f}")

            # Count the number of positive samples
            bs = images.shape[0]
            count += bs
            acc_num_pos1 += cur_acc1 * bs
            acc_num_pos5 += cur_acc1 * bs

        # top1 acc
        acc1 = acc_num_pos1 / count
        acc5 = acc_num_pos5 / count

    # switch to train mode
    model.train()

    return loss, acc1, acc5


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    main()
