import os
import shutil
from argparse import ArgumentParser
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.autograd import Variable

import transforms
from datasets import DWTDataset
from models import SegmentationNet


parser = ArgumentParser()
parser.add_argument('--seed', type=int,
                    help='random generator seed')
parser.add_argument('--batch_size', type=int, default=10,
                    help='input batch size for training (default: 10)')
parser.add_argument('--val_batch_size', type=int, default=10,
                    help='input batch size for validation (default: 10)')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--log_interval', type=int, default=10,
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--eval', dest='eval', action='store_true',
                    help='evaluate model on validation set')


class DiceCoefficient(object):

    def __init__(self, ignore_index=None, smooth=1e-5):
        self.ignore_index = ignore_index
        self.smooth = smooth

    def __call__(self, input, target):
        probabilities = F.softmax(input, dim=1).data

        encoded_target = torch.zeros_like(input.data)
        encoded_target.scatter_(1, target.data.unsqueeze(1), 1)
        if self.ignore_index is not None:
            encoded_target[:, self.ignore_index] = 0

        intersection = probabilities * encoded_target
        numerator = 2 * intersection.sum(3).sum(2).sum(0) + self.smooth
        denominator = (probabilities + encoded_target).sum(3).sum(2).sum(0) + self.smooth
        dice_coef = numerator / denominator
        return dice_coef


class DiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        probabilities = F.softmax(input, dim=1)

        encoded_target = torch.zeros_like(input.data)
        encoded_target.scatter_(1, target.data.unsqueeze(1), 1)
        if self.ignore_index is not None:
            encoded_target[:, self.ignore_index] = 0
        encoded_target = torch.autograd.Variable(encoded_target, requires_grad=False)

        intersection = probabilities * encoded_target
        numerator = 2 * intersection.sum(3).sum(2).sum(0) + self.smooth
        denominator = (probabilities + encoded_target).sum(3).sum(2).sum(0) + self.smooth
        dice_coef = numerator / denominator

        weight_var = torch.autograd.Variable(self.weight) if not isinstance(self.weight, torch.autograd.Variable) else self.weight
        if weight_var.numel() > 0:
            dice_coef *= weight_var

        return 1 - dice_coef.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.mean()


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_data_loaders(train_batch_size, val_batch_size):
    normalize = transforms.Normalize(mean=torch.Tensor([0.5]), std=torch.Tensor([0.2]))

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.MultiplicativeGaussianNoise(1, 0.01),
        normalize
    ])

    val_transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor(), normalize])

    train_loader = DataLoader(DWTDataset('dataset', split='train', transform=train_transform),
                              batch_size=train_batch_size, shuffle=True)

    val_loader = DataLoader(DWTDataset('dataset', split='valid', transform=val_transform),
                            batch_size=val_batch_size, shuffle=False)

    return train_loader, val_loader


def main():
    global args
    args = parser.parse_args()

    global cuda
    cuda = torch.cuda.is_available()
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if cuda:
            torch.cuda.manual_seed_all(args.seed)

    print('=> creating model ')
    model = SegmentationNet(in_channels=1, out_channels=2, in_planes=48, depth=3)
    if cuda:
        model = model.cuda()

    criterion = DiceLoss()#FocalLoss(gamma=2)
    if cuda:
        criterion = criterion.cuda()

    accuracy_metric = DiceCoefficient()

    optimizer = Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    best_accuracy = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))

            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_accuracy = checkpoint['best_accuracy']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            print('=> loaded checkpoint {} (epoch {})'.format(
                  args.resume, checkpoint['epoch']))
        else:
            print('=> no checkpoint found at {}'.format(args.resume))

    train_loader, val_loader = get_data_loaders(args.batch_size, args.val_batch_size)

    if args.eval:
        validate(val_loader, model, criterion, accuracy_metric)
        return

    for epoch in range(start_epoch + 1, args.epochs + 1):
        # train for one epoch
        train(train_loader, model, criterion, accuracy_metric, optimizer, epoch)

        if epoch % 1 == 0 or epoch == (args.epochs - 1):
            # evaluate on validation set
            val_loss, val_accuracy = validate(val_loader, model, criterion, accuracy_metric)

            # remember best prec and save checkpoint
            is_best = val_accuracy > best_accuracy
            best_accuracy = max(val_accuracy, best_accuracy)
            save_checkpoint({
                'epoch': epoch,
                'best_accuracy': best_accuracy,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, is_best)


def train(train_loader, model, criterion, accuracy_metric, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, mask, _, _, _) in enumerate(train_loader):
        if cuda:
            input = input.cuda()
            mask = mask.cuda(async=True)

        input_var, mask_var = Variable(input), Variable(mask.long())

        # compute output
        output = model(input_var)

        loss = criterion(output, mask_var)
        accuracy = accuracy_metric(output, mask_var).mean()

        # record loss and accuracy
        losses.update(loss.data[0], input.size(0))
        accuracies.update(accuracy, input.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0:
            print('Epoch: [{}][{}/{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric {accuracy.val:.4f} ({accuracy.avg:.4f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time, loss=losses, accuracy=accuracies))


def validate(val_loader, model, criterion, accuracy_metric):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, mask, _, _, _) in enumerate(val_loader):
        if cuda:
            input = input.cuda()
            mask = mask.cuda(async=True)

        input_var, mask_var = Variable(input, volatile=True), Variable(mask.long(), volatile=True)

        # compute output
        output = model(input_var)

        loss = criterion(output, mask_var)
        accuracy = accuracy_metric(output, mask_var).mean()

        # record loss and accuracy
        losses.update(loss.data[0], input.size(0))
        accuracies.update(accuracy, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric {accuracy.val:.4f} ({accuracy.avg:.4f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses, accuracy=accuracies))

    print('Validation Results - Loss {loss.avg:.4f}\tMetric {accuracy.avg:.4f}'.format(loss=losses, accuracy=accuracies))

    return losses.avg, accuracies.avg


def save_checkpoint(state, is_best, filename='checkpoint.pt'):
    torch.save(state, filename)
    if is_best:
        print('=> best checkpoint')
        shutil.copyfile(filename, 'model_best.pt')


if __name__ == '__main__':
    main()
