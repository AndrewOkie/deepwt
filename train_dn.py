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
from models import UniversalNet


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


class AngularLoss(nn.Module):
    def __init__(self):
        super(AngularLoss, self).__init__()

    def forward(self, input, target, mask, weight):
        input = F.normalize(input, dim=1) * 0.999999
        target = F.normalize(target, dim=1) * 0.999999

        angle_error = torch.acos((input * target).sum(dim=1))
        loss = (angle_error * angle_error * weight).mean(dim=0).mean()
        return loss


class ExceedingAngleThreshold(nn.Module):
    def __init__(self, threshold):
        super(ExceedingAngleThreshold, self).__init__()
        self.threshold = threshold

    def forward(self, input, target, mask):
        input = F.normalize(input, dim=1) * 0.999999
        target = F.normalize(target, dim=1) * 0.999999

        angle_error = torch.acos((input * target).sum(dim=1)) * mask
        exceed = (self.threshold / 180 * 3.14159265359 < angle_error).float().sum(dim=2).sum(dim=1) / (mask.sum(dim=2).sum(dim=1) + 1)
        return exceed.mean()


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
    model = UniversalNet(in_channels=2, out_channels=2, in_planes=32, depth=3)
    if cuda:
        model = model.cuda()

    criterion = AngularLoss()
    if cuda:
        criterion = criterion.cuda()

    accuracy_metric = ExceedingAngleThreshold(10.0)
    if cuda:
        accuracy_metric = accuracy_metric.cuda()

    optimizer = Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    best_accuracy = 1e10

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
            val_loss = validate(val_loader, model, criterion, accuracy_metric)

            # remember best prec and save checkpoint
            is_best = val_loss < best_accuracy
            best_accuracy = min(val_loss, best_accuracy)
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
    for i, (image, mask, _, gradient, weight) in enumerate(train_loader):
        if cuda:
            image = image.cuda()
            mask = mask.cuda()
            gradient = gradient.cuda(async=True)
            weight = weight.cuda(async=True)

        image_var, mask_var = Variable(image), Variable(mask)
        gradient_var, weight_var = Variable(gradient, requires_grad=False), Variable(weight, requires_grad=False)
        
        input = torch.cat((image, mask.unsqueeze_(1)), dim=1)
        input_var = Variable(input)

        # compute output
        output = model(input_var)
        output = F.normalize(output, dim=1)

        loss = criterion(output, gradient_var, mask_var, weight_var)
        accuracy = accuracy_metric(output, gradient_var, mask_var)

        # record loss and accuracy
        losses.update(loss.data[0], input.size(0))
        accuracies.update(accuracy.data[0], input.size(0))

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
    for i, (image, mask, _, gradient, weight) in enumerate(val_loader):
        if cuda:
            image = image.cuda()
            mask = mask.cuda()
            gradient = gradient.cuda(async=True)
            weight = weight.cuda(async=True)

        image_var, mask_var = Variable(image, volatile=True), Variable(mask, volatile=True)
        gradient_var, weight_var = Variable(gradient, volatile=True), Variable(weight, volatile=True)

        input = torch.cat((image, mask.unsqueeze_(1)), dim=1)
        input_var = Variable(input)

        # compute output
        output = model(input_var)
        output = F.normalize(output, dim=1)

        loss = criterion(output, gradient_var, mask_var, weight_var)
        accuracy = accuracy_metric(output, gradient_var, mask_var)

        # record loss and accuracy
        losses.update(loss.data[0], input.size(0))
        accuracies.update(accuracy.data[0], input.size(0))

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

    return losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pt'):
    torch.save(state, filename)
    if is_best:
        print('=> best checkpoint')
        shutil.copyfile(filename, 'model_best.pt')


if __name__ == '__main__':
    main()
