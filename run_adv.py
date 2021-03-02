import os
import sys
import shutil
import time
import random
import copy
import json
import argparse

import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as utils
from torch.utils.tensorboard import SummaryWriter
import lib.models as models
from lib.utils.utils import *
from lib.utils.radam import RAdam
from lib.datasets.data_utils import generate_dataset
from lib.adversarial.adversarial import *
from lib.adversarial.tvm import TotalVarMin

from art.classifiers import PyTorchClassifier
import art.defences as defences
import numpy as np
import foolbox as fb
import eagerpy as ep

# get list of valid models from custom models directory
model_names = sorted(name for name in models.__dict__
  if name.islower() and not name.startswith("__")
  and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Adversarial Training Benchmarking',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str,
                    default='/nobackup/users/jzpan/datasets/aptos2019', help='path to dataset')
parser.add_argument('--dataset', type=str,
                    help='choose dataset to benchmark adversarial training techniques on.')
parser.add_argument('--arch', metavar='ARCH', default='resnet50',
                    help='model architecture: to evaluate robustness on (default: resnet50)')
parser.add_argument('--workers', type=int, default=16,
                    help='number of data loading workers to use')
parser.add_argument('--pretrained', type=str, default='',
                    help='path to pretrained model')
parser.add_argument('--gpu_ids', type=str, default='0,1,2,3',
                    help='list of gpu ids to use for acceleration (-1 for cpu only)')
# Hyperparameter for Adversarial Retrainings
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to perform adversarial training for')
parser.add_argument('--optimizer', type=str,
                    default='sgd', help='optimizer to use')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float,
                    default=0.1, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float,
                    default=0.0001, help='weight decay')
parser.add_argument('--schedule', type=int, nargs='+',
                    default=[83, 123], help='list of epochs to reduce lr at')
parser.add_argument('--gammas', type=float, nargs='+', default=[
                    0.1, 0.1], help='list of gammas to multiply with lr at each scheduled epoch; length of gammas should be the same as length of schedule')
parser.add_argument('--pretrained_adv', action='store_true', help='whether to use pretrained adv_trained model')

# Model checkpoint flags
parser.add_argument('--print_freq', type=int, default=200,
                    metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='checkpoints',
                    help='Folder to save checkpoints and log.')
parser.add_argument('--resume', type=str, default=None, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', type=int, default=0,
                    metavar='N', help='manual epoch number (useful on restarts)')

# Experiments
parser.add_argument('--attacks', type=str, nargs='+', default=[
                    'fgsm', 'pgd', 'deepfool', 'bim'], help='list of attacks to evaluate')
parser.add_argument('--epsilons', type=float, nargs='+', default=[2/255, 4/255, 8/255, 16/255], help='epsilon values to use for attacks')
parser.add_argument('--defences', type=str, nargs='+', default=[], help='list of defences to evaluate')
parser.add_argument('--input_size', type=int, default=-1,
                    help='input size for adv training; use -1 to use default input size')
parser.add_argument('--inc_contrast', type=float, default=1, help='factor to increase the dataset contrast')

global best_acc1, best_loss

# Below training loop, train function, and val function are adapted from: https://github.com/pytorch/examples/blob/master/imagenet/main.py


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top2],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inputs, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if '-1' not in args.gpu_ids:
            inputs = inputs.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # compute output
        output = model(inputs)
        loss = criterion(output, target)
        # measure accuracy and record loss
        acc1, acc2 = accuracy(output, target, topk=(1, 2))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top2.update(acc2[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    summary.add_scalar('train acc2', top2.avg, epoch)
    summary.add_scalar('train acc1', top1.avg, epoch)
    summary.add_scalar('train loss', losses.avg, epoch)


def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top2],
        prefix="Test: ")

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (inputs, target) in enumerate(val_loader):
            if '-1' not in args.gpu_ids:
                inputs = inputs.to(f'cuda:{args.gpu_ids}', non_blocking=True)
                target = target.to(f'cuda:{args.gpu_ids}', non_blocking=True)

            # compute output
            output = model(inputs)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc2 = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top2.update(acc2[0], inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f}'
             .format(top1=top1))

    summary.add_scalar('test acc2', top2.avg, epoch)
    summary.add_scalar('test acc1', top1.avg, epoch)
    summary.add_scalar('test loss', losses.avg, epoch)

    return top1.avg, losses.avg


def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    assert len(gammas) == len(
        schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    args = parser.parse_args()
    best_acc1 = 0
    best_loss = np.iinfo(np.int16).max

    # check that CUDA is actually available and pass in GPU ids, use CPU if not
    if torch.cuda.is_available():
        gpu_id_list = [int(i.strip()) for i in args.gpu_ids.split(',')]
    else:
        gpu_id_list = [-1]

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.isdir(os.path.join(args.save_path, 'adv_trained_model')):
        os.makedirs(os.path.join(args.save_path, 'adv_trained_model'))

    # initialize tensorboard logger
    summary = SummaryWriter(args.save_path)
    print('==> Output path: {}...'.format(args.save_path))

    print(args)

    assert args.arch in model_names, 'Error: model {} not supported'.format(
        args.arch)

    # set variables based on dataset to evaluate on
    input_size = 456 if args.input_size == -1 else args.input_size

    criterion = torch.nn.CrossEntropyLoss()
    train_loader, test_loader, num_classes = generate_dataset(args.data_path, input_size, args.batch_size, args.workers, args.inc_contrast)

    model = models.__dict__[args.arch](num_classes=num_classes)

    assert os.path.isfile(
        args.resume), 'Adversarial benchmarking requires a pretrained model — use train_models.py to train a model'
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    # checkpointed models are wrapped in a nn.DataParallel, so rename the keys in the state_dict to match
    try:
        # our checkpoints save more than just the state_dict, but other checkpoints may only save the state dict, causing a KeyError
        checkpoint['state_dict'] = {
             n.replace('module.', ''): v for n, v in checkpoint['state_dict'].items()}
        model.load_state_dict(checkpoint['state_dict'])
    except KeyError:
        model.load_state_dict(checkpoint)
    print("=> loaded checkpoint '{}'" .format(args.resume))
    print(model)

    # wrap model in DataParallel for multi-gpu support
    if -1 not in gpu_id_list:
        model = torch.nn.DataParallel(model, device_ids=gpu_id_list)

    # define optimizer and set optimizer hyperparameters

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate,
                    weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=5, factor=0.5, verbose=True)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate, momentum=args.momentum,
                    weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
       optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum,
                    weight_decay=args.weight_decay, nesterov=True)
    elif args.optimizer == 'radam':
        optimizer = RAdam(model.parameters(), args.learning_rate,
                    weight_decay=args.weight_decay)

    if -1 not in gpu_id_list:
        model.to(f'cuda:{args.gpu_ids}')
        criterion.to(f'cuda:{args.gpu_ids}')

    cudnn.benchmark = True

    input_shape = (3, input_size, input_size)

    # get initial validation set accuracy

    initial_acc, _ = validate(test_loader, model, criterion, 1, args)

    # perform attacks and defences on dataset

    attack_list = {}
    defence_list = {}

    # initialize attacks and append to dict

    classifier = fb.PyTorchModel(copy.deepcopy(model).eval(), (0, 1))
    epsilons = args.epsilons

    #white box attacks
    if 'fgsm' in args.attacks:
        attack_list['fgsm'] = fb.attacks.FGSM()
    if 'carliniLinf' in args.attacks:
        # Use ART implementation as Foolbox doesn't have Linf CW attack
        art_classifier = PyTorchClassifier(copy.deepcopy(model), loss=criterion, optimizer=optimizer, input_shape=input_shape, nb_classes=num_classes)
        cw_dict = cw_linf(art_classifier, test_loader, epsilons)

    if 'pgd' in args.attacks:
        attack_list['pgd'] = fb.attacks.PGD()
    if 'deepfool' in args.attacks:
        attack_list['deepfool'] = fb.attacks.LinfDeepFoolAttack()
    if 'bim' in args.attacks:
        attack_list['bim'] = fb.attacks.LinfBasicIterativeAttack()

    # initialize defences and append to dict

    if 'tvm' in args.defences:
        tvm_params = parameter_list['tvm']
        defence_list['tvm'] = TotalVarMin(clip_values=(
            tvm_params['clip_min'], tvm_params['clip_max']), prob=tvm_params['prob'], lamb=tvm_params['lamb'], max_iter=tvm_params['max_iter'])
    if 'jpeg' in args.defences:
        jpeg_params = parameter_list['jpeg']
        defence_list['jpeg'] = defences.JpegCompression(clip_values=(
            jpeg_params['clip_min'], jpeg_params['clip_max']), channel_index=jpeg_params['channel_index'], quality=jpeg_params['quality'])

    initial_acc, _ = validate(test_loader, model, criterion, 1, args)
    print("Initial Model Accuracy ===> {}".format(initial_acc))

    #convert dataloader into an eagerPy tensor for FoolBox attack generation
    adv_dict = gen_attacks(test_loader,
                           classifier, attack_list, epsilons, args.gpu_ids)

    # loop through all generated dataloaders with adversarial images
    results_dict = {}
    for attack_name in adv_dict:

        # measure attack success
        print("Testing performance of attack {}: ".format(attack_name))
        for epsilon_attack, epsilon in zip(adv_dict[attack_name], epsilons):
            attacked_acc, _ = validate(
                epsilon_attack, model, criterion, 1, args)

            # save adv images for visualization purposes
            dataiter = iter(epsilon_attack)
            images, _ = dataiter.next()
            img_grid = utils.make_grid(images)
            summary.add_image("Training Images Adversarially Attacked Using {} with eps {}".format(
                    attack_name, epsilon), img_grid)

            print("Generating defences for attack {} with eps {}: ".format(attack_name, epsilon))

            def_adv_dict = gen_defences(epsilon_attack, attack_name, defence_list)
            accuracies = {'initial': initial_acc.item(
            ), 'attacked': attacked_acc.item()}

            for def_name in def_adv_dict:
                print("Testing performance of defence {}: ".format(def_name))
                top1, _ = validate(def_adv_dict[def_name], model, criterion, 1, args)
                def_images , _ = zip(*[batch for batch in def_adv_dict[def_name]])
                def_images = torch.cat(def_images).numpy()

                accuracies[def_name] = top1.item()

            results_dict[attack_name + ' eps {}'.format(epsilon)] = accuracies
        print(results_dict)

    with open(os.path.join(args.save_path, 'results.json'), 'w') as save_file:
        json.dump(results_dict, save_file)
