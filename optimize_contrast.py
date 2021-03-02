import skopt
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import tensorflow as tf
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as utils
import lib.models as models
from torch.utils.tensorboard import SummaryWriter

from lib.utils.utils import *
from lib.utils.radam import RAdam
from lib.datasets.data_utils import get_data_statistics, generate_dataset
from lib.adversarial.adversarial import *

import argparse
import numpy as np
import copy
import time
from datetime import timedelta

import foolbox as fb
import eagerpy as ep

#get list of valid models from custom models directory
model_names = sorted(name for name in models.__dict__
          if name.islower() and not name.startswith("__")
            and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Train Models from Scratch', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default='/nobackup/users/jzpan/datasets/aptos2019', help='path to dataset')
parser.add_argument('--arch', metavar='ARCH', default='resnet50', help='model architecture: to evaluate robustness on (default: resnet50)')
parser.add_argument('--workers', type=int, default=16, help='number of data loading workers to use')
parser.add_argument('--pretrained', type=str, default='', help='path to pretrained model')
parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='comma-seperated string of gpu ids to use for acceleration (-1 for cpu only)')
parser.add_argument('--input_size', type=int, default=-1, help='input size of network; use -1 to use default input size')
parser.add_argument('--inc_contrast', type=float, default=1, help='factor to increase contrast for images')
# Hyperparameters
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer to use')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
parser.add_argument('--cosine', action='store_true', help='use cosine annealing schedule to decay learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--schedule', type=int, nargs='+', default=[83,123], help='list of epochs to reduce lr at')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1], help='list of gammas to multiply with lr at each scheduled epoch; length of gammas should be the same as length of schedule')

# Model checkpoint flags
parser.add_argument('--print_freq', type=int, default=200, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='checkpoints', help='Folder to save checkpoints and log.')
parser.add_argument('--train', action='store_true', help='train the model')
parser.add_argument('--resume', type=str, default=None, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

# Attack to defend against
parser.add_argument('--attacks', type=str, default='fgsm', help='attack to defend against')
parser.add_argument('--epsilon', type=float, default=16/255, help='epsilon value of attack to defend against')

global best_acc1, best_loss

image_contrast = Real(low=-2, high=5, prior='uniform', name='robust_image_contrast')
dimensions = [image_contrast]
default_parameters = [2]


if __name__ == '__main__':
    args = parser.parse_args()

    # set gpus ids to use
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    #parse gpu ids from argument
    if torch.cuda.is_available():
        gpu_id_list = [int(i.strip()) for i in args.gpu_ids.split(',')]
    else:
        gpu_id_list = [-1]

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    # initialize tensorboard logger
    summary = SummaryWriter(args.save_path)
    print('==> Output path: {}...'.format(args.save_path))

    print(args)

    assert args.arch in model_names, 'Error: model {} not supported'.format(args.arch)

    # set variables based on dataset to evaluate on
    input_size = 456 if args.input_size == -1 else args.input_size

    criterion = torch.nn.CrossEntropyLoss()
    train_loader, test_loader, _ = generate_dataset(args.data_path, input_size, args.batch_size, args.workers, args.inc_contrast)

    model = models.__dict__[args.arch](num_classes=5)

    assert os.path.isfile(
        args.resume), "Contrast optimization requires a pretrained model — use train_models.py to train a model"


    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    # checkpointed models are wrapped in a nn.DataParallel, so rename the keys in the state_dict to match
    try:
         # our checkpoints save more than just the state_dict, but other checkpoints may only save the state dict, causing a KeyError
        checkpoint['state_dict'] = {n.replace('module.', '') : v for n, v in checkpoint['state_dict'].items()}
        model.load_state_dict(checkpoint['state_dict'])
    except KeyError:
        model.load_state_dict(checkpoint)
    print("=> loaded checkpoint '{}'" .format(args.resume))
    print(model)

    # wrap model in DataParallel for multi-gpu support
    if -1 not in gpu_id_list:
        model = torch.nn.DataParallel(model, device_ids = gpu_id_list)

    # define optimizer and set optimizer hyperparameters

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate,
                    weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=5,factor=0.5,verbose=True)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate, momentum=args.momentum,
                    weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
       optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum,
                    weight_decay=args.weight_decay, nesterov=True)
    elif args.optimizers == 'radam':
        optimizer = RAdam(model.parameters(), args.learning_rate,
                    weight_decay=args.weight_decay)

    if -1 not in gpu_id_list and torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

    @use_named_args(dimensions=dimensions)
    def fitness(robust_image_contrast):

        image_contrast = np.float32(robust_image_contrast)
        val_loader = adv_dict[args.attacks][0]
        #import pdb
        #pdb.set_trace()
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5],
            prefix="Test: ")

        # switch to evaluate mode
        model.eval()

        val_loader.dataset.change_contrast(image_contrast)
        print("Current Contrast: {}".format(image_contrast))

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
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(acc1[0], inputs.size(0))
                top5.update(acc5[0], inputs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i)

            # TODO: this should also be done with the ProgressMeter
            print(' * Acc@1 {top1.avg:.3f}'
                .format(top1=top1))


        return losses.avg

    attack_list = {}

    cudnn.benchmark = True
    input_shape = (3, input_size, input_size)
    args.dataset = 'aptos'
    epsilons = args.epsilon
    epsilons = [epsilons] # epsilons is expected in list format
    classifier = fb.PyTorchModel(copy.deepcopy(model).eval(), (0, 1))

    if 'fgsm' in args.attacks:
        attack_list['fgsm'] = fb.attacks.FGSM()
    if 'pgd' in args.attacks:
        attack_list['pgd'] = fb.attacks.PGD()
    if 'bim' in args.attacks:
        attack_list['bim'] = fb.attacks.LinfBasicIterativeAttack()
    if 'deepfool' in args.attacks:
        attack_list['deepfool'] = fb.attacks.LinfDeepFoolAttack()

    adv_dict = gen_attacks(train_loader,
                           classifier, attack_list, epsilons, args.gpu_ids)
    val_adv_dict = gen_attacks(test_loader,
                           classifier, attack_list, epsilons, args.gpu_ids)

    search_result = gp_minimize(func=fitness,
                                dimensions=dimensions,
                                acq_func='EI', # Expected Improvement.
                                n_calls=100,
                                x0=default_parameters,
                                random_state=46)
    print("Optimal Found Image Contrast: {}".format(search_result.x))
    print("Lowest Loss: {}".format(search_result.fun))
