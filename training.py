import os
import sys
import yaml
import time
import shutil
import torch
import random
import argparse
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from torch.utils import data
from tqdm import tqdm
from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loader 
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from pdb import set_trace
from dataset import RoboticsDataset
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    RandomRotate90,
    Normalize,
    Compose,
    PadIfNeeded,
    RandomCrop,
    CenterCrop,
    HueSaturationValue,
    RandomBrightnessContrast,
    ElasticTransform,
)
import pandas as pd
import seaborn as sns
import time
from datetime import datetime
import re
from itertools import chain
from torch.nn import Module, Conv2d, ConvTranspose2d
import torch.nn.init as init
from torch.autograd import Variable

def count_conv2d(module: Module):
    """
    Counts the number of convolutions and transposed convolutions in a Module
    """
    return len([m for m in module.modules() if isinstance(m, Conv2d) or isinstance(m, ConvTranspose2d)])

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        # init.xavier_uniform(m.weight, gain=np.sqrt(2.0))
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        init.constant(m.bias,0.0)

def train(cfg, writer, logger):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup seeds
    torch.manual_seed(cfg.get('seed', 999))
    np.random.seed(999)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(999)

    n_classes = 7 # the number of tissue subtype classes + 1
    logger.info("n_classes: {}".format(n_classes))
    
    tile_size = 1024
    problem_type = 'tissue'
    workers = 12
    batch_size = cfg['training']['batch_size']
    device_ids='0'
    
    fileEpochLoss = open(os.path.join(writer.file_writer.get_logdir(),'epoch_loss_train_seg.txt'),'w')
    fileEpochLossVal = open(os.path.join(writer.file_writer.get_logdir(),'epoch_loss_val_seg.txt'),'w')
    fileLR = open(os.path.join(writer.file_writer.get_logdir(),'lr.txt'),'w')
    fileMiou = open(os.path.join(writer.file_writer.get_logdir(),'miou.txt'),'w')

    
    def make_loader(file_names, shuffle=False, transform=None, problem_type='tissue', batch_size=batch_size):
        return DataLoader(
            dataset=RoboticsDataset(file_names, transform=transform, problem_type=problem_type),
            shuffle=shuffle,
            num_workers=workers,
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available())
    
    def train_transform(p=1):
        return Compose([
            RandomCrop(height=tile_size, width=tile_size, p=1),
            RandomRotate90(p=0.5),
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            HueSaturationValue(hue_shift_limit=20,sat_shift_limit=30,val_shift_limit=0,p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.25,0.25),contrast_limit=(0.25,1.75),p=0.5),
            ElasticTransform(alpha=1,sigma=4),
            # Normalize(p=1)
        ], p=p)

    def val_transform(p=1):
        return Compose([
            CenterCrop(height=tile_size, width=tile_size, p=1),
            # Normalize(p=1)
        ], p=p)
    
    train_file = "train_tiles.txt" # the list of training patches
    with open(train_file) as f:
        train_file_names = [line.rstrip('\n') for line in f]

    val_file = "val_tiles.txt" # the list of validation patches
    with open(val_file) as f:
        val_file_names = [line.rstrip('\n') for line in f]
        
    trainloader = make_loader(train_file_names, shuffle=True, transform=train_transform(p=1), problem_type=problem_type,
                           batch_size=batch_size)
    valloader = make_loader(val_file_names, transform=val_transform(p=1), problem_type=problem_type,
                               batch_size=len(device_ids))

    num_train_files = len(train_file_names)
    num_val_files = len(val_file_names)
    logger.info('num_train = {}, num_val = {}'.format(num_train_files, num_val_files))

    # Setup Metrics
    running_metrics_val = runningScore(n_classes)

    # Setup Model
    model = get_model(cfg['model'], n_classes).to(device)
    model.apply(init_weights)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info("pytorch_total_params {}".format(pytorch_total_params))
    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("pytorch_trainable_params {}".format(pytorch_trainable_params))
    logger.info("Model Layers {}".format(count_conv2d(model)))
    

    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k:v for k, v in cfg['training']['optimizer'].items() 
                        if k != 'name'}

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg['training']['lr_schedule'])

    loss_fn = get_loss_function(cfg)
    logger.info("Using loss {}".format(loss_fn))

    # weighted cross entropy
    # Note class 0 is unannotated regions and will not contribute to the loss function.
    d = {1: 1089821796.0, 2: 919491139.0, 3: 1161456798.0, 4: 1175882130.0, 5: 1170302499.0, 6: 1335408670.0} # the number of pixels of each class in the training set
    d_sum = sum(d.values())
    class_weights = [0, 1-d[1]/d_sum, 1-d[2]/d_sum, 1-d[3]/d_sum, 1-d[4]/d_sum, 1-d[5]/d_sum, 1-d[6]/d_sum]

    start_time=datetime.now()
    start_iter = 0
    if cfg['training']['resume'] is not None:
        if os.path.isfile(cfg['training']['resume']):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg['training']['resume'])
            )
            checkpoint = torch.load(cfg['training']['resume'])
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_iter = checkpoint["epoch"]
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg['training']['resume'], checkpoint["epoch"]
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(cfg['training']['resume']))

    train_loss_meter = averageMeter()
    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    best_iou = -100.0
    i = start_iter
    flag = True
    
    while i <= cfg['training']['train_iters'] and flag:
        for (images, labels) in trainloader:
            i += 1
            
            images_20x = images[:,:,384:640,384:640]
            images_10x = images[:,:,::2,::2]
            images_10x = images_10x[:,:,128:384,128:384]
            images_5x = images[:,:,::4,::4]
            labels_20x = labels[:,384:640,384:640]

            start_ts = time.time()
            scheduler.step()
            model.train()
            images_20x = images_20x.to(device)
            images_10x = images_10x.to(device)
            images_5x = images_5x.to(device)
            labels_20x = labels_20x.to(device)

            images_20x, images_10x, images_5x, labels_20x = Variable(images_20x), Variable(images_10x), Variable(images_5x), Variable(labels_20x)

            outputs = model(images_20x, images_10x, images_5x)
            
            loss = loss_fn(input=outputs, target=labels_20x, class_weights=class_weights)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
                
            time_meter.update(time.time() - start_ts)
            train_loss_meter.update(loss.item())
            if (i + 1) % cfg['training']['print_interval'] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(i + 1,
                                           cfg['training']['train_iters'], 
                                           loss.item(),
                                           time_meter.avg / cfg['training']['batch_size'])

                logger.info(print_str)
                writer.add_scalar('loss/train_loss', loss.item(), i+1)
                time_meter.reset()
                logger.info("avg train loss: " + str(train_loss_meter.avg))
                fileEpochLoss.write(str(train_loss_meter.avg))
                fileEpochLoss.write('\n')
                train_loss_meter.reset()
                for param_group in optimizer.param_groups:
                    logger.info('current_lr  {}'.format(param_group['lr']))
                    fileLR.write(str(param_group['lr']))
                    fileLR.write('\n')

            if (i + 1) % cfg['training']['val_interval'] == 0 or \
               (i + 1) == cfg['training']['train_iters']:
                model.eval()
                with torch.no_grad():
                    for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):

                        images_20x_val = images_val[:,:,384:640,384:640]
                        images_10x_val = images_val[:,:,::2,::2]
                        images_10x_val = images_10x_val[:,:,128:384,128:384]
                        images_5x_val = images_val[:,:,::4,::4]
                        labels_20x_val = labels_val[:,384:640,384:640]

                        images_20x_val = images_20x_val.to(device)
                        images_10x_val = images_10x_val.to(device)
                        images_5x_val = images_5x_val.to(device)
                        labels_20x_val = labels_20x_val.to(device)

                        outputs = model(images_20x_val,images_10x_val,images_5x_val)
                        val_loss = loss_fn(input=outputs, target=labels_20x_val, class_weights=class_weights)

                        pred = outputs.data.max(1)[1].cpu().numpy()
                        gt = labels_20x_val.data.cpu().numpy()

                        running_metrics_val.update(gt, pred)
                        val_loss_meter.update(val_loss.item())

                writer.add_scalar('loss/val_loss', val_loss_meter.avg, i+1)
                logger.info("Iter %d Loss: %.4f" % (i + 1, val_loss_meter.avg))
                fileEpochLossVal.write(str(val_loss_meter.avg))
                fileEpochLossVal.write('\n')

                score, class_iou, hist, mean_iu, recalls, precisions, average_recall, average_precision = running_metrics_val.get_scores()
                for k, v in score.items():
                    print(k, v)
                    logger.info('{}: {}'.format(k, v))
                    writer.add_scalar('val_metrics/{}'.format(k), v, i+1)

                for k, v in class_iou.items():
                    logger.info('{}: {}'.format(k, v))
                    writer.add_scalar('val_metrics/cls_{}'.format(k), v, i+1)

                val_loss_meter.reset()
                running_metrics_val.reset()
                fileMiou.write(str(mean_iu))
                fileMiou.write('\n')
                np.savetxt(os.path.join(writer.file_writer.get_logdir(), "hist.csv"), hist, delimiter=",")
                logger.info('recalls  {}'.format(recalls))
                logger.info('average_recall  {}'.format(average_recall))
                logger.info('precisions  {}'.format(precisions))
                logger.info('average_precision  {}'.format(average_precision))
                
                logger.info('time since start = {}'.format(datetime.now()-start_time))

                if score["Mean IoU : \t"] >= best_iou:
                    best_iou = score["Mean IoU : \t"]
                    state = {
                        "epoch": i + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_iou": best_iou,
                    }
                    save_path = os.path.join(writer.file_writer.get_logdir(),
                                             "{}_{}_best_model.pkl".format(
                                                 cfg['model']['arch'],
                                                 cfg['data']['dataset']))
                    torch.save(state, save_path)
                    logger.info('current_best_iou_value  {}'.format(best_iou))

            if (i + 1) == cfg['training']['train_iters']:
                flag = False
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/DMMN-breast.yml",
        help="Configuration file to use"
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    timestr = time.strftime("%Y%m%d_%H%M%S")
    run_id = random.randint(1,100000)
    folder_name = str(timestr) + "_" +str(cfg['model']['arch'])
    print(folder_name)
    logdir = os.path.join('runs', os.path.basename(args.config)[:-4] , str(folder_name))
    print(logdir)
    writer = SummaryWriter(log_dir=logdir)

    print('RUNDIR: {}'.format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info('Start training:')

    train(cfg, writer, logger)
