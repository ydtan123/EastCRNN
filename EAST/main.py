#!/usr/bin/python
import argparse
import torch
from torch.autograd import Variable
import os
from torch import nn
from torch.optim import lr_scheduler
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torchvision import transforms
import logging
from model import East
from loss import *
from data_utils import custom_dset, collate_fn
import time
from tensorboardX import SummaryWriter
from utils.init import *
from utils.util import *
from utils.save import *
from utils.myzip import *
import torch.backends.cudnn as cudnn
from eval_old import predict
from hmean import compute_hmean
import zipfile
import glob
import warnings
import numpy as np


def train(train_loader, model, criterion, scheduler, optimizer, epoch, config):
    start = time.time()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    model.train()
    log_file = os.path.join(config["result"], "log_loss.txt")
    
    for i, (img, score_map, geo_map, training_mask) in enumerate(train_loader):
        data_time.update(time.time() - end)

        img, score_map, geo_map, training_mask = img.cuda(), score_map.cuda(), geo_map.cuda(), training_mask.cuda()

        f_score, f_geometry = model(img)
        loss1 = criterion(score_map, f_score, geo_map, f_geometry, training_mask)
        losses.update(loss1.item(), img.size(0))
        
        # backward
        scheduler.step()
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config["print_freq"] == 0:
            logging.debug('Training, Epoch: [{0}][{1}/{2}] Loss {loss.val:.4f} Avg Loss {loss.avg:.4f}'.format(
                epoch, i, len(train_loader), loss=losses))

        save_loss_info(losses, epoch, i, train_loader, log_file)

        
def configure(args):
    config_file = args.config if (args.config != "") else "config"
    cfg = __import__(config_file)
    config = vars(cfg)
    
    config["dataroot"] = cfg.dataroot if (args.data_root == "") else args.data_root

    if (not os.path.exists(config["dataroot"])):
        logging.error("Cannot find data set: {}".format(config["dataroot"]))
        sys.exit(0)
        
    config["result"] = os.path.abspath(cfg.result)
    if not os.path.exists(config["result"]):
        os.mkdir(config["result"])
    logging.debug("Import config from {}".format(config_file))
    return config


def load_checkpoint(config, model, optimizer):
    weightpath = os.path.abspath(config["checkpoint"])
    checkpoint = torch.load(weightpath)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    logging.debug("Loaded checkpoint from {}".format(weightpath))
    return start_epoch


def model_init(config):
    train_root_path = os.path.abspath(os.path.join(config["dataroot"], 'train'))
    train_img = os.path.join(train_root_path, 'img')
    train_gt  = os.path.join(train_root_path, 'gt')

    trainset = custom_dset(train_img, train_gt)
    train_loader = DataLoader(
        trainset, batch_size=config["train_batch_size_per_gpu"] * config["gpu"],
        shuffle=True, collate_fn=collate_fn, num_workers=config["num_workers"])
 
    logging.debug('Data loader created: Batch_size:{}, GPU {}:({})'.format(
        config["train_batch_size_per_gpu"] * config["gpu"], config["gpu"], config["gpu_ids"]))

    # Model
    model = East()
    model = nn.DataParallel(model, device_ids=config["gpu_ids"])
    model = model.cuda()
    init_weights(model, init_type=config["init_type"])
    logging.debug("Model initiated, init type: {}".format(config["init_type"]))
    
    cudnn.benchmark = True
    criterion = LossFunc()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.94)
    
    # init or resume
    if config["resume"] and os.path.isfile(config["checkpoint"]):
        start_epoch = load_checkpoint(config, model, optimizer)
    else:
        start_epoch = 0
    logging.debug("Model is running...")
    return model, criterion, optimizer, scheduler, train_loader, start_epoch

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--data-root", type=str, default="", help='dataset root')
    parser.add_argument("-c", "--config", type=str, default="", help='specify the config file')
    parser.add_argument("-L", "--log", type=str, help='set logging level', default="DEBUG")
    args = parser.parse_args()

    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.log)
    logging.basicConfig(level=numeric_level)

    is_debug = logging.root.level == logging.DEBUG
    config = configure(args)

    hmean = .0
    is_best = False

    warnings.simplefilter('ignore', np.RankWarning)
    model, criterion, optimizer, scheduler, train_loader, start_epoch = model_init(config)
    
    for epoch in range(start_epoch, config["max_epochs"]):

        train(train_loader, model, criterion, scheduler, optimizer, epoch, config)

        if epoch % config["eval_iteration"] == 0:

            # create res_file and img_with_box
            output_txt_dir_path = predict(config, model, criterion, epoch)

            # Zip file
            MyZip(output_txt_dir_path, epoch)

            # submit and compute Hmean
            hmean_ = compute_hmean(os.path.join(config["result"], "submit.zip"))

            if hmean_ > hmean:
                is_best = True
                hmean = hmean_

            state = {
                    'epoch'      : epoch,
                    'state_dict' : model.state_dict(),
                    'optimizer'  : optimizer.state_dict(),
                    'is_best'    : is_best,
                    }
            save_checkpoint(state, epoch)


if __name__ == "__main__":
    main()
