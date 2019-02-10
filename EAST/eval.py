#!/usr/bin/python
# -*- coding:utf-8 -*-

import argparse
import cv2
import time
import math
import os
import pathlib
import numpy as np
from PIL import Image
import locality_aware_nms as nms_locality
import lanms
import shutil
import torch
import logging
import model
from data_utils import restore_rectangle, polygon_area
from torch.autograd import Variable
import sys
from torchvision import transforms
import model
from utils.init import *
from torch import nn
from torch.optim import lr_scheduler
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torchvision import transforms
from model import East
from loss import *
import torch.backends.cudnn as cudnn


class EASTPredictor(object):
    def __init__(self, lr, weight_path, output_path):
        self.output_path = output_path
        self.model = East()
        self.model = nn.DataParallel(self.model, device_ids=[0])
        self.model = self.model.cuda()
        init_weights(self.model, init_type='xavier')
        cudnn.benchmark = True
    
        self.criterion = LossFunc()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.94)
        self.weightpath = os.path.abspath(weight_path)
        logging.debug("EAST <==> Prepare <==> Loading checkpoint '{}' <==> Begin".format(self.weightpath))
        checkpoint = torch.load(self.weightpath)

        self.start_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        logging.debug("EAST <==> Prepare <==> Loading checkpoint '{}', epoch={} <==> Done".format(self.weightpath, self.start_epoch))
        self.model.eval()

    def resize_image(self, im, max_side_len=2400):
        '''
        resize image to a size multiple of 32 which is required by the network
        :param im: the resized image
        :param max_side_len: limit of max image size to avoid out of memory in gpu
        :return: the resized image and the resize ratio
        '''
        h, w, _ = im.shape
    
        resize_w = w
        resize_h = h
    
        resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
        resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
        
        #resize_h, resize_w = 512, 512
        im = cv2.resize(im, (int(resize_w), int(resize_h)))
    
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
    
        return im, (ratio_h, ratio_w)
    
    def detect(self, score_map, geo_map, score_map_thresh=1e-5, box_thresh=1e-8, nms_thres=0.1):
        '''
        restore text boxes from score map and geo map
        :param score_map:
        :param geo_map:
        :param score_map_thresh: threshhold for score map
        :param box_thresh: threshhold for boxes
        :param nms_thres: threshold for nms
        :return:
        '''
        if len(score_map.shape) == 4:
            score_map = score_map[0, :, :, 0]
            geo_map = geo_map[0, :, :, ]
        # filter the score map
        xy_text = np.argwhere(score_map > score_map_thresh)
        # sort the text boxes via the y axis
        xy_text = xy_text[np.argsort(xy_text[:, 0])]
        # restore
        start = time.time()
        text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
        logging.debug('{} text boxes before nms'.format(text_box_restored.shape[0]))
        boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
        boxes[:, :8] = text_box_restored.reshape((-1, 8))
        boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
        # nms part
        start = time.time()
        # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
        logging.debug('{} boxes before merging'.format(boxes.shape[0]))
        boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
        if boxes.shape[0] == 0:
            return None
        logging.debug('{} boxes before checking scores'.format(boxes.shape[0]))
        # here we filter some low score boxes by the average score map, this is different from the orginal paper
        for i, box in enumerate(boxes):
            mask = np.zeros_like(score_map, dtype=np.uint8)
            cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
            boxes[i, 8] = cv2.mean(score_map, mask)[0]
        boxes = boxes[boxes[:, 8] > box_thresh]
        return boxes
    
    
    def sort_poly(self, p):
        min_axis = np.argmin(np.sum(p, axis=1))
        p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
        if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
            return p
        else:
            return p[[0, 3, 2, 1]]
    
    
    def predict_one(self, img_file):
        im = cv2.imread(img_file)[:, :, ::-1]
    
        im_resized, (ratio_h, ratio_w) = self.resize_image(im)
        im_resized = im_resized.astype(np.float32)
        im_resized = im_resized.transpose(2, 0, 1)
        im_resized = torch.from_numpy(im_resized)
        im_resized = im_resized.cuda()
        im_resized = im_resized.unsqueeze(0)
    
        score, geometry = self.model(im_resized)
    
        score = score.permute(0, 2, 3, 1)
        geometry = geometry.permute(0, 2, 3, 1)
        score = score.data.cpu().numpy()
        geometry = geometry.data.cpu().numpy()

        boxes = self.detect(score_map=score, geo_map=geometry)

        letters = None
        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h
            logging.debug("found {} boxes".format(len(boxes)))
            fstem = pathlib.Path(img_file).stem
            letters, im = self.save_boxes(os.path.join(self.output_path, fstem + "_boxes.txt"), im, boxes)
            cv2.imwrite(os.path.join(self.output_path, fstem + "_with_box.jpg"), im[:, :, ::-1])
        else:
            logging.debug("Did not find boxes")
    
        return letters, im
    
    
    def save_boxes(self, filename, im, boxes):
        letters = []
        with open(filename, 'w+') as f:
            for box in boxes:
                box = self.sort_poly(box.astype(np.int32))
    
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                    logging.debug('wrong direction')
                    continue
                
                #if box[0, 0] < 0 or box[0, 1] < 0 or box[1,0] < 0 or box[1,1] < 0 or box[2,0]<0 or box[2,1]<0 or box[3,0] < 0 or box[3,1]<0:
                #    logging.debug("wrong box, {}".format(box))
                #    continue
                for x in range(4):
                    for y in [0, 1]:
                        if (box[x, y] < 0):
                            box[x, y] = 0
                    
                poly = np.array([[box[0, 0], box[0, 1]], [box[1, 0], box[1, 1]], [box[2, 0], box[2, 1]], [box[3, 0], box[3, 1]]])
                
                p_area = polygon_area(poly)
                if p_area > 0:
                    poly = poly[(0, 3, 2, 1), :]
    
                f.write('{},{},{},{},{},{},{},{}\r\n'
                        .format(poly[0, 0], poly[0, 1], poly[1, 0], poly[1, 1], poly[2, 0], poly[2, 1], poly[3, 0], poly[3, 1],))
                cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
                letters.append(('', poly[0, 0], poly[0, 1], poly[2, 0], poly[2, 1]))
        return letters, im


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="", help='specify the config file')
    parser.add_argument("-w", "--weight", type=str, help='specify the weight file')
    parser.add_argument("-i", "--image", type=str, help='specify the image to predict', default='')
    parser.add_argument("-l", "--image-list", type=str, help='specify the list of images to predict', default='')
    parser.add_argument("-o", "--output", type=str, help='specify the directory to save result files')
    parser.add_argument("-L", "--log", type=str, help='set logging level', default="DEBUG")
    parser.add_argument("-n", "--number", type=int, help='the number of images to predict', default=10)
    parser.add_argument("--lr", type=float, help='learning rate', default=0.0001)

    args = vars(parser.parse_args())

    numeric_level = getattr(logging, args["log"].upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args["log"])
    logging.basicConfig(level=numeric_level)

    predictor = EASTPredictor(args["lr"], args["weight"], args["output"])
    
    if (args["image"] != ''):
        predictor.predict_one(args["image"])

    count = 0
    if (args["image_list"] != ''):
        with open(args["image_list"]) as f:
            for l in f:
                predictor.predict_one(l.strip())
                count += 1
                if (args["number"] <= count):
                    break
