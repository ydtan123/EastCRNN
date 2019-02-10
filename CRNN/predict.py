#!/usr/bin/python
import argparse
import cv2
import dataset
import logging
import os
import pathlib
from PIL import Image
import random
import sys
import torch
from torch.autograd import Variable
import utils
from tool.image_boxes import get_groups
import models.crnn as crnn


class CRNNPredictor(object):
    
    def __init__(self, model_file, debug=False):
        self.setup_crnn(model_file)
        self.debug = debug

    def set_debug(self, debug):
        self.debug = debug
        
    def predict_one_file(self, fname, word_length):
        imgf = pathlib.Path(fname)
        gtf = imgf.with_suffix(".txt")
        if (not os.path.exists(str(gtf))):
            logging.error("Cannot find gt file %s" % gtf)
            return None, None, None
        img = cv2.imread(fname)
        groups = get_groups(gtf, img.shape[1], img.shape[0], is_digit=True, debug=False)
        gt_list, pred_list = self.predict_one_image(img, groups, word_length)
        if (self.debug and ''.join(gt_list) != ''.join(pred_list)):
            cv2.imwrite(imgf.name + "_debug.jpg", img)
        return gt_list, pred_list

    def predict_one_image(self, image, groups, word_length):
        gt_list = []
        pred_list = []
        for g in groups:
            gt, pred = self.predict_one_group(image, g, word_length)
            gt_list.append(gt)
            pred_list.append(pred)
        return gt_list, pred_list

    def predict_one_group(self, image, group, word_len=4):
        pred = ''
        gt = ''
        for idx in range(0, len(group), word_len):
            last_idx = min(len(group), idx + word_len)
            tx = group[idx][1]
            ty = min([g[2] for g in group[idx : last_idx]])
            bx = group[last_idx - 1][3]
            by = max([g[4] for g in group[idx : last_idx]])
            r, p = self.predict_one_seg(Image.fromarray(image[ty:by+1, tx:bx+1]))
            seg_gt = ''.join([s[0] for s in group[idx : last_idx]])
            cv2.rectangle(image, (tx, ty), (bx, by), (0,255,0), 1)
            logging.debug("Seg{}: {} => {}".format(idx, seg_gt, p))
            gt += seg_gt
            pred += p
        return gt, pred

    def predict_one_seg(self, image):
        #image = Image.open(imagefile).convert('L')
        image = image.convert('L')
        image = self.transformer(image)
        if torch.cuda.is_available():
            image = image.cuda()
        image = image.view(1, *image.size())
        image = Variable(image)
        
        preds = self.model(image)
        
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        raw_pred = self.converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = self.converter.decode(preds.data, preds_size.data, raw=False)
        return raw_pred, sim_pred

    def setup_crnn(self, model_file):
        alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
        
        self.model = crnn.CRNN(32, 1, len(alphabet) + 1, 256)    
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        logging.debug('CRNN: loading pretrained model from %s' % model_file)

        state_dict = torch.load(model_file)
        #hack:  remove `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        # load params
        self.model.load_state_dict(new_state_dict)
        self.model.eval()

        self.converter = utils.strLabelConverter(alphabet)
        self.transformer = dataset.resizeNormalize((100, 32))

    
if __name__ == '__main__':
    ## Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, help="Path to the input images", default='')
    parser.add_argument("-l", "--image-list", type=str, help="the file that lists raw images. One gt file for each image", default='')
    parser.add_argument("-d", "--debug", action='store_true', help="Debug mode")
    parser.add_argument("-m", "--model", type=str, help="the model file")
    parser.add_argument("-n", "--number", type=int, help="the number of images to predict", default=100)
    parser.add_argument("-w", "--word-length", type=int, help="the length of a prediction unit", default=4)

    args = parser.parse_args()

    predictor = CRNNPredictor(args.model, args.debug)
#    model, converter, transformer = setup_crnn(args.model, args.debug)
    
    if (args.image != ''):
        if (not os.path.exists(args.image)):
            logging.error("Cannot find image file %s" % fname)
            sys.exit(0)
        g, p = predictor.predict_one_file(args.image, args.word_length)
        if (g is None):
            sys.exit(0)
        gstring = '-'.join(g)
        pstring = '-'.join(p)
        if (gstring != pstring):
            print('Wrong: %-20s => %-20s, %s' % (gstring, pstring, args.image))
        else:
            print("Match!")
            
    if (args.image_list != ''):
        total_images = 0
        wrong_images = 0
        with open(args.image_list) as f:
            for l in f:
                total_images += 1
                fname = l.strip()
                if (not os.path.exists(fname)):
                    logging.error("Cannot find image file %s" % fname)
                    continue
                g, p = predictor.predict_one_file(fname, args.word_length)
                if (g is None):
                    continue
                gstring = '-'.join(g)
                pstring = '-'.join(p)
                if (gstring != pstring):
                    print('wrong: %-20s => %-20s, %s' % (gstring, pstring, fname))
                    wrong_images += 1
                if (total_images >= args.number):
                    break
        print("Total images: {}, wrong images: {}, error image: {}".format(
            total_images, wrong_images, float(wrong_images)/total_images))
