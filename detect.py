#!/usr/bin/python
import argparse
import cv2
import logging
import os
import pathlib

from EAST.eval import EASTPredictor
from CRNN.predict import CRNNPredictor
from CRNN.tool.image_boxes import group_by_y
from CRNN.tool.image_boxes import get_gt_symbols


class Detector(object):
    def __init__(self, east_weight, crnn_weight, output):
        if (not os.path.exists(output)):
            os.mkdir(output)
        self.east = EASTPredictor(0.0001, east_weight, output)
        self.crnn = CRNNPredictor(crnn_weight)


    def draw_boxes(self, letters, img_file):
        im = cv2.imread(img_file)
        for l in letters:
            cv2.rectangle(im, (l[1], l[2]), (l[3], l[4]), (255, 0, 0), 1)
        cv2.imwrite(img_file, im)


    def recognize_only(self, img_file, gt_file):
        gt_group_strings = []
        if (gt_file is not None):
            gt_letters = get_gt_symbols(gt_file, im.shape[1], im.shape[0])
            gt_groups = group_by_y(gt_letters)
            for g in gt_groups:
                gt_group_strings.append(''.join([s[0] for s in g]))

        if (gt_letters is not None):
            groups = group_by_y(sorted(gt_letters, key=lambda x: x[1]))
            g, p = self.crnn.predict_one_image(im, groups, args.word_length)
        else:
            p = None
        return gt_group_strings, p, gt_letters
        
    def detect_one_file(self, img_file, gt_file=None):
        letters, im = self.east.predict_one(img_file)
        gt_group_strings = []
        if (gt_file is not None):
            gt_letters = get_gt_symbols(gt_file, im.shape[1], im.shape[0])
            gt_groups = group_by_y(gt_letters)
            for g in gt_groups:
                gt_group_strings.append(''.join([s[0] for s in g]))

        if (letters is not None):
            groups = group_by_y(sorted(letters, key=lambda x: x[1]))
            g, p = self.crnn.predict_one_image(im, groups, args.word_length)
        else:
            p = None
        return gt_group_strings, p, gt_letters

    
def is_contained(gt, pred):
    """check if gt is contained in prediction"""
    for g in gt:
        if not g in p:
            return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--east-weight", type=str, help='specify the weight file for EAST')
    parser.add_argument("--crnn-weight", type=str, help='specify the weight file for CRNN')
    parser.add_argument("-i", "--image", type=str, help='specify the image to predict', default='')
    parser.add_argument("-l", "--image-list", type=str, help='specify the list of images to predict', default='')
    parser.add_argument("-o", "--output", type=str, help='specify the directory to save result files')
    parser.add_argument("-L", "--log", type=str, help='set logging level', default="WARNING")
    parser.add_argument("-w", "--word-length", type=int, help='length of a unit for prediction', default=4)
    parser.add_argument("-n", "--number", type=int, help='the number of images to predict', default=10)
    args = parser.parse_args()

    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.log)
    logging.basicConfig(level=numeric_level)

    detector = Detector(args.east_weight, args.crnn_weight, args.output)
    logging.debug("Detector created")
    
    if (args.image != ''):
        gt_file = str(pathlib.Path(args.image).with_suffix('.txt'))
        if (not os.path.exists(gt_file)):
            gt_file = None
        g, p, letters = detector.detect_one_file(args.image, gt_file)
        if (logging.root.level == logging.DEBUG):
            detector.draw_boxes(letters, os.path.join(args.output, pathlib.Path(args.image).stem + "_with_box.jpg"))
        gstring = '-'.join(g)
        pstring = '-'.join(p)
        if (is_contained(g, p)):
            match = "Match"
        else:
            match = "Wrong"
        print("{} gt: {} => pred: {} , {}".format(match, gstring, pstring, args.image))

    if (args.image_list != ''):
        total_count = 0
        wrong = 0
        with open(args.image_list) as f:
            for l in f:
                total_count += 1
                fname = l.strip()
                gt_file = str(pathlib.Path(fname).with_suffix('.txt'))
                if (not os.path.exists(gt_file)):
                    gt_file = None
                g, p, letters = detector.detect_one_file(fname, gt_file)
                if (logging.root.level == logging.DEBUG):
                    detector.draw_boxes(letters, os.path.join(args.output, pathlib.Path(args.image).stem + "_with_box.jpg"))

                gstring = '-'.join(g)
                if (p is not None):                    
                    pstring = '-'.join(p)
                if (p is None or not is_contained(g, p)):
                    wrong += 1
                    match = "Wrong"
                else:
                    match = "Match"
                print("{} gt: {} => pred: {} , {}".format(match, gstring, pstring, fname))
                if (total_count > args.number):
                    break
        print("Total images: {}, error rate: {}".format(total_count, float(wrong) / total_count))
            
                                            


