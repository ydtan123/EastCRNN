#!/usr/bin/python3
import argparse
import math
import os
import pathlib
import sys

import numpy as np
import cv2

sys.path.append("../common")
sys.path.append("../mnist")
from image_boxes import get_random_segments
from create_samples import MNISTSample


def get_boudary(symbols):
    tx = symbols[0][1]
    ty = min([s[2] for s in symbols])
    bx = symbols[-1][3]
    by = max([s[4] for s in symbols])
    return tx, ty, bx, by


def writeGT(words, filename):
    with open(filename, 'w+') as fgt:
        for gidx, idx, symbols in words:
            tx, ty, bx, by = get_boudary(symbols)
            fgt.write("{0},{1},{2},{3},{4},{5},{6},{7},{8}\n".format(
                tx, ty, bx, ty, bx, by, tx, by, ''.join([s[0] for s in symbols])))


def gen_test_gt_zip(test_root, result_root):
    os.system("zip {} {}".format(os.path.join(result_root, "gt.zip"), os.path.join(test_root, "*.txt")))


def gen_from_images(image_list, train_img_root, train_gt_root, train_img_with_box, test_root, train_size, test_size):
    file_dict = {}
    max_width = 0
#    for f in pathlib.Path(args["image"]).glob("**/*.jpg"):
    count = 0
    with open(image_list) as ff:
        for l in ff:
            fname = l.strip()
            f = pathlib.Path(fname)
            txtfile = str(f.with_suffix(".txt"))
            if (not os.path.isfile(txtfile)):
                print("GT file for {0} does not exist".format(f))
                continue
            
            img = cv2.imread(str(f))
            h, w, _ = img.shape
            if (w > max_width):
                max_width = w
            words, _ = get_random_segments(txtfile, img, max_length=5, is_digit=True, debug=False)

            if (len(words) == 0):
                print("Non-digit image {0}".format(f))
                continue
        
            if (trn < train_size):
                writeGT(words, "{0}/img_{1}.txt".format(train_gt_root, trn))
                cv2.imwrite("{0}/img_{1}.jpg".format(train_img_root,trn), img)
                file_dict[f] = "train_{0}".format(trn)
                count += 1
                if (count % 1000 == 0):
                    print("Train: {0} -> {1}".format(f, trn))

                for _, _, symbols in words:
                    tx, ty, bx, by = get_boudary(symbols)
                    cv2.rectangle(img, (tx, ty), (bx, by),(0,255,0),1)
                    cv2.imwrite("{0}/img_{1}.jpg".format(train_img_with_box,trn), img)
                trn += 1
            elif (tst < test_size):
                writeGT(words, "{0}/gt_img_{1}.txt".format(test_root, tst))
                cv2.imwrite("{0}/img_{1}.jpg".format(test_root, tst), img)
                print("Test: {0} -> {1}".format(f, tst))
                file_dict[f] = "test_{0}".format(tst)
                tst += 1
            if (trn >= train_size and tst >= test_size):
                break

            if (count % 1000 == 0):
                print("max width of all training images is : {0}".format(max_width))
            with open(os.path.join(args['data_root'], "filelog.txt"), "w+") as filelog:
                for f, idx in file_dict.items():
                    filelog.write("{0} {1}\n".format(f, idx))


def gen_from_mnist(mnist_root, img_root, gt_root, img_with_box, number_of_samples, is_test=False):
    mn = MNISTSample(mnist_root)
    for i in range(number_of_samples):
        gt, im = mn.gen_random_img(data_len=5)
        gt_prefix = 'gt_' if is_test else ''
        writeGT(gt, "{0}/{1}img_{2}.txt".format(gt_root, gt_prefix, i))
        cv2.imwrite("{0}/img_{1}.jpg".format(img_root, i), im)
        if (is_test):
            for _, _, symbols in gt:
                tx, ty, bx, by = get_boudary(symbols)
                cv2.rectangle(im, (tx, ty), (bx, by),(0,255,0),1)
                cv2.imwrite("{0}/img_{1}.jpg".format(img_with_box, i), im)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, help="the file that has a list of images", default='')
    parser.add_argument("-m", "--mnist", type=str, help="gen random images from mnist data", default='')
    parser.add_argument("-d", "--debug", action='store_true', help="Debug mode", default=0)
    parser.add_argument("-p", "--prepare", type=str, help="Prepare training and testing sets", default='10,10')
    parser.add_argument("-a", "--data-root", type=str, help='dataset root')
    parser.add_argument("-r", "--result-root", type=str, help='result root')
    
    args = vars(parser.parse_args())
    
    d = args["prepare"].split(',')
    train_size, test_size = int(d[0]), int(d[1])
    print("Generating {} train sampels, {} test samples".format(train_size, test_size))
    train_img_root = os.path.join(args['data_root'], "train/img")
    train_gt_root = os.path.join(args['data_root'], "train/gt")
    train_img_with_box = os.path.join(args['data_root'], "train/img_with_box")
    test_root = os.path.join(args['data_root'], "test")
    if not os.path.exists(args['data_root']):
        os.makedirs(test_root)
        os.makedirs(train_img_with_box)
        os.makedirs(train_gt_root)
        os.makedirs(train_img_root)
    
    if not os.path.exists(args["result_root"]):
        os.makedirs(args["result_root"])

    if (args["image"] != ''):
        gen_from_images(args["image"], train_img_root, train_gt_root, train_img_with_box, test_root, train_size, test_size)
    
    if (args["mnist"] != ''):
        gen_from_mnist(args["mnist"], train_img_root, train_gt_root, train_img_with_box,  train_size)
        gen_from_mnist(args["mnist"], test_root, test_root, train_img_with_box,  train_size)

    # generate gt.zip of test files and copy to the result directory
    os.system("zip -j {} {}".format(os.path.join(args["result_root"], "gt.zip"), os.path.join(test_root, "*.txt")))