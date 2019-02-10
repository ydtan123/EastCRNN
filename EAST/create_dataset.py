#!/usr/bin/python3
## Libraries
import argparse
import math
import os
import pathlib
import sys

import numpy as np
import cv2


def writeGT(words, filename, img=None):
    with open(filename, 'w') as fgt:
        for w in words:
            fgt.write("{0},{1},{2},{3},{4},{5},{6},{7},{8}\n".format(w[1], w[2], w[3], w[2], w[3], w[4], w[1], w[4], w[0]))
            if (img is not None):
                print("Rectange {0} {1} {2} {3}".format(w[1], w[2], w[3], w[4]))
                cv2.rectangle(img, (w[1], w[2]), (w[3], w[4]), (0, 255, 0), 1)

                    
def getLocation(txtfile, imgw, imgh, merge):
    #0 0.056711 0.629032 0.030246 0.204301
    symbols = []
    with open(str(txtfile)) as f:
        for line in f:
            data = [d for d in line.split(' ')]
            if (len(data) < 5):
                print("Incorrect data {0} in {1}".format(line, txtfile))
                continue
            dval = int(data[0])
            if (dval < 0 or dval > 9):
                if (dval == 10):
                    print("Skip 10 in {0}".format(txtfile))
                else:
                    print("Invalid number {0} in {1}".format(data[0], txtfile))
                continue
            cx = float(data[1]) * imgw
            cy = float(data[2]) * imgh
            w = float(data[3]) * imgw
            h = float(data[4]) * imgh
            symbols.append((data[0], cx, cy, w/2, h/2))
    if (len(symbols) == 0):
        return None
    symbols_sorted = sorted(symbols, key=lambda x: x[1])
    words = []
    for s in symbols_sorted:
        ty = s[2] - s[4]
        tx = s[1] - s[3]
        by = s[2] + s[4] 
        bx = s[1] + s[3]
        found = False
        if (merge):
            for w in words:
                if (len(w[0]) < 3
                    and math.fabs(ty - w[2]) < s[4] and math.fabs(tx - w[3]) < s[3] * 4):
                    w[0] += s[0]
                    if (ty < w[2]):
                        w[2] = int(ty)
                    w[3] = int(bx)
                    if (by > w[4]):
                        w[4] = int(by)
                    found = True
                    break
        if (not found):
            words.append([s[0], int(tx), int(ty), int(bx), int(by)])
    return words

## Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str, help="Path to the input images", default='')
parser.add_argument("-d", "--debug", action='store_true', help="Debug mode", default=0)
parser.add_argument("-m", "--max", type=int, help="Maximum images to process", default=100000)
parser.add_argument("-p", "--prepare", type=str, help="Prepare training and testing sets", default='')
parser.add_argument("-s", "--save", action='store_true', help="Save text area")
parser.add_argument("-g", "--merge-text", action='store_true', help='Merge nearby text areas')
parser.add_argument("-a", "--data-root", type=str, help='dataset root')
parser.add_argument("-r", "--resize", type=str, default="", help='resize the image')

args = vars(parser.parse_args())

if (args["prepare"] != ''):
    d = args["prepare"].split(',')
    train_size, test_size = int(d[0]), int(d[1])
    trn = 0
    tst = 0

    train_img_root = os.path.join(args['data_root'], "train/img")
    train_gt_root = os.path.join(args['data_root'], "train/gt")
    train_img_with_box = os.path.join(args['data_root'], "train/img_with_box")
    test_root = os.path.join(args['data_root'], "test")
    if not os.path.exists(args['data_root']):
        os.makedirs(test_root)
        os.makedirs(train_img_with_box)
        os.makedirs(train_gt_root)
        os.makedirs(train_img_root)

    file_dict = {}
    max_width = 0
    for f in pathlib.Path(args["image"]).glob("**/*.jpg"):
        if (f in file_dict):
            print("{0} has more than one copy".format(f))
            continue
        if (not os.path.isfile(str(f.with_suffix(".txt")))):
            print("GT file for {0} does not exist".format(f))
            continue

        img = cv2.imread(str(f))
        h, w, _ = img.shape
        if (w > max_width):
            max_width = w
        words = getLocation(f.with_suffix(".txt"), img.shape[1], img.shape[0], args["merge_text"])
        if (words is None):
            print("Non-digit image {0}".format(f))
            continue
        
        new_img = img
        if (args["resize"] != ""):
            new_img = np.zeros((784, 1280, 3), dtype=np.uint8)
            new_img[0:img.shape[0], 0:img.shape[1]] = img

        if (trn < train_size):
            writeGT(
                words, "{0}/img_{1}.txt".format(train_gt_root, trn),
                new_img if args["debug"] else None)
            cv2.imwrite("{0}/img_{1}.jpg".format(train_img_root,trn), new_img)
            print("Train: {0} -> {1}".format(f, trn))
            file_dict[f] = "train_{0}".format(trn)

            for s in words:
                cv2.rectangle(new_img, (s[1], s[2]),(s[3],s[4]),(0,255,0),1)
            cv2.imwrite("{0}/img_{1}.jpg".format(train_img_with_box,trn), new_img)
            trn += 1
        elif (tst < test_size):
            writeGT(words, "{0}/gt_img_{1}.txt".format(test_root, tst))
            cv2.imwrite("{0}/img_{1}.jpg".format(test_root, tst), new_img)
            print("Test: {0} -> {1}".format(f, tst))
            file_dict[f] = "test_{0}".format(tst)
            tst += 1
        if (trn >= train_size and tst >= test_size):
            break
        
    print("max width of all training images is : {0}".format(max_width))
    with open(os.path.join(args['data_root'], "filelog.txt"), "w+") as filelog:
        for f, idx in file_dict.items():
            filelog.write("{0} {1}\n".format(f, idx))

"""
processed = 0
for f in pathlib.Path(args["image"]).glob("*.jpg"):
    if (args["debug"] > 0):
        print("------------Image {0}---------------".format(processed))
        print(f)
    img = cv2.imread(str(f))
    words = getLocation(f.with_suffix(".txt"), img.shape[1], img.shape[0])
    if (args["debug"] > 0):
        for s in words:
            print("{0}: ({1}, {2}), ({3}, {4})".format(s[0], s[1], s[2], s[3], s[4]))
            if(args["save"]):
                cv2.imwrite("text{0}.jpg".format(processed), img[s[2]:s[4], s[1]:s[3]])
            if (args["debug"] > 1):
                cv2.rectangle(img,(s[1], s[2]),(s[3],s[4]),(0,255,0),1)
        if (args["debug"] > 1):
            cv2.imshow("img", img)
            cv2.waitKey(0)
    processed += 1
    if (processed > args["max"]):
        break
"""
