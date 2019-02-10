#!/usr/bin/python
import argparse
import cv2
import os
import lmdb # install lmdb by "pip install lmdb"
import logging
import math
import numpy as np
import pathlib
import random
import sys

from image_boxes import gen_samples

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    try:
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    except:
        print("something is wrong in checkImageIsValid")
        return False
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.iteritems():
            txn.put(k, v)


def createDB(dataPath, outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(os.path.join(dataPath, outputPath), map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = os.path.join(dataPath, imagePathList[i])
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


def get_image_list(filename):
    file_list = []
    with open(filename) as f:
        for l in f:
            file_list.append(l.strip())
    return file_list


def createDataSet(image_files, data_root, dataset_name, db=True, length=10):
    print("Total number of images: {}".format(len(image_files)))
    max_len = 0
    count = 0
    for ff in image_files:
        #pathlib.Path(args["image"]).glob("**/*.jpg"):
        f = pathlib.Path(ff)
        txtfile = f.with_suffix(".txt")
        if (not os.path.isfile(str(txtfile))):
            print("GT file for {0} does not exist".format(f))
            continue

        img_labels, mlen = gen_samples(txtfile, f, data_root, length)

        if (mlen > max_len):
            max_len = mlen
        if (len(img_labels) == 0):
            print("Did not find labels in {0}".format(f))
            continue
        with open(os.path.join(data_root, dataset_name + "_labels.txt"), "a+") as lf:
            for l in img_labels:
                lf.write("{} {}\n".format(l[0], ''.join(l[1])))
        count += 1
        if (count % 1000 == 0):
            print("{} files in {} processed".format(count, dataset_name))

    with open(os.path.join(data_root, dataset_name + ".txt"), "w+") as f:
        for l in image_files:
            f.write("{0}\n".format(l))

    if (not db):
        return
    
    img_list = []
    label_list = []
    with open(os.path.join(data_root, dataset_name + "_labels.txt"), 'r') as f:
        for l in f:
            img, label = l.strip().split(' ')
            img = img.strip()
            label = label.strip()
            if (img == '' or label == ''):
                continue
            img_list.append(img)
            label_list.append(label)
    createDB(data_root, dataset_name + '.lmdb', img_list, label_list)
    return 

if __name__ == '__main__':
    ## Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, help="Path to one input image", default='')
    parser.add_argument("-f", "--file-list", type=str, help="list of image files", default='')

    parser.add_argument("-d", "--debug", action='store_true', help="Debug mode", default=0)
    parser.add_argument("-n", "--number", type=int, help="the number of training images", default=100000)
    parser.add_argument("-t", "--tests", type=int, help="the number of test images", default=100000)
    parser.add_argument("-l", "--len", type=int, help="the max length of words", default=10)

    parser.add_argument("-a", "--data-root", type=str, help='dataset root')
    args = parser.parse_args()

    if not os.path.exists(args.data_root):
        os.makedirs(args.data_root)

    if (args.image != ''):
        createDataSet([args.image], args.data_root, "tmp", db=False)
        sys.exit(0)
    
    img_files = get_image_list(args.file_list)
    logging.debug("Found {} images".format(len(img_files)))
    if (args.number + args.tests > len(img_files)):
        logging.error("The number of images are less than needed")
        sys.exit(0)
        
    random.shuffle(img_files)
    createDataSet(img_files[ : args.number], args.data_root, "trainset", length=args.len)

    last_sample = min(args.number + args.tests, len(img_files))
    createDataSet(img_files[args.number : last_sample], args.data_root, "valset", length=args.len) 
        
