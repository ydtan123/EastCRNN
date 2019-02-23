#!/usr/bin/env python3

import random
import argparse
import numpy as np
from mnist import MNIST
import cv2
import random
import sys

sys.path.append("../common")
from image_boxes import noisy


class MNISTSample(object):
    def __init__(self, dataroot):
        mn = MNIST(dataroot, return_type='numpy')
        self.images, self.label = mn.load_training()
        self.dataset_size = len(self.images)

    def gen_random_img(self, data_len=0):
        if (data_len == 0):
            data_len = random.randint(1, 10)
        digits = random.sample(range(self.dataset_size), data_len)
        img = np.zeros((28, 28 * data_len), dtype=np.uint8)
    
        xstart = 0
        xratio = 0.5
        labels = []
        for i in range(data_len):
            displacement = 0 if i==0 else random.randint(1, 5)  # move to left
            distortion = random.random() * xratio + (1 - xratio)  # compress horizontally
            xstart = xstart - displacement
            xlen = int(28 * distortion)
            xend = xstart + xlen
            labels.append((str(self.label[digits[i]]), xstart + 1, 1, xend - 1, 26))
            #new_img = self.images[digits[i]].copy()
            img[:, xstart : xend] = cv2.resize(self.images[digits[i]], (xlen, 28))
            xstart = xend
        gray = cv2.bitwise_not(noisy("gauss", img))
        im = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return [(0, 0, labels)], im


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default=None, type=int,
                        help="ID (position) of the letter to show")
    parser.add_argument("--training", action="store_true",
                        help="Use training set instead of testing set")

    parser.add_argument("--data", default="./data",
                        help="Path to MNIST data dir")

    args = parser.parse_args()

    mn = MNIST(args.data, return_type='numpy')

    image_gen = MNISTSample(args.data)
    label, img = image_gen.gen_random_img()
    print('Showing num: {}'.format(label))
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #print(mn.display(img[which]))
