#!/usr/bin/python
import cv2
import argparse
import math
import os
import pathlib
import sys

import numpy as np


def increase_contrast(img_file):

    img = cv2.imread(img_file, 1)
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    cv2.imwrite(pathlib.Path(img_file).stem + "_new.jpg", final)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, help="Path to the input images", default='')
    args = parser.parse_args()

    increase_contrast(args.image)
