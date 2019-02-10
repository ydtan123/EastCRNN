#!/usr/bin/python3
import argparse
import cv2
import os
import lmdb # install lmdb by "pip install lmdb"
import math
import numpy as np
import pathlib

files = {}
for f in pathlib.Path("./trainset").glob("**/*.jpg"):
    files[f.name] = 1

print("Found {0} files".format(len(files)))

count = 0
dup = 0
labels = {}
with open("trainset/labels.txt") as f:
    for l in f:
        s = l.split(' ')[0]
        count += 1
        if (s not in files):
            print("Cannot find {} in labels".format(s))
        if (s in labels):
            print("{} duplicated".format(s))
            dup += 1
        labels[s] = 1
print("Total lines in labels: {}, dup: {}, uniq:{}".format(count, dup, count-dup))
