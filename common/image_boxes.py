import cv2
import logging
import math
import numpy as np
import os
import pathlib
import random
import sys


def line_intersect(a, b):
    return (a[0] <= b[0] and b[0] <= a[1]) or (b[0] <= a[0] and a[0] <= b[1])


def group_by_y(letters):
    groups = []
    for l in sorted(letters, key=lambda x: x[1]):
        found = False
        for g in groups:
            logging.debug("l:{}, g:{}, intersect:{}".format(
                l, g[-1], line_intersect((g[-1][2], g[-1][4]), (l[2], l[4]))))
            if (line_intersect((g[-1][2], g[-1][4]), (l[2], l[4]))):
#                and (math.fabs(l[1] - g[-1][3]) < (l[3] - l[1]) * 2)):
                g.append(l)
                found = True
                break
        if (not found):
            groups.append([l])
    for g in groups:
        logging.debug("Group: l{} {}".format(
            len(g), ''.join([s[0] for s in g])))
    return groups


def get_gt_symbols(txtfile, imgw, imgh, is_digit=True, debug=False):
    symbols = []
    with open(str(txtfile)) as f:
        for line in f:
            data = [d for d in line.split(' ')]
            if (len(data) < 5):
                logging.error(
                    "Incorrect data {0} in {1}".format(line, txtfile))
                continue
            dval = int(data[0])
            if (is_digit and (dval < 0 or dval > 9)):
                logging.debug(
                    "Invalid number {0} in {1}".format(data[0], txtfile))
                continue
            cx = float(data[1]) * imgw
            cy = float(data[2]) * imgh
            w = float(data[3]) * imgw / 2
            h = float(data[4]) * imgh / 2
            symbols.append(
                (data[0], int(cx - w), int(cy - h), int(cx + w), int(cy + h)))
            logging.debug("symbol: {}, ({},{})({},{})".format(
                data[0], int(cx - w), int(cy - h), int(cx + w), int(cy + h)))
    return symbols


def get_groups(txtfile, imgw, imgh, is_digit, debug=False):
    symbols = get_gt_symbols(txtfile, imgw, imgh, is_digit, debug)
    if (len(symbols) == 0):
        logging.error("Cannot find symbols in gt file")
        return []
    return group_by_y(symbols)


def get_segments(txtfile, img, length, is_digit, debug=False):
    imgh, imgw, _ = img.shape
    symbols = get_gt_symbols(txtfile, imgw, imgh, is_digit, debug)
    if (len(symbols) == 0):
        logging.error("Cannot find symbols in gt file")
        return [], 0
    letter_groups = group_by_y(symbols)
    gidx = 0
    max_len = 0
    img_label_list = []
    for g in letter_groups:
        len_g = len(g)
        if (len_g > max_len):
            max_len = len_g
        for idx in range(len_g):
            for word_len in range(1, min(length, len_g - idx) + 1, 1):
                last_idx = idx + word_len - 1
                img_label_list.append(
                    (gidx, idx, [s for s in g[idx: last_idx + 1]]))
        gidx += 1
    return img_label_list, max_len


def get_random_segments(txtfile, img, max_length, is_digit, debug=False):
    imgh, imgw, _ = img.shape
    symbols = get_gt_symbols(txtfile, imgw, imgh, is_digit, debug)
    if (len(symbols) == 0):
        logging.error("Cannot find symbols in gt file")
        return [], 0
    letter_groups = group_by_y(symbols)
    gidx = 0
    max_len = 0
    img_label_list = []
    for g in letter_groups:
        len_g = len(g)
        if (len_g > max_len):
            max_len = len_g
        idx = 0
        while (idx < len_g):
            wlen = random.randint(1, max_length)
            last_idx = min(len_g, idx + wlen) - 1
            img_label_list.append(
                (gidx, idx, [s for s in g[idx: last_idx + 1]]))
            idx = last_idx + 1
        gidx += 1
    return img_label_list, max_len


def gen_samples(txtfile, img_file, data_out, length, is_digit=True, debug=False):
    img = cv2.imread(str(img_file))
    label_info_list, max_len = get_segments(
        txtfile, img, length, is_digit, debug)
    img_label_list = []
    for label in label_info_list:
        g = label[2]
        ty = min([s[2] for s in g])
        tx = g[0][1]
        by = max([s[4] for s in g])
        bx = g[-1][3]
        img_name = 'img_{}_g{}_i{}_l{}.jpg'.format(
            img_file.stem, label[0], label[1], len(g))
        if (not cv2.imwrite(os.path.join(data_out, img_name), img[ty:by+1, tx:bx+1])):
            logging.error("failed to write image file: {}, ({},{}), ({},{})".format(
                img_name, ty, by+1, tx, bx+1))
        img_label_list.append((img_name, [s[0] for s in g]))
    return img_label_list, max_len


def noisy(noise_typ, image):

    """
    The Function adds gaussian , salt-pepper , poisson and speckle noise in an image
    
    Parameters
    ----------
    image : ndarray
        Input image data. Will be converted to float.
    mode : str
        One of the following strings, selecting the type of noise to add:
    
        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        's&p'       Replaces random pixels with 0 or 1.
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n is uniform noise with specified mean & variance.
    """
    if noise_typ == "gauss":
        row, col = image.shape
        mean = 0
        var = 10
        sigma = var**0.5
        gauss = np.absolute(np.random.normal(mean, sigma, (row, col))).astype(np.uint8)
        #gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1
  
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy
