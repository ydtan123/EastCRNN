#!/usr/bin/python

import argparse

""" return the lines that are in the first, but not in the second file"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-1", "--first", type=str, help="first file", default='')
    parser.add_argument("-2", "--second", type=str, help="second file", default='')
    parser.add_argument("-o", "--output", type=str, help="output file", default='')
    args = parser.parse_args()

    second = {}
    with open(args.second) as f:
        for l in f:
            fname = l.strip().split("/")[-1].split(" ")[0]
            second[fname] = 1

    diff = []
    with open(args.first) as first, open(args.output, "w+") as o:
        for l in first:
            line = l.strip()
            fname = line.split("/")[-1]
            if (not fname in second):
                o.write("{}\n".format(line))
