#!/usr/bin/env python
# coding: utf-8
from numpy import *
import shelve
import sys, getopt
import os
# import time

from params import *

def print_usage():
    print("usage is: get_networks.py -n <num_networks> -p <filename_prefix>") 


def main(argv):
    num_networks = 1
    fnprefix = ""

    try:
        opts, args = getopt.getopt(argv,"hn:p:")
    except getopt.GetoptError:
        print_usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print_usage()
            sys.exit()
        elif opt == "-n":
            num_networks = int(arg)
        elif opt == "-p":
            fnprefix = arg

    # call gen_network n times
    # tic = time.perf_counter()
    for fn in range(num_networks):
        os.system("python3 ./src/gen_network.py -o " + fnprefix + f"{fn:06}")
    # toc = time.perf_counter()
    # print(f"elapsed time: {toc-tic} s, approx {(toc-tic)/num_networks} per network")

if __name__ == "__main__":
    main(sys.argv[1:])
