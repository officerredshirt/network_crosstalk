#!/usr/bin/env python
# coding: utf-8
from numpy import *
import shelve
import sys, getopt

from params import *

def print_usage():
    print("usage is: gen_network.py -o <filename_out>")


# TODO:
# [ ] assign connections according to particular distributions
# [ ] correct to ensure all PFs/TFs(/enhancers) are connected to at least one enhancer(/gene)

def main(argv):
    filename_out = "architecture.out"

    try:
        opts, args = getopt.getopt(argv,"ho:")
    except getopt.GetoptError:
        print_usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print_usage()
            sys.exit()
        elif opt == "-o":
            filename_out = arg

    # generate connections from PF to enhancers
    if N_PF != 0:
        R = zeros([M_ENH,N_PF])
        for ii in range(M_ENH):
            R[ii,random.randint(0,N_PF-1)] = 1
    else:
        R = array([])
    
    # generate connections from TFs to enhancers
    T = zeros([M_ENH,N_TF])
    temp = concatenate((ones(THETA),zeros(N_TF - THETA)))
    for ii in range(M_ENH):
        T[ii,] = random.permutation(temp)
    
    # generate connections from enhancers to genes
    # trivial for now (one gene per enhancer)
    G = identity(M_GENE)
    
    
    # -- SAVE ARCHITECTURE TO FILE -- #
    with shelve.open(filename_out + ".arch",'n') as ms:
        ms['R'] = R
        ms['T'] = T
        ms['G'] = G


if __name__ == "__main__":
    main(sys.argv[1:])
