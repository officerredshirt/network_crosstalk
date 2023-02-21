#!/usr/bin/env python
# coding: utf-8
from numpy import *
from multiprocess import Pool
import random as ran
import shelve
import sys, argparse
import manage_db
# import time

from params import *
from boolarr import *


# here logical vectors are often treated as binary strings and represented as integers up to 2^VEC_LENGTH
def main(argv):

    parser = argparse.ArgumentParser(
            prog = "ss_get_achievable_patterns",
            description = "",
            epilog = "")
    parser.add_argument("-i","--filename_in",required=True)
    parser.add_argument("-d","--database",default="temp.db")

    args = parser.parse_args()
    filename_in = args.filename_in
    database = args.database

    # load architecture
    local_id = manage_db.extract_local_id(filename_in)
    R, T, G = manage_db.get_network(database,local_id)

    achieved_pattern = zeros(M_GENE)
    achieved_pattern[0:N_ON] = 1
    
    u = zeros(N_TF + N_PF)
    if sc == 1:
        u[0:N_ON] = 1
    elif sc == 2 or sc == 3:
        u[0] = 1

    manage_db.add_pattern(database,local_id,u,achieved_pattern)


    # -- SAVE PROOF OF COMPLETION FOR SNAKEMAKE FLOW -- #
    with open(filename_in + ".achieved","w") as file:
        pass


if __name__ == "__main__":
    main(sys.argv[1:])
    exit(0)
