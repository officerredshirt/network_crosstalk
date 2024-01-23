from numpy import *

import shelve
import dill
import manage_db
import plot_db
import sys, argparse
import os
import pandas
from pandarallel import pandarallel

pandarallel.initialize()

def main(argv):
    parser = argparse.ArgumentParser(
            prog = "convert_hdf_to_parquet",
            description = "",
            epilog = "")
    parser.add_argument("resfile")

    args = parser.parse_args()
    resfile = args.resfile

    if os.path.exists(resfile):
        df = pandas.read_hdf(resfile,"df")
    else:
        print(f"error: {resfile} does not exist")
        sys.exit()

    print("Converting multidimensional arrays to lists of lists...")
    def ndim_to_lol(row,name):
        return row[name].tolist()
    multidim_columns = ["output_error"]
    for name in multidim_columns:
        print(f"  {name}")
        df[name] = df.parallel_apply(lambda x: ndim_to_lol(x,name),axis=1)

    print("Compressing sparse matrices...")
    def sparsify(row,name):
        return row[name].nonzero()
    sparse_columns = ["T","R","G"]
    for name in sparse_columns:
        print(f"  {name}")
        df[name] = df.parallel_apply(lambda x: sparsify(x,name),axis=1)

    print("Saving...")
    df.to_parquet("par_combined_res.pq")


if __name__ == "__main__":
    main(sys.argv[1:])
    exit(0)
