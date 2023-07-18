from numpy import *

import shelve
import dill
import manage_db
import plot_db
import sys, argparse
import os
import pandas

def main(argv):
    parser = argparse.ArgumentParser(
            prog = "add_col_to_hdf",
            description = "",
            epilog = "")
    parser.add_argument("hdf_filename",nargs='*')

    args = parser.parse_args()
    hdf_filenmae = args.hdf_filename

    hdf_filename = "../combined_res.hdf"

    if os.path.exists(hdf_filename):
        df = pandas.read_hdf(hdf_filename,key="df")
        #df["ratio_KNS_KS"] = df["K_NS"]/df["K_S"]
        df["layer1_static"] = True
    else:
        print(f"error: nonexistent hdf file {hdf_filename}")
        return

    df.to_hdf(hdf_filename,key="df",mode="w")
    print(df)


if __name__ == "__main__":
    main(sys.argv[1:])
    exit(0)
