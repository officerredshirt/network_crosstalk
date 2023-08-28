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
            prog = "check_hdf",
            description = "",
            epilog = "")
    parser.add_argument("filename")

    args = parser.parse_args()
    filename = args.filename

    if os.path.exists(filename):
        df = pandas.read_hdf(filename,key="df")
    else:
        print(f"file {filename} not found")
        return

    print(df.iloc[-1])
    print(df.columns.values)


if __name__ == "__main__":
    main(sys.argv[1:])
    exit(0)
