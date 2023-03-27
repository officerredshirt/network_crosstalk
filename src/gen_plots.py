from numpy import *

import manage_db
import sys, argparse
import os

def main(argv):
    parser = argparse.ArgumentParser(
            prog = "gen_plots",
            description = "",
            epilog = "")
    parser.add_argument("folder")

    args = parser.parse_args()
    folder = args.folder
    database = os.path.join(folder,"res","local_db.db")

    assert os.path.exists(database), "database " + database + " does not exist"

    output_folder = os.path.join(folder,"plots")

    manage_db.plot_xtalk_errors(database,output_folder)

if __name__ == "__main__":
    main(sys.argv[1:])
    exit(0)
