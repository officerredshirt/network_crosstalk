from numpy import *

import shelve
import dill
import sqlite3
import manage_db
import sys, argparse
import os

def main(argv):
    parser = argparse.ArgumentParser(
            prog = "plot_combinatorial",
            description = "",
            epilog = "")
    parser.add_argument("database")
    parser.add_argument("folder_out")

    args = parser.parse_args()
    database = args.database
    folder_out = args.folder_out

    ext = os.path.splitext(database)[1]
    if ext == "":
        print("no database file provided--defaulting to local_db.db in " + database)
        database = database + "local_db.db"
    elif not(ext == ".db"):
        print(database + " is not a valid database file")
        sys.exit(1)

    assert os.path.exists(database), "database " + database + " does not exist"

    manage_db.plot_xtalk_results(database,folder_out)


if __name__ == "__main__":
    main(sys.argv[1:])
    exit(0)
