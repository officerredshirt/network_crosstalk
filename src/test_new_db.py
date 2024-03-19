from numpy import *

import shelve
import dill
import manage_db
import plot_db
import sys, argparse
import os
import matplotlib.pyplot as plt
import numpy as np

def main(argv):
    parser = argparse.ArgumentParser(
            prog = "test_new_db",
            description = "",
            epilog = "")
    parser.add_argument("database")

    args = parser.parse_args()
    database = args.database

    ext = os.path.splitext(database)[1]
    if ext == "":
        print("no database file provided--defaulting to local_db.db in " + database)
        database = database + "local_db.db"
    elif not(ext == ".db"):
        print(database + " is not a valid database file")
        sys.exit(1)

    assert os.path.exists(database), "database " + database + " does not exist"


    print("PARAMETERS")
    #manage_db.print_res(database,"parameters")
    print("")

    print("NETWORKS")
    #manage_db.print_res(database,"networks")
    print("")

    print("PATTERNS")
    #manage_db.print_res(database,"patterns")
    print("")
    
    print("XTALK")
    #manage_db.print_res(database,"xtalk")
    print("")

    target_patterns = np.array(list(manage_db.get_target_patterns(database,0)[1])).flatten()
    fig, ax = plt.subplots()
    ax.hist(target_patterns[target_patterns >= 0])
    plt.savefig("target_dist_test.png")



if __name__ == "__main__":
    main(sys.argv[1:])
    exit(0)
