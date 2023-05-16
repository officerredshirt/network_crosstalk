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
            prog = "test_df",
            description = "",
            epilog = "")
    parser.add_argument("databases",nargs='*')

    args = parser.parse_args()
    databases = args.databases

    db_filenames = []
    for database in databases:
        ext = os.path.splitext(database)[1]
        if ext == "":
            print("no database file provided--defaulting to local_db.db in " + database)
            database = database + "local_db.db"
        elif not(ext == ".db"):
            print(database + " is not a valid database file")
            sys.exit(1)

        assert os.path.exists(database), "database " + database + " does not exist"
        db_filenames.append(database)

    df = plot_db.combine_databases(db_filenames)
    #print(df.columns.values.tolist())
    #print(df.loc[df["K_NS"] == 10000])
    #gb = df.groupby(["K_NS","M_GENE"],group_keys=True)
    #print(dict(list(gb)))
    #print(gb.get_group((10000,150)))
    plot_db.boxplot_groupby("test_patt_err.png",df.loc[df["minimize_noncognate_binding"] == 0],["tf_first_layer","K_NS","M_GENE"],plot_db.error_by_gene)
    plot_db.boxplot_groupby("test_noncog.png",df.loc[df["minimize_noncognate_binding"] == 1],["tf_first_layer","K_NS","M_GENE"],plot_db.error_by_gene)

    """
    print("PARAMETERS")
    manage_db.print_res(database,"parameters")
    print("")

    print("NETWORKS")
    manage_db.print_res(database,"networks")
    print("")

    print("PATTERNS")
    manage_db.print_res(database,"patterns")
    print("")
    
    print("XTALK")
    manage_db.print_res(database,"xtalk")
    print("")
    """


if __name__ == "__main__":
    main(sys.argv[1:])
    exit(0)
