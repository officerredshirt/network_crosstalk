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

    COMBINED_RESULTS = "../combined_res.csv"

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

    if os.path.exists(COMBINED_RESULTS):
        df = pandas.read_csv(COMBINED_RESULTS)
        df = plot_db.combine_databases(db_filenames,df=df)
    else:
        df = plot_db.combine_databases(db_filenames)
    df = df.loc[df["K_NS"] > 100]
    df = df.loc[df["success"] == 1]
    df = plot_db.calc_modulating_concentrations(df)
    df.to_csv(COMBINED_RESULTS)

    #plot_db.temp_plot(df)
    prefixes = ['patterning','noncognate_binding']
    for mnb in [0,1]:
        plot_db.subplots_groupby(df.loc[df["minimize_noncognate_binding"] == mnb],"K_NS",["M_GENE","MAX_CLUSTERS_ACTIVE"],plot_db.ratio_xtalk_chromatin_tf_by_pair,filename=f"../plots/ratio_{prefixes[mnb]}_error.png",title=f"log ratio of {prefixes[mnb].replace('_',' ')} error in chromatin to equivalent TF networks")
        plot_db.subplots_groupby(df.loc[df["minimize_noncognate_binding"] == mnb],"K_NS",["M_GENE","MAX_CLUSTERS_ACTIVE","tf_first_layer"],plot_db.xtalk_by_gene,filename=f"../plots/{prefixes[mnb]}_by_gene.png",title=f"log {prefixes[mnb].replace('_',' ')} error per gene")


if __name__ == "__main__":
    main(sys.argv[1:])
    exit(0)
