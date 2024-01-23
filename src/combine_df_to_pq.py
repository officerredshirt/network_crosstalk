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
            prog = "combine_df_to_pq",
            description = "",
            epilog = "")
    parser.add_argument("-o","--outfile",required=False,default="../combined_res.pq")
    parser.add_argument("databases",nargs='*')

    args = parser.parse_args()
    databases = args.databases

    COMBINED_RESULTS = args.outfile

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
        df = pandas.read_parquet(COMBINED_RESULTS)
        df = plot_db.combine_databases(db_filenames,df=df)
    else:
        df = plot_db.combine_databases(db_filenames)
    df = df.loc[df["K_NS"] > 100]
    df = df.loc[df["success"] == 1]
    #df.reset_index(inplace=True)
    ix_to_drop = df.loc[:,df.columns != "filename"].astype(str).drop_duplicates().index
    print(f"Dropping {len(df)-len(ix_to_drop)} duplicate indices...")
    df = df.loc[ix_to_drop]
    df = plot_db.calc_modulating_concentrations(df)

    print("Saving...")
    df.to_parquet(COMBINED_RESULTS)
    print(df)


if __name__ == "__main__":
    main(sys.argv[1:])
    exit(0)
