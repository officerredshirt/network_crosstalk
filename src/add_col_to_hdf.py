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
    parser.add_argument("hdf_filename")

    args = parser.parse_args()
    hdf_filename = args.hdf_filename

    #hdf_filename = "../combined_res.hdf"

    if os.path.exists(hdf_filename):
        df = pandas.read_hdf(hdf_filename,key="df",mode="r")
        #df["ratio_KNS_KS"] = df["K_NS"]/df["K_S"]
        #df["layer1_static"] = True
        #df["ignore_off_during_optimization"] = 0
        #df["target_independent_of_clusters"] = 0
        #df["layer1_static"] = df["layer1_static"].map({True:int(1),1:int(1),0:int(0)})
        #df["ignore_off_during_optimization"] = df["ignore_off_during_optimization"].fillna(int(0))
        #df["target_independent_of_clusters"] = df["target_independent_of_clusters"].fillna(int(0))
        #df["ignore_off_during_optimization"] = df["ignore_off_during_optimization"].map({1:int(1),0:int(0),False:int(0),True:int(1)})
        #df["target_independent_of_clusters"] = df["target_independent_of_clusters"].map({1:int(1),0:int(0),False:int(0),True:int(1)})
        df["layer2_repressors"] = df["layer2_repressors"].fillna(int(0))
        df["layer2_repressors"] = df["layer2_repressors"].map({1:int(1),0:int(0),False:int(0),True:int(1)})

        #df = df.loc[df["filename"].apply(lambda x: '-07-' in x) | df["filename"].apply(lambda x: '-08-' in x)]
        #df_filenames = set(df["filename"])
        #for file in df_filenames:
            #print(file)
    else:
        print(f"error: nonexistent hdf file {hdf_filename}")
        return

    df.to_hdf(hdf_filename,key="df",mode="w",complevel=9)
    print(df)


if __name__ == "__main__":
    main(sys.argv[1:])
    exit(0)
