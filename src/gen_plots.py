from numpy import *

import manage_db
import plot_db
import matplotlib.pyplot as plt
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

    #output_folder = os.path.join(folder,"plots")

    #manage_db.plot_xtalk_errors(database,output_folder)
    #manage_db.plot_xtalk_errors(os.path.join(folder,"res"),output_folder)

    df = plot_db.combine_databases([database])
    varnames_dict = plot_db.get_varname_to_value_dict(df)
    fntsz=36
    fig, ax = plt.subplots(2,2,figsize=(24,24))
    plot_db.subplots_groupby(df,
                             "tf_first_layer",
                             [],[],
                             plot_db.scatter_target_expression_groupby,
                             ["ratio_KNS_KS"],
                             ax=ax[0][0:],fontsize=fntsz,
                             varnames_dict=varnames_dict)
    plot_db.subplots_groupby(df,
                             "tf_first_layer",
                             [],[],
                             plot_db.scatter_repressor_activator,
                             ["ratio_KNS_KS"],
                             ax=ax[1][0:],fontsize=fntsz,
                             varnames_dict=varnames_dict)

    plt.savefig("../testing.png")
    plt.close()

if __name__ == "__main__":
    main(sys.argv[1:])
    exit(0)
