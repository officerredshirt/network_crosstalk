#!/usr/bin/env python
# coding: utf-8
from numpy import *
from multiprocess import Pool
import shelve
import sys, argparse
import manage_db
# import time

from params import *
from boolarr import *


def main(argv):

    parser = argparse.ArgumentParser(
            prog = "set_custom_achievable_patterns",
            description = "",
            epilog = "")
    parser.add_argument("-i","--filename_in",required=True)
    parser.add_argument("-d","--database",default="temp.db")

    args = parser.parse_args()
    filename_in = args.filename_in
    database = args.database

    # load architecture
    local_id = manage_db.extract_local_id(filename_in)
    R, T, G = manage_db.get_network(database,local_id)

    # generate target patterns where ON genes have random expression level
    # between MIN_EXPRESSION and MAX_EXPRESSION
    for ii in range(NUM_TARGETS):
        # PFs and TFs that are present to generate this pattern
        u = [False]*(N_PF+N_TF)

        # randomly choose number active clusters
        n_active_clusters = random.randint(MIN_CLUSTERS_ACTIVE,MAX_CLUSTERS_ACTIVE+1)

        if target_distribution == "uni":
            target_pattern = random.default_rng().uniform(MIN_EXPRESSION,MAX_EXPRESSION,M_GENE)
        elif target_distribution == "loguni":
            exponent_vals = random.default_rng().uniform(log10(MIN_EXPRESSION),log10(MAX_EXPRESSION),M_GENE)
            target_pattern = power(10,exponent_vals)
        elif target_distribution == "invloguni":
            exponent_vals = random.default_rng().uniform(log10(MIN_EXPRESSION),log10(MAX_EXPRESSION),M_GENE)
            target_pattern = MAX_EXPRESSION - power(10,exponent_vals) + MIN_EXPRESSION
        else:
            print(f"unrecognized target distribution option {target_distribution}")
            sys.exit()

        if target_independent_of_clusters:
            off_genes = random.choice(range(M_GENE),
                                      size=GENES_PER_CLUSTER*(N_CLUSTERS - n_active_clusters),
                                      replace=False)
            target_pattern[off_genes.astype(int)] = 0
        else:
            u[0:n_active_clusters] = [True]*n_active_clusters

            # randomly assign expression levels to genes in clusters
            n_active_genes = GENES_PER_CLUSTER*n_active_clusters
            u[N_PF:N_PF+n_active_genes] = [True]*n_active_genes
            #target_pattern[0:n_active_genes] = random.default_rng().uniform(MIN_EXPRESSION,MAX_EXPRESSION,n_active_genes)
            target_pattern[n_active_genes:] = 0
        
        manage_db.add_pattern(database,local_id,array(u),target_pattern)


    # -- SAVE PROOF OF COMPLETION FOR SNAKEMAKE FLOW -- #
    with open(filename_in + ".target","w") as file:
        pass


if __name__ == "__main__":
    main(sys.argv[1:])
    exit(0)
