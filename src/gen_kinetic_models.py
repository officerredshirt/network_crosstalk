#!/usr/bin/env python
# coding: utf-8

import os, sys, shutil
from numpy import *
import sympy as sym

import matplotlib.pyplot as plt

import shelve
import dill

from params import *

def print_usage():
    print("usage is: gen_kinetic_models.py <output_folder>")

def get_stationary_dist(Q):
    d = Q.shape[0]
    QT = (sym.transpose(Q))
    temp = [[0]]*d
    temp.append([1])
    b = sym.Matrix(temp)
    x = sym.MatrixSymbol('pi',d,1)
    QT = QT.row_insert(d,sym.Matrix([[1]*d]))

    x = x.as_explicit()
    QT = QT.as_explicit()

    linear_eq = sym.Eq(QT*x,b)
    
    return sym.solve(linear_eq,x), x
    

def get_pr_open(Pi,x,state_open):
    Pi_open = sum([Pi[x[ii-1,0]] for ii in state_open])
    return sym.lambdify((C,c_S),Pi_open,"numpy")

def get_error_rate(Pi,x,corr_state,err_state):
    return sym.lambdify((C,c_S),Pi[x[err_state-1,0]] /
                        (Pi[x[err_state-1,0]] + Pi[x[corr_state-1,0]]),"numpy")


# Dill dumps binary files with lambda functions for probability
# of being open and error rate of a 6-state kinetic proofreading
# model for chromatin opening, an equivalent TF model (with the
# same dissociation rates as the binding in the chromatin layer),
# and a TF model with the parameters specified in params.py.
def main(argv):
    if len(argv) < 1:
        print_usage()

    dill.settings['recurse'] = True
    sym.init_printing()

    output_folder = argv[0]

    if not(os.path.exists(output_folder)):
        print(f"Making directory {output_folder}...")
        os.mkdir(output_folder)
        print(f"Copying params.py into {output_folder}...")
        shutil.copy("src/params.py",os.path.join(output_folder,"params.py"))
    
    # choose parameters before calculating
    sym.var('C c_S C_A c_A C_R c_R',real=True)


    print("Generating models...")

    ## --- TF BINDING --- ##
    # assume single cluster with single site
    print("  Generating TF binding models...")
    # for use in chromatin equivalent model
    K_NS_pfeq = kh_NSm/kh_NSp
    K_S_pfeq = kh_Sm/kh_Sp

    if not layer2_repressors:
        # "normal" model (with K_NS, K_S in params.py)
        tf_pr_bound = sym.lambdify((C,c_S),(1 - 1/(c_S/K_S + (C-c_S)/K_NS + 1)),"numpy")
        tf_error_rate = sym.lambdify((C,c_S),((C-c_S)/K_NS)/((C-c_S)/K_NS + c_S/K_S),"numpy")
              
        tf_chrom_equiv_pr_bound = sym.lambdify((C,c_S),(1 - 1/(c_S/K_S_pfeq + (C-c_S)/K_NS_pfeq + 1)),"numpy")
        tf_chrom_equiv_error_rate = sym.lambdify((C,c_S),((C-c_S)/K_NS_pfeq)/((C-c_S)/K_NS_pfeq + c_S/K_S_pfeq),"numpy")
    else:
        ## --- TF WITH REPRESSOR --- ##
        # TODO: fix error rates
        # C_R = total concentration of repressor
        # c_R = concentration of target repressor
        tf_chrom_equiv_pr_bound = sym.lambdify((C,c_S,C_R), \
                (1 - (1 + C_R/K_NS_pfeq)/(C_R/K_NS_pfeq + c_S/K_S_pfeq + (C-c_S)/K_NS_pfeq + 1)),"numpy")
        tf_chrom_equiv_error_rate = sym.lambdify((C_A,c_A,C_R,c_R),1,"numpy")

        tf_pr_bound = sym.lambdify((C_A,c_A,C_R,c_R), \
                (c_A/K_S + (C_A-c_A)/K_NS + C_A/K_NS + (C_A/K_NS)*(c_A/K_S) + (C_A/K_NS)*((C_A-c_A)/K_NS)) / \
                (1 + (c_R/K_S)*(c_A/K_S) + (c_R/K_S)*((C_A-c_A)/K_NS) + ((C_R-c_R)/K_NS)*((C_A-c_A)/K_NS) + \
                ((C_R-c_R)/K_NS)*(c_A/K_S) + (c_R/K_S) + ((C_R-c_R)/K_NS) + (c_R/K_S)*(C_R/K_NS) + \
                ((C_R-c_R)/K_NS)*(C_R/K_NS) + (C_A/K_NS)*(C_R/K_NS) + (C_R/K_NS) + \
                c_A/K_S + (C_A-c_A)/K_NS + C_A/K_NS + (C_A/K_NS)*(c_A/K_S) + (C_A/K_NS)*((C_A-c_A)/K_NS)), \
                "numpy")
        tf_error_rate = sym.lambdify((C_A,c_A,C_R,c_R),1,"numpy")

    dill.dump(tf_chrom_equiv_pr_bound, open(os.path.join(output_folder,"tf_chrom_equiv_pr_bound.out"),"wb"))
    dill.dump(tf_pr_bound, open(os.path.join(output_folder,"tf_pr_bound.out"),"wb"))
    dill.dump(tf_error_rate, open(os.path.join(output_folder,"tf_error_rate.out"),"wb"))
    dill.dump(tf_chrom_equiv_error_rate, open(os.path.join(output_folder,"tf_chrom_equiv_error_rate.out"),"wb"))
    

    ## --- CHROMATIN OPENING WITH PROOFREADING --- ##
    print("  Generating chromatin opening model...")
    Q = sym.Matrix([[-c_S*kh_Sp - (C-c_S)*kh_NSp - rp, c_S*kh_Sp, (C-c_S)*kh_NSp, 0, 0, rp],
                   [kh_Sm, -kh_Sm - k_neq, 0, k_neq, 0, 0],
                   [kh_NSm, 0, -kh_NSm - k_neq, 0, k_neq, 0],
                   [kh_Sm, 0, 0, -kh_Sm - k_neq, 0, k_neq],
                   [kh_NSm, 0, 0, 0, -kh_NSm - k_neq, k_neq],
                   [rm, 0, 0, 0, 0, - rm]])
    # state numbering: (1) closed/unbound, (2) closed/bound specific (1), (3) closed/bound nonspecific (1),
    # (4) closed/bound specific (2), (5) closed/bound nonspecific (2), (6) open/unbound
    
    Pi, x = get_stationary_dist(Q)
    pr_chromatin_open = get_pr_open(Pi,x,[6])
    
    # "error rate" as ratio of rate of transitions to "open" driven by nonspecific vs.
    # specific binding (k_neq cancels out, so equivalent to ratio of probabilities of
    # being in closed/bound (2))
    chromatin_opening_error_rate = get_error_rate(Pi,x,4,5)
    
    dill.dump(pr_chromatin_open, open(os.path.join(output_folder,"kpr_pr_open.out"), "wb"))
    dill.dump(chromatin_opening_error_rate, open(os.path.join(output_folder,"kpr_opening_error_rate.out"), "wb"))

if __name__ == "__main__":
    main(sys.argv[1:])
