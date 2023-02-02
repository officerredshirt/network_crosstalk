#!/usr/bin/env python
# coding: utf-8

from numpy import *
from chromatin_params import *
import sympy as sym
import matplotlib.pyplot as plt
import shelve
import dill

dill.settings['recurse'] = True
sym.init_printing()

# choose parameters before calculating
sym.var('C c_S kh_NSm_var k_neq_var',real=True)

def get_stationary_dist(Q):
    d = Q.shape[0]
    #varnames = [locals()[f"pi{x+1}"] for x in range(d)]

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
    return sym.lambdify((kh_NSm_var,k_neq_var,C,c_S),Pi_open,"numpy")

def get_error_rate(Pi,x,corr_state,err_state):
    return sym.lambdify((kh_NSm_var,k_neq_var,C,c_S),Pi[x[err_state-1,0]] /
                        (Pi[x[err_state-1,0]] + Pi[x[corr_state-1,0]]),"numpy")


# --- CHROMATIN 4-STATE (no PF binding open DNA) --- #
Q_4state_no_kpr = sym.Matrix([[-c_S*kh_Sp - (C-c_S)*kh_NSp - rp, c_S*kh_Sp, (C-c_S)*kh_NSp, rp],
               [kh_Sm, -kh_Sm - k_neq_var, 0, k_neq_var],
               [kh_NSm_var, 0, -kh_NSm_var - k_neq_var, k_neq_var],
               [rm, 0, 0, - rm]])
# state numbering: (1) closed/unbound, (2) closed/bound specific, (3) closed/bound nonspecific, (4) open/unbound

Pi, x = get_stationary_dist(Q_4state_no_kpr)
pr_chromatin_open = get_pr_open(Pi,x,[4])

# "error rate" as ratio of rate of transitions to "open" driven by nonspecific vs.
# specific binding (k_neq cancels out, so equivalent to ratio of probabilities of
# being in closed/bound)
chromatin_opening_error_rate = get_error_rate(Pi,x,2,3)

dill.dump(pr_chromatin_open, open("VARPARAM_no_kpr_4state_pr_open.out", "wb"))
dill.dump(chromatin_opening_error_rate, open("VARPARAM_no_kpr_4state_opening_error_rate.out", "wb"))


## --- WITH PROOFREADING --- ##
Q = sym.Matrix([[-c_S*kh_Sp - (C-c_S)*kh_NSp - rp, c_S*kh_Sp, (C-c_S)*kh_NSp, 0, 0, rp],
               [kh_Sm, -kh_Sm - k_neq_var, 0, k_neq_var, 0, 0],
               [kh_NSm_var, 0, -kh_NSm_var - k_neq_var, 0, k_neq_var, 0],
               [kh_Sm, 0, 0, -kh_Sm - k_neq_var, 0, k_neq_var],
               [kh_NSm_var, 0, 0, 0, -kh_NSm_var - k_neq_var, k_neq_var],
               [rm, 0, 0, 0, 0, - rm]])
# state numbering: (1) closed/unbound, (2) closed/bound specific (1), (3) closed/bound nonspecific (1),
# (4) closed/bound specific (2), (5) closed/bound nonspecific (2), (6) open/unbound

Pi, x = get_stationary_dist(Q)
pr_chromatin_open = get_pr_open(Pi,x,[6])

# "error rate" as ratio of rate of transitions to "open" driven by nonspecific vs.
# specific binding (k_neq cancels out, so equivalent to ratio of probabilities of
# being in closed/bound (2))
chromatin_opening_error_rate = get_error_rate(Pi,x,4,5)

dill.dump(pr_chromatin_open, open("VARPARAM_kpr_pr_open.out", "wb"))
dill.dump(chromatin_opening_error_rate, open("VARPARAM_kpr_opening_error_rate.out", "wb"))
