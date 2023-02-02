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
sym.var('C c_S',real=True)

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
    return sym.lambdify((C,c_S),Pi_open,"numpy")

def get_error_rate(Pi,x,corr_state,err_state):
    return sym.lambdify((C,c_S),Pi[x[err_state-1,0]] /
                        (Pi[x[err_state-1,0]] + Pi[x[corr_state-1,0]]),"numpy")


# --- CHROMATIN 4-STATE (no PF binding open DNA) --- #
Q_4state_no_kpr = sym.Matrix([[-c_S*kh_Sp - (C-c_S)*kh_NSp - rp, c_S*kh_Sp, (C-c_S)*kh_NSp, rp],
               [kh_Sm, -kh_Sm - k_neq, 0, k_neq],
               [kh_NSm, 0, -kh_NSm - k_neq, k_neq],
               [rm, 0, 0, - rm]])
# state numbering: (1) closed/unbound, (2) closed/bound specific, (3) closed/bound nonspecific, (4) open/unbound

Pi, x = get_stationary_dist(Q_4state_no_kpr)
pr_chromatin_open = get_pr_open(Pi,x,[4])

# "error rate" as ratio of rate of transitions to "open" driven by nonspecific vs.
# specific binding (k_neq cancels out, so equivalent to ratio of probabilities of
# being in closed/bound)
chromatin_opening_error_rate = get_error_rate(Pi,x,2,3)

dill.dump(pr_chromatin_open, open("no_kpr_4state_pr_open.out", "wb"))
dill.dump(chromatin_opening_error_rate, open("no_kpr_4state_opening_error_rate.out", "wb"))


# --- CHROMATIN 6-STATE (PF binding open DNA) --- #
# stationary distribution
Q_6state_no_kpr = sym.Matrix([[-c_S*kh_Sp - (C-c_S)*kh_NSp - rp, c_S*kh_Sp, (C-c_S)*kh_NSp, 0, 0, rp],
               [kh_Sm, -kh_Sm - k_neq, 0, k_neq, 0, 0],
               [kh_NSm, 0, -kh_NSm - k_neq, 0, k_neq, 0],
               [0, 0, 0, -k_Sm, 0, k_Sm],
               [0, 0, 0, 0, -k_NSm, k_NSm],
               [rm, 0, 0, c_S*k_Sp, (C-c_S)*k_NSp, -c_S*k_Sp - (C-c_S)*k_NSp - rm]])
# state numbering: (1) closed/unbound, (2) closed/bound specific, (3) closed/bound nonspecific, (4) open/bound specific, (5) open/bound nonspecific, (6) open/unbound

Pi, x = get_stationary_dist(Q_6state_no_kpr)
pr_chromatin_open = get_pr_open(Pi,x,[4,5,6])

# "error rate" as ratio of rate of transitions to "open" driven by nonspecific vs.
# specific binding (k_neq cancels out, so equivalent to ratio of probabilities of
# being in closed/bound)
chromatin_opening_error_rate = get_error_rate(Pi,x,2,3)

dill.dump(pr_chromatin_open, open("no_kpr_6state_pr_open.out", "wb"))
dill.dump(chromatin_opening_error_rate, open("no_kpr_6state_opening_error_rate.out", "wb"))



## --- WITH PROOFREADING --- ##
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

dill.dump(pr_chromatin_open, open("kpr_pr_open.out", "wb"))
dill.dump(chromatin_opening_error_rate, open("kpr_opening_error_rate.out", "wb"))
