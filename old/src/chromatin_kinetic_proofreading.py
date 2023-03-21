#!/usr/bin/env python
# coding: utf-8

from numpy import *
import sympy as sym
import matplotlib.pyplot as plt
import shelve
import dill

sym.init_printing()

# choose parameters before calculating
# sym.var('k_Sp k_Sm k_NSp k_NSm k_neq eta epsilon rm',real=True)
sym.var('pi1 pi2 pi3 pi4 pi5 pi6 pi7 pi8',real=True)
sym.var('C c_S',real=True)

"""
rp = 0
rm = 0.0002     # s-1
kh_Sp = 0.0005  # s-1 nM-1
kh_Sm = 0.005   # s-1
kh_NSp = 0.0005 # s-1 nM-1
kh_NSm = 5      # s-1
k_neq = 0.2     # s-1
k_Sp = 0.025    # s-1 nM-1
k_Sm = 0.25     # s-1
k_NSp = 0.025   # s-1 nM-1
k_NSm = 250     # s-1
"""
rp = 0
rm = 0.0002     # s-1
kh_Sp = 0.0005  # s-1 nM-1
kh_Sm = 0.005   # s-1
kh_NSp = 0.0005 # s-1 nM-1
kh_NSm = 75     # s-1  **this appears to be the most important for proper proofreading
k_neq = 0.2     # s-1
k_Sp = 0.025    # s-1 nM-1
k_Sm = 0.25     # s-1
k_NSp = 0.025   # s-1 nM-1
k_NSm = 250     # s-1

Q = sym.Matrix([[-c_S*kh_Sp - (C-c_S)*kh_NSp - rp, c_S*kh_Sp, (C-c_S)*kh_NSp, 0, 0, 0, 0, rp],
               [kh_Sm, -kh_Sm - k_neq, 0, k_neq, 0, 0, 0, 0],
               [kh_NSm, 0, -kh_NSm - k_neq, 0, k_neq, 0, 0, 0],
               [kh_Sm, 0, 0, -kh_Sm - k_neq, 0, k_neq, 0, 0],
               [kh_NSm, 0, 0, 0, -kh_NSm - k_neq, 0, k_neq, 0],
               [0, 0, 0, 0, 0, -k_Sm, 0, k_Sm],
               [0, 0, 0, 0, 0, 0, -k_NSm, k_NSm],
               [rm, 0, 0, 0, 0, c_S*k_Sp, (C-c_S)*k_NSp, -c_S*k_Sp - (C-c_S)*k_NSp - rm]])
# state numbering: (1) closed/unbound, (2) closed/bound specific (1), (3) closed/bound nonspecific (1),
# (4) closed/bound specific (2), (5) closed/bound nonspecific (2), (6) open/bound specific,
# (7) open/bound nonspecific, (8) open/unbound


# stationary distribution
QT = (sym.transpose(Q))
b = sym.Matrix([[0],[0],[0],[0],[0],[0],[0],[0],[1]])
x = sym.Matrix([[pi1],[pi2],[pi3],[pi4],[pi5],[pi6],[pi7],[pi8]])
QT = QT.row_insert(8,sym.Matrix([[1,1,1,1,1,1,1,1]]))
linear_eq = sym.Eq(QT*x,b)

Pi = sym.solve(linear_eq,x)


Pi_open = Pi[pi6] + Pi[pi7] + Pi[pi8]
pr_chromatin_open = sym.lambdify((C,c_S),Pi_open,"numpy")

# "error rate" as ratio of rate of transitions to "open" driven by nonspecific vs.
# specific binding (k_neq cancels out, so equivalent to ratio of probabilities of
# being in closed/bound (2))
chromatin_opening_error_rate = sym.lambdify((C,c_S),Pi[pi5]/(Pi[pi4] + Pi[pi5]),"numpy")


dill.settings['recurse'] = True
dill.dump(pr_chromatin_open, open("src/chromatin_kpr_pr_open.out", "wb"))
dill.dump(chromatin_opening_error_rate, open("src/chromatin_opening_error_rate.out", "wb"))
