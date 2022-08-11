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
sym.var('c_S c_NS',real=True)

k_Sp = 1
k_Sm = 0.1
k_NSp = 1
k_NSm = 1
k_neq = 0.1
eta = 0.1
epsilon = 0
rm = 0.001

Q = sym.Matrix([[-c_S*k_Sp - c_NS*k_NSp - epsilon, c_S*k_Sp, c_NS*k_NSp, 0, 0, 0, 0, epsilon],
               [eta*k_Sm, -eta*k_Sm - k_neq, 0, k_neq, 0, 0, 0, 0],
               [eta*k_NSm, 0, -eta*k_NSm - k_neq, 0, k_neq, 0, 0, 0],
               [eta*k_Sm, 0, 0, -eta*k_Sm - k_neq, 0, k_neq, 0, 0],
               [eta*k_NSm, 0, 0, 0, -eta*k_NSm - k_neq, 0, k_neq, 0],
               [0, 0, 0, 0, 0, -k_Sm, 0, k_Sm],
               [0, 0, 0, 0, 0, 0, -k_NSm, k_NSm],
               [rm, 0, 0, 0, 0, c_S*k_Sp, c_NS*k_NSp, -c_S*k_Sp - c_NS*k_NSp - rm]])
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
pr_chromatin_open = sym.lambdify((c_NS,c_S),Pi_open,"numpy")


dill.settings['recurse'] = True
dill.dump(pr_chromatin_open, open("chromatin_kpr_pr_open.out", "wb"))
