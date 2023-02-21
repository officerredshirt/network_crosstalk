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
sym.var('pi1 pi2 pi3 pi4',real=True)
sym.var('c_S c_NS',real=True)
sym.var('k_Sp,k_Sm,k_NSp,k_NSm,k_neq,eps',real=True)

Q = sym.Matrix([[-c_S*k_Sp-c_NS*k_NSp,c_S*k_Sp,c_NS*k_NSp,0],
                [k_Sm,-k_Sm-k_neq,0,k_neq],
                [k_NSm,0,-k_NSm-k_neq,k_neq],
                [eps,0,0,-eps]])
# state numbering: (1) closed unbound, (2) closed bound specific, (3) closed bound nonspecific, (4) open unbound


# stationary distribution
QT = (sym.transpose(Q))
b = sym.Matrix([[0],[0],[0],[0],[1]])
x = sym.Matrix([[pi1],[pi2],[pi3],[pi4]])
QT = QT.row_insert(4,sym.Matrix([[1,1,1,1]]))
linear_eq = sym.Eq(QT*x,b)

Pi = sym.solve(linear_eq,x)

print(Pi[pi1])
print(Pi[pi2])
print(Pi[pi3])
print(Pi[pi4])
