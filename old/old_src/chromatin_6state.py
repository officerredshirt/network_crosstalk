#!/usr/bin/env python
# coding: utf-8
from numpy import *
import sympy as sym
import matplotlib.pyplot as plt
import shelve
import dill

sym.init_printing()


# -- GENERATE KINETIC MODEL -- #
# full symbolic solution is too slow; choose parameters before calculating

#sym.var('c_S c_NS nu xi eta_mS eta_pS eta_mNS eta_pNS k_mS k_pS k_mNS k_pNS r_m r_p',real=True)
sym.var('pi1 pi2 pi3 pi4 pi5 pi6',real=True)
sym.var('c_S c_NS',real=True)
nu = 10000
xi = 1
eta_mS = 10
eta_pS = 0.01
eta_mNS = 10
eta_pNS = 0.01
k_mS = 0.01
k_pS = 1
k_mNS = 10
k_pNS = 1
r_m = 0.1
r_p = 0.0001

Q = sym.Matrix([[-c_S*k_pS-c_NS*k_pNS-r_p, c_S*k_pS, c_NS*k_pNS, 0, 0, r_p],
               [k_mS, -k_mS-nu*r_p, 0, nu*r_p, 0, 0],
               [k_mNS, 0, -k_mNS-nu*r_p, 0, nu*r_p, 0],
               [0, r_m, 0, -r_m-eta_mS*k_mS, 0, eta_mS*k_mS],
               [0, 0, r_m, 0, -r_m-eta_mNS*k_mNS, eta_mNS*k_mNS],
               [xi*r_m, 0, 0, c_S*eta_pS*k_pS, c_NS*eta_pNS*k_pNS, -xi*r_m-c_S*eta_pS*k_pS-c_NS*eta_pNS*k_pNS]])
# state numbering: (1) closed/unbound, (2) closed/bound specific, (3) closed/bound nonspecific,
# (4) open/bound specific, (5) open/bound nonspecific, (6) open/unbound

QT = (sym.transpose(Q))


# -- CALCULATE STATIONARY DISTRIBUTION -- #
b = sym.Matrix([[0],[0],[0],[0],[0],[0],[1]])
x = sym.Matrix([[pi1],[pi2],[pi3],[pi4],[pi5],[pi6]])
QT = QT.row_insert(6,sym.Matrix([[1,1,1,1,1,1]]))
linear_eq = sym.Eq(QT*x,b)

Pi = sym.solve(linear_eq,x)

Pi_open = Pi[pi4] + Pi[pi5] + Pi[pi6]
pr_chromatin_open = sym.lambdify((c_NS,c_S),Pi_open,"numpy")

#print(pr_chromatin_open(0,0.1))
#print(pr_chromatin_open(1,0.1))


# -- WRITE FUNCTION FOR STEADY-STATE PROBABILITIES TO FILE -- #
filename = 'chromatin_6state.out'
ms = shelve.open(filename,'n')
vars_to_save = ['c_S','c_NS','nu','xi','eta_mS','eta_pS','eta_mNS','eta_pNS','k_mS','k_pS','k_mNS','k_pNS','r_m','r_p',
                'pi1','pi2','pi3','pi4','pi5','pi6','Pi']
for key in vars_to_save:
    ms[key] = globals()[key]
ms.close()

dill.settings['recurse'] = True
dill.dump(pr_chromatin_open, open("chromatin_6state_pr_open.out", "wb"))
