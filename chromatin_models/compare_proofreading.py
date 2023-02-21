#!/usr/bin/env python
# coding: utf-8

from numpy import *
from math import *
from chromatin_params import *
import sympy as sym
import matplotlib.pyplot as plt
import shelve
import dill
plt.rcParams.update({'font.size': 20})

no_kpr_4state_pr_on = dill.load(open("no_kpr_4state_pr_open.out", "rb"))
no_kpr_4state_error_rate = dill.load(open("no_kpr_4state_opening_error_rate.out", "rb"))
no_kpr_6state_pr_on = dill.load(open("no_kpr_6state_pr_open.out", "rb"))
no_kpr_6state_error_rate = dill.load(open("no_kpr_6state_opening_error_rate.out", "rb"))
kpr_pr_on = dill.load(open("kpr_pr_open.out", "rb"))
kpr_error_rate = dill.load(open("kpr_opening_error_rate.out", "rb"))


## TF BINDING ##
K_S = kh_Sm/kh_Sp
K_NS = kh_NSm/kh_NSp
n = 1

# construct matrix of combinatorial coefficients
A = zeros([n+1,n+1])
for m_ns in range(n+1):
    for m_s in range(n+1-m_ns):
        A[m_ns,m_s] = factorial(n)/(factorial(m_ns)*factorial(m_s)*factorial(n-m_ns-m_s))

def pr_cluster_unbound(C,c_j):
    return 1/(power(((C-c_j)/K_NS)*ones(n+1),range(n+1))@A@power((c_j/K_S)*ones(n+1),range(n+1)))

# assume theta = 1 (we have only one homotypic cluster)
def pr_gene_on(C,c_s):
    return (1 - pr_cluster_unbound(C,c_s))

# assume single cluster with single site (to match single PF site for chromatin)
def pr_gene_on(C,c_s):
    return (1 - 1/(c_s/K_S + (C-c_s)/K_NS + 1))

def gene_on_error_rate(C,c_s):
    return ((C-c_s)/K_NS)/((C-c_s)/K_NS + c_s/K_S)


## COMPARE SCHEMES ##
c_ns = 100000
# c_s_set = linspace(0,50,num=100)
c_s_set = logspace(-5,5,100)
C_set = c_ns + c_s_set

tf_frac_on = pr_gene_on(C_set,c_s_set)
no_kpr4_frac_on = no_kpr_4state_pr_on(C_set-c_s_set,c_s_set)
no_kpr6_frac_on = no_kpr_6state_pr_on(C_set-c_s_set,c_s_set)
kpr_frac_on = kpr_pr_on(C_set-c_s_set,c_s_set)

tf_error_rates = gene_on_error_rate(C_set,c_s_set)
no_kpr4_error_rates = no_kpr_4state_error_rate(C_set-c_s_set,c_s_set)
no_kpr6_error_rates = no_kpr_6state_error_rate(C_set-c_s_set,c_s_set)
kpr_error_rates = kpr_error_rate(C_set-c_s_set,c_s_set)

# plot error rate
fig, ax = plt.subplots(figsize = (24,12))
ax.plot(c_s_set,tf_error_rates,label="tf")
ax.plot(c_s_set,no_kpr4_error_rates,label="no kpr (4-state)")
# ax.plot(c_s_set,no_kpr6_error_rates,label="no kpr (6-state)")
ax.plot(c_s_set,kpr_error_rates,label="kpr")
ax.set_xlabel("binding factor concentration")
ax.set_ylabel("error rate")
ax.set_xscale("log")
ax.legend()
plt.savefig("tf_nokpr_kpr_error_rate.png")


# plot probability on or open
fig, ax = plt.subplots(figsize = (24,12))
ax.plot(c_s_set,tf_frac_on,label="tf")
ax.plot(c_s_set,no_kpr4_frac_on,label="no kpr (4-state)")
# ax.plot(c_s_set,no_kpr6_frac_on,label="no kpr (6-state)")
ax.plot(c_s_set,kpr_frac_on,label="kpr")
ax.set_xlabel("binding factor concentration")
ax.set_ylabel("probability on/open")
ax.set_xscale("log")
ax.legend()
plt.savefig("tf_nokpr_kpr_pr_on.png")


# plot probability on or open vs. error rate
fig, ax = plt.subplots(figsize = (24,12))
ax.plot((0.9,0.9),(0,1),label="90% induction",color="red",linestyle="--")
ax.plot(tf_frac_on,tf_error_rates,label="tf")
ax.plot(no_kpr4_frac_on,no_kpr4_error_rates,label="no kpr (4-state)")
ax.plot(kpr_frac_on,kpr_error_rates,label="kpr")
ax.set_xlabel("probability on/open")
ax.set_ylabel("error rate")
ax.set_yscale("log")
ax.legend()
plt.savefig("tf_nokpr_kpr_pr_v_error.png")
