#!/usr/bin/env python
# coding: utf-8

from math import *
from numpy import *

from params import *

# construct matrix of combinatorial coefficients
A = zeros([n+1,n+1])
for m_ns in range(n+1):
    for m_s in range(n+1-m_ns):
        A[m_ns,m_s] = factorial(n)/(factorial(m_ns)*factorial(m_s)*factorial(n-m_ns-m_s))

def pr_cluster_unbound(C,c_j):
    return 1/(power(((C-c_j)/K_NS)*ones(n+1),range(n+1))@A@power((c_j/K_S)*ones(n+1),range(n+1)))

# inputs are:
# C : total concentration C of all TFs
# c : vector of concentrations c for TFs binding the region
def pr_gene_on(C,c):
    pr_cu_wrapper = lambda c_j: pr_cluster_unbound(C,c_j)
    
    p_off = prod(list(map(pr_cu_wrapper,c)))
    return (1 - p_off)
