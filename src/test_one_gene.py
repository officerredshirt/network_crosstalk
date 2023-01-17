#!/user/bin/env python

from numpy import *
from math import *
from params import *
from boolarr import *
import dill

# construct matrix of combinatorial coefficients
A = zeros([n+1,n+1])
for m_ns in range(n+1):
    for m_s in range(n+1-m_ns):
        A[m_ns,m_s] = factorial(n)/(factorial(m_ns)*factorial(m_s)*factorial(n-m_ns-m_s))

def pr_cluster_unbound(C,c_j):
    return 1/(power(((C-c_j)/K_NS)*ones(n+1),range(n+1))@A@power((c_j/K_S)*ones(n+1),range(n+1)))

# assume theta = 1 (we have only one homotypic cluster)
def pr_gene_on(C,c):
    return (1 - pr_cluster_unbound(C,c))

pr_chromatin_open = dill.load(open("./src/chromatin_kpr_pr_open.out","rb"))



## -- TEST -- ##
c_s = 10
C_ns = 1000

print(f"TF: Pr(on when should be off) = {pr_gene_on(C_ns,0)}")
print(f"PF: Pr(open when should be closed) = {pr_chromatin_open(C_ns,0)}")
print(f"TF: Pr(on when should be) = {pr_gene_on(C_ns,c_s)}")
print(f"PF: Pr(open when should be) = {pr_chromatin_open(C_ns,c_s)}")
