#!/user/bin/env python

from numpy import *
from math import *
from params import *
from boolarr import *
import dill
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

pr_chromatin_open = dill.load(open("./src/chromatin_kpr_pr_open.out","rb"))


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

## -- TEST -- ##
c_s = 1
c_ns = 1000
C = c_s + c_ns

print(f"TF: Pr(on when should be off) = {pr_gene_on(c_ns,0)}")
print(f"PF: Pr(open when should be closed) = {pr_chromatin_open(c_ns,0)}")
print(f"TF: Pr(on when should be) = {pr_gene_on(C,c_s)}")
print(f"PF: Pr(open when should be) = {pr_chromatin_open(C,c_s)}")


# -- COMPARE ERROR RATES -- #
# single cluster, single site
def gene_on_error_rate(C,c_s):
    return ((C-c_s)/K_NS)/((C-c_s)/K_NS + c_s/K_S)

chromatin_opening_error_rate = dill.load(open("./src/chromatin_opening_error_rate.out","rb"))

tf_error_rate = gene_on_error_rate(C,c_s)
pf_error_rate = chromatin_opening_error_rate(C,c_s)
print(f"TF error rate = {tf_error_rate}")
print(f"PF error rate = {pf_error_rate}")
print(f"PF error rate/TF error rate = {pf_error_rate/tf_error_rate}")


# -- GENERATE PLOT -- #
c_s_set = linspace(0,100,num=100)
C_set = c_ns + c_s_set

tf_error_rates = gene_on_error_rate(C_set,c_s_set)
pf_error_rates = chromatin_opening_error_rate(C_set,c_s_set)

fig, ax = plt.subplots(figsize = (24,12))
ax.plot(c_s_set,tf_error_rates,label="tf error rate")
ax.plot(c_s_set,pf_error_rates,label="pf error rate")
ax.set_xlabel("binding factor concentration")
ax.set_ylabel("error rate")
plt.savefig("tf_pf_single_gene_error_rates.png")

fig, ax = plt.subplots(figsize = (24,12))
ax.plot(c_s_set,pf_error_rates/tf_error_rates)
ax.set_xlabel("binding factor concentration")
ax.set_ylabel("ratio of error rates (PF/TF)")
plt.savefig("tf_pf_single_gene_ratio_error_rates.png")
