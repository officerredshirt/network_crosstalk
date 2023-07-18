#!/usr/bin/env python
# coding: utf-8

# ARCHITECTURE
N_PF = 5               # number pioneer factors
N_CLUSTERS = N_PF       # number clusters
GENES_PER_CLUSTER = 5   # number genes per cluster
M_GENE = GENES_PER_CLUSTER*N_CLUSTERS   # number genes
N_TF = M_GENE           # number transcription factors
M_ENH = M_GENE          # number enhancers

ratio_KNS_KS = 1000
layer1_static = False   # determines whether Layer 1 K_NS scales with same ratio as Layer 2

# LAYER 1 (CHROMATIN)
rp = 0
rm = 0.00020    # s-1
kh_Sp = 0.0005  # s-1 nM-1
kh_Sm = 0.005   # s-1
kh_NSp = 0.0005 # s-1 nM-1
if layer1_static:
    kh_NSm = 50     # s-1  **this appears to be the most important for proper proofreading
else:
    kh_NSm = kh_NSp*ratio_KNS_KS*(kh_Sm/kh_Sp)
k_neq = 0.05    # s-1

# LAYER 2 EQUILIBRIUM TF
THETA = 1     # number TFs per enhancer (layer 2)
n = 1         # number binding sites per TF (layer 2)
K_S = kh_Sm/kh_Sp  # TF binding dissociation rate (specific), s-1
K_NS = 100      # TF binding dissociation rate (nonspecific), s-1

# TARGET PATTERNS
NUM_TARGETS = 10        # number random target patterns to generate
MIN_CLUSTERS_ACTIVE = 2 # minimum number of clusters containing expressing genes
MAX_CLUSTERS_ACTIVE = 2 # maximum number of clusters containing expressing genes
MIN_EXPRESSION = 0.3    # minimum expression level for a gene in a target pattern
MAX_EXPRESSION = 0.9    # maximum expression level for a gene in a target pattern

# OPTIMIZATION
eps = 1e-10
