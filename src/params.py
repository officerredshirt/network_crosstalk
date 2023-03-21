#!/usr/bin/env python
# coding: utf-8

# ARCHITECTURE
N_PF = 5                # number pioneer factors
N_CLUSTERS = N_PF       # number clusters
GENES_PER_CLUSTER = 5   # number genes per cluster
M_GENE = GENES_PER_CLUSTER*N_CLUSTERS   # number genes
N_TF = M_GENE           # number transcription factors
M_ENH = M_GENE          # number enhancers

# LAYER 1 (CHROMATIN)
rp = 0
rm = 0.00020    # s-1
kh_Sp = 0.0005  # s-1 nM-1
kh_Sm = 0.005   # s-1
kh_NSp = 0.0005 # s-1 nM-1
kh_NSm = 50     # s-1  **this appears to be the most important for proper proofreading
k_neq = 0.05    # s-1

# LAYER 2 EQUILIBRIUM TF
THETA = 1     # number TFs per enhancer (layer 2)
n = 1         # number binding sites per TF (layer 2)
K_S = kh_Sm/kh_Sp  # TF binding dissociation rate (specific)
#K_NS = 100000      # TF binding dissociation rate (nonspecific)
K_NS = kh_NSm/kh_NSp    # s-1

# TARGET PATTERN
NUM_TARGETS = 10        # number random target patterns to generate
MIN_CLUSTERS_ACTIVE = 1 # minimum number of clusters containing expressing genes
MAX_CLUSTERS_ACTIVE = 5 # maximum number of clusters containing expressing genes
MIN_EXPRESSION = 0.3    # minimum expression level for a gene in a target pattern
MAX_EXPRESSION = 0.9    # maximum expression level for a gene in a target pattern

# OPTIMIZATION
eps = 1e-10
