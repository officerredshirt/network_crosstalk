#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# ARCHITECTURE
N_PF = 5      # number pioneer factors
N_TF = 10     # number transcription factors
M_ENH = 100   # number enhancers
M_GENE = 100  # number genes
THETA = 3     # number TFs per enhancer
n = 3         # number binding sites per TF

# EQUILIBRIUM TF BINDING
K_S = 0.1     # TF specific binding constant
K_NS = 100    # TF nonspecific binding constant

NUM_RANDINPUTS = pow(2,N_PF+N_TF)     # number random input patterns to test for achievable outputs