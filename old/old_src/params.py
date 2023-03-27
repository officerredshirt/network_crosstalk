#!/usr/bin/env python
# coding: utf-8

# ARCHITECTURE
N_PF = 2#5      # number pioneer factors
N_TF = 5#10     # number transcription factors
M_ENH = 50#100   # number enhancers
M_GENE = 50#100  # number genes
THETA = 2#3     # number TFs per enhancer
n = 3         # number binding sites per TF

# EQUILIBRIUM TF BINDING
K_S = 10      # TF specific binding constant
K_NS = 10000  # TF nonspecific binding constant
n0 = 1        # minimum # bound TFs to induce expression

NUM_RANDINPUTS = pow(2,N_PF+N_TF)     # number random input patterns to test for achievable outputs
