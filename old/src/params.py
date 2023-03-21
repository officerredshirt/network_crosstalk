#!/usr/bin/env python
# coding: utf-8


# "pattern" specification
M_GENE = 100        # total number genes
M_ENH = M_GENE
N_ON = 10           # number ON genes
N_OFF = M_GENE - N_ON    # number OFF genes
N_OFF_shared = 18   # (scenario 3) number of OFF genes sharing PF with N_ON
sc = 3              # choice of scenario

# ARCHITECTURE
N_TF = M_GENE       # number transcription factors

if sc == 1:
    N_PF = M_GENE       # number pioneer factors
elif sc == 2:
    N_PF = M_GENE - N_ON + 1
elif sc == 3:
    N_PF = M_GENE - N_ON - N_OFF_shared + 1
THETA = 1           # number TFs per enhancer
n = 1               # number binding sites per TF

# EQUILIBRIUM TF BINDING
K_S = 10            # TF specific binding constant
K_NS = 100000       # TF nonspecific binding constant
n0 = 1              # minimum # bound TFs to induce expression

NUM_RANDINPUTS = 1
