#!/usr/bin/env python
# coding: utf-8


# "pattern" specification
"""
M = 100             # total number genes
N_ON = 50           # number ON genes
N_OFF = M - N_ON    # number OFF genes
N_OFF_shared = 25   # (scenario 3) number of OFF genes sharing PF with N_ON
"""
M = 4          # total number genes
N_ON = 2           # number ON genes
N_OFF = M - N_ON    # number OFF genes
N_OFF_shared = 1   # (scenario 3) number of OFF genes sharing PF with N_ON


# parameters for xtalk heatmap plots
npts = 500          # number sample points

# parameters for optimization
C0 = [10]*2         # perform optimization over total concentrations
eps = 1e-10          # tolerance
