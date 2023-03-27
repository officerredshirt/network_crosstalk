#!/usr/bin/env python
# coding: utf-8


# "pattern" specification
M = 100             # total number genes
N_ON = 10           # number ON genes
N_OFF = M - N_ON    # number OFF genes
N_OFF_frac_shared = 0.2 # (scenario 3) fraction of OFF genes sharing PF with ON genes

# parameters for xtalk heatmap plots
npts = 500          # number sample points
