#!/usr/bin/env python
# coding: utf-8

rp = 0
rm = 0.0002     # s-1
kh_Sp = 0.0005  # s-1 nM-1
kh_Sm = 0.005   # s-1
kh_NSp = 0.0005 # s-1 nM-1
kh_NSm = 50     # s-1  **this appears to be the most important for proper proofreading
k_neq = 0.05    # s-1
K_S = kh_Sm/kh_Sp  # TF binding dissociation rate (specific)
K_NS = 100      # TF binding dissociation rate (nonspecific)
