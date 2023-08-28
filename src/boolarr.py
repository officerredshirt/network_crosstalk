#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from numpy import *

# convert u as boolean array to u as int
def bool2int(u):
    return int(f"{''.join(str(int(x)) for x in u)}",2)

# convert u as int into u as boolean array
def int2bool(u,N):
    return array([(ch == "1") for ch in str(f"{u:0{N}b}")])

