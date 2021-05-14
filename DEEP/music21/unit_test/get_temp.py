#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 10:54:02 2019

@author: pabou
"""

import numpy as np

s = [0.1, 0.2, 0.2, 0.45, 0.05]
s= np.asarray(s)
print (s)
for t in [0.5,0.7,1,1.2,1.5,2]:
    s1 = np.log(s)/t
    s1 = np.exp(s1)
    s1 = s1/np.sum(s1)
    print(t, s1)