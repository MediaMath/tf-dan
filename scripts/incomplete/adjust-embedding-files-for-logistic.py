#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 15:36:23 2018

@author: dc
"""

import os

data_path = '../data/processed/mm-cpc-logistic/train-negative-'
for i in range(12,101):
    idx = '{:04}'.format(i)
    a = data_path + idx +'.csv'
    os.remove(a)
    
