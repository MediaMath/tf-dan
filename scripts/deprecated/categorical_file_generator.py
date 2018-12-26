#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 18:37:48 2018

@author: james.f.xue

Script to create txt files for all columns

1. Needs to be able to update the file with new file if present
2. Needs to be able to use product in categorical_column_with_vocabulary_file 
"""

import pandas as pd 
import numpy as np
import os.path 
from pathlib import Path
import glob
import time
import argparse

def categorical_file_generator(input_file, output_dir): 
    """
    A script to generate a .txt file for each column of the file_dir.
    If a .txt file with a matching name exists, add to the elements in the .txt file. 
    
    Parameters
    ==========
    input_file: a csv file of ctr data
    output_dir: 1. directory to output .txt files
                2. also the location to see if there is a matching .txt file
                
    >>> Usage Example: 
        input_file = "/Users/james.f.xue/Desktop/test2.csv"
        output_dir = "/Users/james.f.xue/Desktop/test/"  
        categorical_file_generator(input_file, output_dir)
                
    """
    
    #load file into pandas
    file = pd.read_csv(input_file, ",")
    
    #find all the columns
    columns = file.columns 
    
    #for each column, generate .txt or add to existing .txt 
    for column in columns: 
        output_file = output_dir + column 
        output_path = Path(output_file)
        if output_path.is_file(): 
            existing_file = pd.read_csv(output_file, header=None, squeeze=True)
            existing_file = pd.Series(existing_file)
            uniques = pd.Series(file[column].unique())
#            print(type(existing_file), type(uniques))
            combined_uniques = pd.Series( pd.concat([existing_file, uniques]).unique()) 
            combined_uniques.to_csv(output_dir + column +".txt", index=False)
            
        else: 
            uniques = pd.Series( file[column].unique() )
            uniques.to_csv(output_dir + column +".txt", index=False)
            
            
            

def categorial_file_mass_generator(input_dir, output_dir): 

    file_list = glob.glob(input_dir+"*.csv")
    
    for file in file_list: 
        start = time.time() 
        input_file = file 
        output_dir = output_dir  
        categorical_file_generator(input_file, output_dir)
        print("completed " + file )
        print("time is :" + str(time.time() - start))
        
parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", help="specify the training directory for use with categorial_file_mass_generator", type=str)
parser.add_argument("--output_dir", help="specify the output directory for use with categorial_file_mass_generator", type=str)
args = parser.parse_args()

categorial_file_mass_generator(args.train_dir, args.output_dir)
        
    
    
    

