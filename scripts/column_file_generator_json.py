#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 18:37:48 2018

@author: james.f.xue

Script to create txt files for all columns

1. Need to be able to update the file with new file if present
2. Need to be able to use product in categorical_column_with_vocabulary_file 
"""

import pandas as pd 
import numpy as np
import os.path 
from pathlib import Path
import glob
import time
import argparse
import json 


def column_file_mass_generator(input_dir, output_dir): 

    """
    A script to generate: 
    1) a .json file for each categorical column and 
    2) another .json file for summary statistics 
    of the file_dir.
    
    Parameters
    ==========
    input_file: a csv file of ctr data
    output_dir: a directory to output .json files
                
    >>> Usage Example: 
        input_file = "/Users/james.f.xue/Desktop/work/raw/"
        output_dir = "/Users/james.f.xue/Desktop/work/processed/"  
        column_file_mass_generator(input_file, output_dir)

    >>> Command Line Examples: 
        python column_file_generator_json.py --train_dir path/to/directory/csv --output_dir path/to/directory/
        python column_file_generator_json.py --train_dir "/Users/james.f.xue/Desktop/work/raw/" --output_dir "/Users/james.f.xue/Desktop/work/processed/"
    """

    #get a list of files and the list of headers of the files 
    file_list = glob.glob(os.path.join(input_dir, "*.csv"))    
    first_file = pd.read_csv(file_list[0], ",")
    headers = np.array(first_file.columns)

    cat_columns = [
            'exchange_id', 'user_frequency', 'site_id', 'deal_id', 'channel_type',
            'size', 'week_part', 'day_of_week', 'dma_id', 'isp_id', 'fold_position',
            'browser_language_id', 'country_id', 'conn_speed', 'os_id',
            'day_part', 'region_id', 'browser_id', 'hashed_app_id', 'interstitial',
            'device_id', 'creative_id', 'browser', 'browser_version', 'os', 'os_version',
            'device_model', 'device_manufacturer', 'device_type', 'exchange_id_cs_vcr',
            'exchange_id_cs_vrate', 'exchange_id_cs_ctr', 'exchange_id_cs_category_id',
            'exchange_id_cs_site_id', 'category_id', 'cookieless', 'cross_device', 
            'conversion_target']

    num_columns = ['id_vintage', 'exchange_viewability_rate', 'exchange_ctr', 'exchange_vcr', 'column_weights']
    num_columns.extend([column for column in headers if column.endswith(('_bpr', '_bpf', '_pixel'))])

    #initialize dictionary 
    dictionary = dict.fromkeys(cat_columns) 
    num_dictionary = dict.fromkeys(num_columns)

    for file in file_list: 
        #timing start
        start = time.time()  
        
        #load file into pandas 
        current_file = pd.read_csv(file, ",")

        #categorical 
        for column in cat_columns: 
            if dictionary[column] is not None: 
                uniques = pd.Series(current_file[column].unique())
                existing_uniques = pd.Series(dictionary[column])
                combined_uniques = pd.concat([existing_uniques, uniques]).unique()
                dictionary[column] = combined_uniques 
            
            else: 
                uniques = current_file[column].unique()
                dictionary[column] = uniques

        #numerical 
        #keep track of n and sum to get the average
        for num_column in num_columns: 
            if num_dictionary[num_column] is not None: 
                num_dictionary[num_column]['n'] += len(current_file[num_column])
                num_dictionary[num_column]['sum'] += float(sum(current_file[num_column]))
            else: 
                num_dictionary[num_column] = {'mean': 0.0, 'std': 0.0, 'sum': 0.0, 'sq_diff': 0.0, 'n': 0}
                num_dictionary[num_column]['n'] += len(current_file[num_column])
                num_dictionary[num_column]['sum'] += float(sum(current_file[num_column]))

        #timing end 
        print("completed categorical columns of fi" + file )
        print("time is :" + str(time.time() - start))

    #calculate mean 
    for num_column in num_columns: 
        num_dictionary[num_column]['mean'] = num_dictionary[num_column]['sum'] / num_dictionary[num_column]['n']

    #loop again to get the standard deviation 
    for file in file_list: 
        start = time.time() 

        #load file into pandas 
        current_file = pd.read_csv(file, ",")

        #now calculate <<REVIEW>>
        for num_column in num_columns: 
            mean_removed_column = np.subtract(current_file[num_column], num_dictionary[num_column]['mean']) 
            num_dictionary[num_column]['sq_diff'] += np.dot(mean_removed_column, mean_removed_column)

        #timing end 
        print("completed numerical columns of " + file )
        print("time is :" + str(time.time() - start))
    
    num_dictionary_2 = dict.fromkeys(num_columns)
    #calculate standard deviation 
    for num_column in num_columns: 
        num_dictionary[num_column]['std'] = np.sqrt(num_dictionary[num_column]['sq_diff']/num_dictionary[num_column]['n']) 
        num_dictionary_2[num_column] = {'mean': num_dictionary[num_column]['mean'], 'std': num_dictionary[num_column]['std']}

    #JSON can't encode numpy arrays. Convert numpy to list. 
    keys = dictionary.keys() 
    for key in keys: 
        dictionary[key] = dictionary[key].tolist() 

    #write to JSON file 
    with open(output_dir+'categorical-vocab.json', 'w') as outfile:
        json.dump(dictionary, outfile, indent=2)

    with open(output_dir+'numerical-stats.json', 'w') as outfile_2:
        json.dump(num_dictionary_2, outfile_2, indent=2)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", help="specify the training directory for use with categorial_file_mass_generator", type=str)
    parser.add_argument("--output_dir", help="specify the output directory for use with categorial_file_mass_generator", type=str)
    args = parser.parse_args()

    column_file_mass_generator(args.train_dir, args.output_dir)
        
    
    
    

