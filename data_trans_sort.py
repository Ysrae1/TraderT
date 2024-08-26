import os
import re
import sys
import random
import pandas as pd
import h5py
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def process_file(filename,process_param):
    
    df = pd.read_csv(filename, header=None)

    if process_param == 'lag_lead': # append a new column contains the prices of next step
        inputs = df.iloc[:-1, :14].values
        outputs = df.iloc[1:, 13].values
        # diffs = np.diff(inputs_[:, 0])
        # end_of_cycle_indices = np.where(diffs < 0)[0]

        # inputs = np.delete(inputs_, end_of_cycle_indices, axis=0)
        # outputs = np.delete(outputs_, end_of_cycle_indices, axis=0)

    if process_param == 'normal':
        inputs = df.iloc[:, :13].values
        outputs = df.iloc[:, 13].values

    return filename, inputs, outputs

def main_train(process_param):

    if process_param == 'lag_lead':
        dim = 14
    else:
        dim = 13

    src_dir = 'simulator/backup/lob_data_train_s'
    output_dir = 'data/lob_data'
    hdf5_filename = 'lob_data_train_s.h5'
    
    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith('.csv')]
    csv_files_sorted = sorted(csv_files, key=lambda x: int(x.split('-')[0].split(os.sep)[-1]))

    results = []

    with ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_file, file, process_param): file for file in csv_files_sorted}
        for future in tqdm(as_completed(future_to_file), total=len(csv_files_sorted), desc="Processing data files"):
            results.append(future.result())

    with h5py.File(os.path.join(output_dir, hdf5_filename), 'w') as h5f:
        input_dataset = h5f.create_dataset('inputs', shape=(0, dim), maxshape=(None, dim), dtype='float32', compression="gzip")
        output_dataset = h5f.create_dataset('outputs', shape=(0,), maxshape=(None,), dtype='float32', compression="gzip")

        for filename, inputs, outputs in sorted(results, key=lambda x: int(x[0].split('-')[0].split(os.sep)[-1])):
            current_size = input_dataset.shape[0]
            additional_size = inputs.shape[0]
            input_dataset.resize(current_size + additional_size, axis=0)
            output_dataset.resize(current_size + additional_size, axis=0)
            input_dataset[current_size:] = inputs
            output_dataset[current_size:] = outputs
            print(filename)

def main_ve_te(process_param):
        
    if process_param == 'lag_lead':
        dim = 14
    else:
        dim = 13

    src_dir = 'simulator/backup/lob_data_verif_test'
    output_dir = 'data/lob_data'
    hdf5_filename_v = 'lob_data_verif_s.h5'
    hdf5_filename_t = 'lob_data_test_s.h5'
    
    os.makedirs(output_dir, exist_ok=True)
    
    pattern = r'(\d+)-'

    files_by_number = {}

    for filename in os.listdir(src_dir):
        if filename.endswith('.csv'):
            match = re.match(pattern, filename)
            if match:
                number = match.group(1)
                if number not in files_by_number:
                    files_by_number[number] = []
                files_by_number[number].append(filename)

    
    v_group = []
    t_group = []

    for number, files in files_by_number.items():
        if len(files) % 2 == 0:  
            m = len(files)
            random.shuffle(files) 
            mid_index = m // 2

            v_group.extend(files[mid_index:])
            t_group.extend(files[:mid_index])
            
    v_group_paths = [os.path.join(src_dir, f) for f in v_group]
    t_group_paths = [os.path.join(src_dir, f) for f in t_group]
    

    print('len_verif:', len(v_group_paths), ', len_test:', len(t_group_paths))
    
    with h5py.File(os.path.join(output_dir, hdf5_filename_v), 'w') as h5f:
        
        input_dataset = h5f.create_dataset('inputs', shape=(0, dim), maxshape=(None, dim), dtype='float32', compression="gzip")
        output_dataset = h5f.create_dataset('outputs', shape=(0,), maxshape=(None,), dtype='float32', compression="gzip")
        
        with ThreadPoolExecutor() as executor:
            
            future_to_file = {executor.submit(process_file, file, process_param): file for file in v_group_paths}
            
            for future in tqdm(as_completed(future_to_file), total=len(v_group_paths), desc="Processing data_verif files"):
                file = future_to_file[future]
                try:
                    filename, inputs, outputs = future.result()
                    
                    current_size = input_dataset.shape[0]
                    
                    additional_size = inputs.shape[0]
                    
                    input_dataset.resize(current_size + additional_size, axis=0)
                    output_dataset.resize(current_size + additional_size, axis=0)
                    
                    input_dataset[current_size:] = inputs
                    output_dataset[current_size:] = outputs
                except Exception as exc:
                    print('%r generated an exception: %s' % (file, exc))

    with h5py.File(os.path.join(output_dir, hdf5_filename_t), 'w') as h5f:
        
        input_dataset = h5f.create_dataset('inputs', shape=(0, dim), maxshape=(None, dim), dtype='float32', compression="gzip")
        output_dataset = h5f.create_dataset('outputs', shape=(0,), maxshape=(None,), dtype='float32', compression="gzip")
        
        with ThreadPoolExecutor() as executor:
            
            future_to_file = {executor.submit(process_file, file, process_param): file for file in t_group_paths}
            
            for future in tqdm(as_completed(future_to_file), total=len(t_group_paths), desc="Processing data_test files"):
                file = future_to_file[future]
                try:
                    filename, inputs, outputs = future.result()
                    
                    current_size = input_dataset.shape[0]
                    
                    additional_size = inputs.shape[0]
                    
                    input_dataset.resize(current_size + additional_size, axis=0)
                    output_dataset.resize(current_size + additional_size, axis=0)
                    
                    input_dataset[current_size:] = inputs
                    output_dataset[current_size:] = outputs
                except Exception as exc:
                    print('%r generated an exception: %s' % (file, exc))


if __name__ == '__main__':

    # sys.argv[1] options: 'train' or 've_te'
    # sys.argv[2] options: 'lag_lead' or 'normal'

    if len(sys.argv) < 3:
        print('Input in the format of \'python3 data_trans.py {train or ve_te} {lag_lead or normal}\', otherwise it would do all 3 using lag_lead.')
        main_train('lag_lead')
        main_ve_te('lag_lead')
    else:
        if sys.argv[1] == 'train':
            main_train(sys.argv[2])
        if sys.argv[1] == 've_te':
            main_ve_te(sys.argv[2])    