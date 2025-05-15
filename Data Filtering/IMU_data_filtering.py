import pandas as pd
import numpy as np
import concurrent.futures
import os
from functools import partial
from scipy import signal
import csv
from functools import partial

# Precompute constants
EXPECTED_ROWS = 20 * 60 * 148  # 20 minutes * 60 seconds * 148 Hz
SAMPLING_RATE = 148  # Hz
CUTOFF_FREQ = 20  # Hz

# TODO: Directory for Saving and Reading Data
OUTPUT_DIR = '/path/to/saving/IMU_Filtered_Data'
INPUT_DIR = '/path/to/reading/Huggingface_Data'

def process_file_wrapper(args):
    return process_file(*args)

def apply_lowpass_filter(data, fs=SAMPLING_RATE, cutoff=CUTOFF_FREQ):
    """Apply a low-pass Butterworth filter to the data"""
    # Design the Butterworth filter
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    # 4th order Butterworth filter
    b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
    
    # Apply the filter
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def process_file(p, gesture):
    """Process a file by applying a low-pass filter and saving to a new location"""
    try:
        # Input and output paths
        in_path = f'{INPUT_DIR}/P{p}/Gesture_{gesture}.csv'
        out_path = f'{OUTPUT_DIR}/P{p}/Gesture_{gesture}_IMU_filtered.csv'
        # Create output directory if it doesn't exist
        out_dir = f'{OUTPUT_DIR}/P{p}'
        os.makedirs(out_dir, exist_ok=True)
        
        # Read the data
        df = pd.read_csv(in_path, low_memory=False)
        
        
        cols = df.columns
        
        # select only the IMU columns
        imu_cols = [col for col in cols if 'ACC' or 'GYR' in col]
        df = df[imu_cols]
        
        time_cols = [col for col in imu_cols if 'Time' in col or 'time' in col]
        cols_to_filter = [col for col in imu_cols if col not in time_cols]
        
        for col in cols_to_filter:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with any NaNs (from non-numeric values or missing data)
        df = df.dropna()
        
        # filter the data
        for col in cols_to_filter:
            df[col] = apply_lowpass_filter(df[col].values)

        # Save the filtered data
        df.to_csv(out_path, index=False)
        
        print(f'Successfully filtered and saved: P{p}/Gesture{gesture}')
        
        
    
    except FileNotFoundError:
        print(f'File not found: {in_path}')
        return False
    except Exception as e:
        print(f'Error processing P{p}/Gesture{gesture}: {str(e)}')
        return False

def main():
    # Create the output directory if it doesn't exist
    os.makedirs('./filtered_data', exist_ok=True)
    participants = []
    for p in range(1, 45):
        if p==10 or p==39 or p==40:
            continue
        participants.append(p)
        
    
    tasks = [(p, g) for p in participants for g in range(1, 6)]
    
    # Use process pool for CPU-bound operations
    # with concurrent.futures.ProcessPoolExecutor(16) as executor:
    #     results = list(executor.map(lambda args: process_file(*args), tasks))
        
    # with concurrent.futures.ProcessPoolExecutor(16) as executor:
    #     results = list(executor.map(process_file_wrapper, tasks))
    
    # with concurrent.futures.ProcessPoolExecutor(16) as executor:
    #     results = list(executor.map(lambda args: process_file(*args), tasks))
    
    # Unpack tasks into separate lists
    ps, gs = zip(*tasks)
    
    with concurrent.futures.ProcessPoolExecutor(16) as executor:
        results = list(executor.map(process_file, ps, gs))
    
    success_count = sum(results)
    print(f"Successfully processed {success_count}/{len(tasks)} files")
    print(f"Failed {len(tasks) - success_count} files")

if __name__ == "__main__":
    main()
