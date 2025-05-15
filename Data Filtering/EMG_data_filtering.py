import pandas as pd
import numpy as np
import concurrent.futures
import os
from functools import partial
from scipy import signal
import csv
from functools import partial

# Precompute constants
EXPECTED_ROWS = 20 * 60 * 1259  # 20 minutes * 60 seconds * 148 Hz
SAMPLING_RATE = 1259  # Hz
LOW_CUTOFF = 50  # Hz
HIGH_CUTOFF = 450  # Hz

# TODO: Directory for Saving and Reading Data
OUTPUT_DIR = '/path/to/saving/EMG_Filtered_Data'
INPUT_DIR = '/path/to/reading/Huggingface_Data'

def process_file_wrapper(args):
    return process_file(*args)

def apply_bandpass_filter(data, fs=SAMPLING_RATE, low_cutoff=LOW_CUTOFF, high_cutoff=HIGH_CUTOFF):
    """Apply a bandpass Butterworth filter to the data"""
    # Design the Butterworth filter
    nyquist = 0.5 * fs
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    
    # 4th order Butterworth bandpass filter
    sos = signal.butter(4, [low, high], btype='bandpass', analog=False, output='sos')
    
    # Apply the filter using sosfiltfilt for zero-phase filtering
    filtered_data = signal.sosfiltfilt(sos, data)
    return filtered_data

def process_file(p, gesture):
    """Process a file by applying a low-pass filter and saving to a new location"""
    try:
        # Input and output paths
        in_path = f'{INPUT_DIR}/P{p}/Gesture_{gesture}.csv'
        # Create output directory if it doesn't exist
        out_dir = f'{OUTPUT_DIR}/P{p}'
        os.makedirs(out_dir, exist_ok=True)
        out_path = f'{out_dir}/Gesture_{gesture}_EMG_filtered.csv'
        
        # Read the data
        df = pd.read_csv(in_path, low_memory=False)
        
        # Identify numeric columns to filter (excluding time  columns)
        cols = df.columns
        
        # select only the EMG columns
        emg_cols = [col for col in cols if 'EMG' in col]
        df = df[emg_cols]
        
        time_cols = [col for col in emg_cols if 'Time' in col or 'time' in col]
        cols_to_filter = [col for col in emg_cols if col not in time_cols]
        
        for col in cols_to_filter:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with any NaNs (from non-numeric values or missing data)
        df = df.dropna()
        
        # filter the data
        for col in cols_to_filter:
            df[col] = apply_bandpass_filter(df[col].values)
        
        # Save the filtered data
        df.to_csv(out_path, index=False)
        
        print(f'Successfully filtered and saved: P{p}/Gesture{gesture}')
        return True
        
    except FileNotFoundError:
        print(f'File not found: {in_path}')
        return False
    except Exception as e:
        print(f'Error processing P{p}/Gesture{gesture}: {str(e)}')
        return False

def main():
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    participants = []
    for p in range(1, 45):
        if p==10 or p==39 or p==40:
            continue
        participants.append(p)
        
    
    tasks = [(p, g) for p in participants for g in range(1, 6)]
    
    ps, gs = zip(*tasks)
    # Multiprocessing for faster processing
    with concurrent.futures.ProcessPoolExecutor(16) as executor:
        results = list(executor.map(process_file, ps, gs))
    
    success_count = sum(results)
    print(f"Successfully processed {success_count}/{len(tasks)} files")
    print(f"Failed {len(tasks) - success_count} files")

if __name__ == "__main__":
    main()
