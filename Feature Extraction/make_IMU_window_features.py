import pandas as pd
import numpy as np
import concurrent.futures
import os
from functools import partial


# Constants
SAMPLING_RATE = 148  # Hz

# Variables for feature extraction
WINDOW_DURATION_SEC = 20.0  # Duration of each window in seconds
WINDOW_SAMPLES = int(WINDOW_DURATION_SEC * SAMPLING_RATE)
STEP_DURATION_SEC = 20.0    # For non-overlapping windows, step == window duration
STEP_SAMPLES = int(STEP_DURATION_SEC * SAMPLING_RATE)

#Paths
FILTERED_DIR: str = "path/to/filtered/data"
OUT_ROOT: str = "path/to/windowed/IMU/features"

# Number of workers for parallel processing
N_WORKERS: int = 16



def calculate_mean(segment):
    return np.mean(segment)

def calculate_variance(segment):
    return np.var(segment)

def calculate_std_dev(segment):
    return np.std(segment)

def calculate_rms(segment):
    return np.sqrt(np.mean(segment**2))

def calculate_mav(segment):
    return np.mean(np.abs(segment))

def calculate_min_val(segment):
    return np.min(segment)

def calculate_max_val(segment):
    return np.max(segment)

def calculate_p2p(segment):
    return np.max(segment) - np.min(segment)

def calculate_zcr(segment):
    """Zero-Crossing Rate (count, not normalized rate for simplicity here)"""
    return np.sum(np.diff(np.sign(segment)) != 0)

def calculate_wl(segment):
    """Waveform Length"""
    return np.sum(np.abs(np.diff(segment)))

# Dictionary of feature functions
FEATURE_FUNCTIONS = {
    'Mean': calculate_mean,
    'Var': calculate_variance,
    'Std': calculate_std_dev,
    'RMS': calculate_rms,
    'MAV': calculate_mav,
    'Min': calculate_min_val,
    'Max': calculate_max_val,
    'P2P': calculate_p2p,
    'ZCR': calculate_zcr,
    'WL': calculate_wl,
}

def extract_features_for_participant_gesture(args):
    p, gesture, input_base_dir, output_base_dir = args
    """Process one filtered file to extract features."""
    
    in_path = f'{input_base_dir}/P{p}/Gesture_{gesture}.csv'
    
    # Create output directory if it doesn't exist
    out_dir_features = f'{output_base_dir}/P{p}'
    os.makedirs(out_dir_features, exist_ok=True)
    out_path_features = f'{out_dir_features}/Gesture_{gesture}_features.csv'

    try:
        df_filtered = pd.read_csv(in_path)
        if df_filtered.empty:
            print(f'Skipping empty file: {in_path}')
            return False

        # Identify numeric columns to extract features from
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
        time_cols = [col for col in df_filtered.columns if 'Time' in col or 'time' in col] 
        
        cols_to_extract_features_from = [ col for col in numeric_cols if col not in time_cols]
        
        if not cols_to_extract_features_from:
            print(f'No numeric columns to extract features from in {in_path} (after excluding time/label).')
            return False

        all_window_features = []
        
        num_windows = (len(df_filtered) - WINDOW_SAMPLES) // STEP_SAMPLES + 1

        for i in range(num_windows):
            start_idx = i * STEP_SAMPLES
            end_idx = start_idx + WINDOW_SAMPLES
            
            if end_idx > len(df_filtered): # Should not happen with the num_windows calculation
                break 
                
            window_df = df_filtered.iloc[start_idx:end_idx]
            
            current_window_feature_dict = {
                'participant': p,
                'gesture': gesture,
                'window_id': i,
            }
            
            
            if time_cols:
                # Try to find a primary time column, e.g., 'Time (s)' or the first one
                primary_time_col = next((tc for tc in time_cols if 'Time (s)' in tc), time_cols[0])
                current_window_feature_dict['window_start_time'] = window_df[primary_time_col].iloc[0]
                current_window_feature_dict['window_end_time'] = window_df[primary_time_col].iloc[-1]

            for sensor_col in cols_to_extract_features_from:
                segment = window_df[sensor_col].values
                if len(segment) < 2: # WL and ZCR need at least 2 points
                    for feat_name in FEATURE_FUNCTIONS:
                        current_window_feature_dict[f'{sensor_col}_{feat_name}'] = np.nan
                    continue

                for feat_name, feat_func in FEATURE_FUNCTIONS.items():
                    try:
                        value = feat_func(segment)
                        current_window_feature_dict[f'{sensor_col}_{feat_name}'] = value
                    except Exception as e_feat:
                        print(f"Warning: Error calculating {feat_name} for {sensor_col} in P{p}/G{gesture}, window {i}: {e_feat}")
                        current_window_feature_dict[f'{sensor_col}_{feat_name}'] = np.nan
            
            all_window_features.append(current_window_feature_dict)

        if not all_window_features:
            print(f'No windows generated for {in_path}. Possibly too short.')
            return False
        features_df = pd.DataFrame(all_window_features)
        
        # Sort the columns in alphabetical order for better readability
        features_df = features_df.reindex(sorted(features_df.columns), axis=1)
        features_df.to_csv(out_path_features, index=False)
        
        print(f'Successfully extracted features and saved: {out_path_features}')
        return True

    except FileNotFoundError:
        print(f'Filtered file not found: {in_path}')
        return False
    except pd.errors.EmptyDataError:
        print(f'Filtered file is empty: {in_path}')
        return False
    except Exception as e:
        print(f'Error extracting features for P{p}/Gesture{gesture}: {str(e)}')
        import traceback
        traceback.print_exc()
        return False

def main_feature_extraction():
    
    os.makedirs(OUT_ROOT, exist_ok=True)

    participants = []
    for p_num in range(1, 45):
        if p_num == 10 or p_num == 39 or p_num == 40:
            continue
        participants.append(p_num)
        
    tasks_for_features = []
    for p in participants:
        for g in range(1, 6): # Gestures 1 to 5
             tasks_for_features.append((p, g, FILTERED_DIR, OUT_ROOT))

    # Adjust max_workers based on your CPU cores
    num_workers = os.cpu_count() // 2 or 1 
    print(f"Using {num_workers} workers for feature extraction.")

    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(extract_features_for_participant_gesture, tasks_for_features))
    
    success_count = sum(r for r in results if r is True) # Filter out None or False
    total_tasks = len(tasks_for_features)
    print(f"\n--- Feature Extraction Summary ---")
    print(f"Successfully processed {success_count}/{total_tasks} files for feature extraction.")
    print(f"Failed {total_tasks - success_count} files.")

if __name__ == "__main__":
    main_feature_extraction()