import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
from sklearn.metrics import accuracy_score, classification_report
from transformers import Trainer, TrainingArguments
import wandb
import psutil
import gc
import time

# TODO: Random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# TODO: Paths 
BASE_IMU_DIR = 'path/to/your/filtered_IMU_ordered'
RESULTS_OUTPUT_DIR = f'path/to/your/results_output_directory_random_seed_{RANDOM_SEED}'
os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

# TODO: Wandb
WANDB_PROJECT_NAME = "IMU_Gesture_LOSO_CNNLSTM_MultiArray" 

# TODO: Hyperparameters
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01


# Data parameters : Don't change these
SAMPLING_RATE = 148  # Hz
WINDOW_DURATION_S = 20  # seconds
WINDOW_LENGTH = SAMPLING_RATE * WINDOW_DURATION_S 
NUM_SENSORS = 48 # IMU data has 48 columns/sensors
NUM_GESTURES = 5 

# --- Utility Functions ---
def monitor_memory():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)  # Convert to MB

def normalize_window_data(X_window):
    """
    Normalizes a single window of data (sensors x time_steps)
    Each sensor's time series is normalized independently.
    """
    mean = np.mean(X_window, axis=1, keepdims=True)
    std = np.std(X_window, axis=1, keepdims=True)
    std[std == 0] = 1.0 
    return (X_window - mean) / (std + 1e-7)


# --- Dataset Class (MultiArrayDataset) ---
# Needed for efficient memory usage
class MultiArrayDataset(Dataset):
    def __init__(self, feature_arrays_list, label_arrays_list):
        """
        Dataset that handles multiple arrays without full concatenation upfront.
        Args:
            feature_arrays_list: List of numpy arrays. Each array is expected to be
                                 (samples_in_this_array, num_sensors, time_steps).
            label_arrays_list: List of corresponding label numpy arrays. Each array is (samples_in_this_array,).
        """
        self.feature_arrays = [arr for arr in feature_arrays_list if arr.size > 0]
        self.label_arrays = [arr for arr in label_arrays_list if arr.size > 0]

        if not self.feature_arrays or not self.label_arrays:
            # Handle cases where one or both lists might be empty after filtering
            self.feature_arrays = []
            self.label_arrays = []
            self.cumulative_lengths = np.array([0])
            self._all_labels_concatenated = np.array([])
        else:
            self.cumulative_lengths = np.cumsum([len(arr) for arr in self.feature_arrays])
            self._all_labels_concatenated = None 

        if len(self.feature_arrays) != len(self.label_arrays):
            raise ValueError("Mismatch between number of feature arrays and label arrays.")
        for i in range(len(self.feature_arrays)):
            if len(self.feature_arrays[i]) != len(self.label_arrays[i]):
                raise ValueError(f"Mismatch in lengths for array set {i}: "
                                 f"{len(self.feature_arrays[i])} features vs {len(self.label_arrays[i])} labels.")

    def __len__(self):
        return self.cumulative_lengths[-1] if len(self.cumulative_lengths) > 0 and self.cumulative_lengths[-1] > 0 else 0

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of bounds")

        # Find the array that contains the index
        array_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')

        if array_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_lengths[array_idx - 1]

        features = self.feature_arrays[array_idx][sample_idx]
        labels = self.label_arrays[array_idx][sample_idx]

        return {
            'pixel_values': torch.tensor(features, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

    def get_all_labels(self):
        if self._all_labels_concatenated is None:
            if not self.label_arrays:
                self._all_labels_concatenated = np.array([])
            else:
                valid_label_arrays = [arr for arr in self.label_arrays if arr.size > 0]
                if not valid_label_arrays:
                     self._all_labels_concatenated = np.array([])
                else:
                    self._all_labels_concatenated = np.concatenate(valid_label_arrays)
        return self._all_labels_concatenated


# --- Model Definition (CNNLSTM_IMU) --- 
# This is the base model for the IMU data
class CNNLSTM_IMU(nn.Module):
    def __init__(self, num_sensors=NUM_SENSORS, sequence_length=WINDOW_LENGTH, num_classes=NUM_GESTURES):
        super(CNNLSTM_IMU, self).__init__()
        self.num_sensors = num_sensors
        
        self.conv_block = nn.Sequential(
            nn.Conv1d(num_sensors, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), 
            nn.Dropout(0.3),

            nn.Conv1d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), 
            nn.Dropout(0.3)
        )
        
        self.LSTM = nn.LSTM(
            input_size=256, 
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 128), 
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def forward(self, pixel_values=None, labels=None, **kwargs):
        x = self.conv_block(pixel_values)
        x = x.permute(0, 2, 1)
        self.LSTM.flatten_parameters() 
        x_LSTM_out, (hidden,_) = self.LSTM(x)
        # Handle bidirectional hidden states
        if self.LSTM.bidirectional:
            hidden = hidden.view(self.LSTM.num_layers, 2, x.size(0), self.LSTM.hidden_size)
            last_forward = hidden[-1, 0, :, :]  
            last_backward = hidden[-1, 1, :, :]  
            x = torch.cat([last_forward, last_backward], dim=1)  
        else:
            x = hidden[-1, :, :]  
        logits = self.fc(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        if loss is not None:
            return {'loss': loss, 'logits': logits}
        return {'logits': logits}

# Preprocess the participant data for the LOSO cross-validation
def preprocess_participant_data(base_imu_dir, participant_id, window_length=WINDOW_LENGTH, num_sensors=NUM_SENSORS):
    p_dir = os.path.join(base_imu_dir, participant_id)
    X_participant, y_participant = [], []
    
    for gesture_num in range(1, NUM_GESTURES + 1):
        file_path = os.path.join(p_dir, f'Gesture_{gesture_num}.csv')
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found {file_path}")
            continue
            
        try:
            df = pd.read_csv(file_path)
            if df.shape[1] != num_sensors:
                print(f"Warning: File {file_path} has {df.shape[1]} columns, expected {num_sensors}. Skipping.")
                continue

            sensor_data = df.values 
            num_windows = len(sensor_data) // window_length
            for i in range(num_windows):
                start_idx = i * window_length
                end_idx = start_idx + window_length
                window = sensor_data[start_idx:end_idx, :] 
                window_transposed = window.T 
                window_normalized = normalize_window_data(window_transposed)
                X_participant.append(window_normalized)
                y_participant.append(gesture_num - 1) 

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
            
    if not X_participant:
        return {'X': np.array([]), 'y': np.array([])}

    return {'X': np.array(X_participant), 'y': np.array(y_participant)}


# --- Training Function for LOSO ---
# All the data for all participants except the current participant is used for training
# The current participant is used for testing
def train_participant_model(current_p_id, all_participant_data, wandb_project_name):
    print(f"\n--- Training for test participant: {current_p_id} ---")
    print(f"Memory usage before this participant's training: {monitor_memory()} MB")

    train_features_list, train_labels_list = [], []
    test_features_list, test_labels_list = [], []

    for p_id, data in all_participant_data.items():
        if not data['X'].size: 
            print(f"Participant {p_id} has no data, skipping.")
            continue
        if p_id == current_p_id:
            test_features_list.append(data['X'])
            test_labels_list.append(data['y'])
        else:
            train_features_list.append(data['X'])
            train_labels_list.append(data['y'])

    if not train_features_list:
        print(f"No training data available when {current_p_id} is test subject. Skipping.")
        return 0.0, {}
    if not test_features_list: 
        print(f"No test data available for {current_p_id}. Skipping.")
        return 0.0, {}
        
    # Create datasets using MultiArrayDataset
    full_train_dataset = MultiArrayDataset(train_features_list, train_labels_list)
    test_dataset = MultiArrayDataset(test_features_list, test_labels_list)

    if len(full_train_dataset) == 0:
        print(f"Training dataset is empty for test participant {current_p_id}. Skipping.")
        return 0.0, {}
    if len(test_dataset) == 0:
        print(f"Test dataset for {current_p_id} is empty. Skipping.")
        return 0.0, {}
        
    print(f"Full train dataset size: {len(full_train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    train_indices = np.arange(len(full_train_dataset))
    all_train_labels_for_split = full_train_dataset.get_all_labels() # For stratification

    min_samples_for_split = max(2, NUM_GESTURES * 2) 
    if len(np.unique(all_train_labels_for_split)) < NUM_GESTURES or len(all_train_labels_for_split) < min_samples_for_split :
        print("Warning: Not enough samples or classes in training data for robust stratified validation split. Using random split or reducing test_size if possible.")
        try:
            val_test_size = 0.1 if len(all_train_labels_for_split) < 50 else 0.2
            train_idx, val_idx = train_test_split(
                train_indices,
                test_size=val_test_size, 
                random_state=42,
                stratify=all_train_labels_for_split if len(np.unique(all_train_labels_for_split)) > 1 else None
            )
        except ValueError as e:
            print(f"Stratified split failed: {e}. Falling back to non-stratified random split for validation.")
            train_idx, val_idx = train_test_split(
                train_indices, test_size=0.2, random_state=42 
            )
    else:
         train_idx, val_idx = train_test_split(
            train_indices,
            test_size=0.2,
            random_state=42,
            stratify=all_train_labels_for_split 
        )

    train_subset = Subset(full_train_dataset, train_idx)
    val_subset = Subset(full_train_dataset, val_idx)

    print(f"Train subset size: {len(train_subset)}")
    print(f"Validation subset size: {len(val_subset)}")
    if len(val_subset) == 0 and len(train_subset) > 0:
        print("Validation subset is empty, using training subset for validation as a fallback.")
        val_subset = train_subset 
    elif len(train_subset) == 0:
        print("Training subset is empty. Cannot proceed with training.")
        return 0.0, {}


    run = wandb.init(
        project=wandb_project_name, 
        name=f"run_test_P{current_p_id}", 
        config={
            "learning_rate": LEARNING_RATE, 
            "epochs": EPOCHS,         
            "batch_size": BATCH_SIZE,    
            "model_type": "CNN_LSTM_IMU_MultiArray",
            "test_participant": current_p_id
        },
        reinit=True 
    )

    model = CNNLSTM_IMU(num_sensors=NUM_SENSORS, sequence_length=WINDOW_LENGTH, num_classes=NUM_GESTURES)

    training_args = TrainingArguments(
        output_dir=f'{RESULTS_OUTPUT_DIR}/participant_{current_p_id}',
        per_device_train_batch_size=run.config.batch_size,
        per_device_eval_batch_size=run.config.batch_size,
        num_train_epochs=run.config.epochs,
        eval_strategy="epoch" if len(val_subset) > 0 else "no",
        save_strategy="epoch", 
        learning_rate=run.config.learning_rate,
        weight_decay=0.01,
        report_to="wandb", 
        load_best_model_at_end=True if len(val_subset) > 0 else False,
        metric_for_best_model="accuracy" if len(val_subset) > 0 else "loss", 
        greater_is_better=True if len(val_subset) > 0 else False, 
        save_total_limit=1, 
        logging_dir=f'{RESULTS_OUTPUT_DIR}/logs/participant_{current_p_id}',
        logging_steps=10,
    )

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {"accuracy": accuracy_score(p.label_ids, preds)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_subset,
        eval_dataset=val_subset if len(val_subset) > 0 else None,
        compute_metrics=compute_metrics if len(val_subset) > 0 else None
    )

    print("Starting training...")
    trainer.train()
    model.eval()
    print("Evaluating on test set...")
    predictions_output = trainer.predict(test_dataset)
    
    y_pred_test = np.argmax(predictions_output.predictions, axis=1)
    y_true_test = test_dataset.get_all_labels() 

    accuracy_test = accuracy_score(y_true_test, y_pred_test)
    report_test_dict = classification_report(y_true_test, y_pred_test, output_dict=True, zero_division=0, labels=np.arange(NUM_GESTURES))
    
    print(f"Test Accuracy for P{current_p_id}: {accuracy_test:.4f}")
    print("Test Classification Report:")
    print(classification_report(y_true_test, y_pred_test, zero_division=0, labels=np.arange(NUM_GESTURES)))


    wandb.log({
        "test_accuracy": accuracy_test,
        "test_classification_report": report_test_dict
    })
    run.finish()

    del model, trainer, full_train_dataset, test_dataset, train_subset, val_subset
    del train_features_list, train_labels_list, test_features_list, test_labels_list
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Memory usage after participant {current_p_id} cleanup: {monitor_memory()} MB")
    
    return accuracy_test, report_test_dict


# --- Main Function for LOSO Cross-Validation ---
def main():
    

    os.makedirs(os.path.join(RESULTS_OUTPUT_DIR, "accuracy_plots"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_OUTPUT_DIR, "classification_reports"), exist_ok=True)

    participant_dirs = sorted([
        d for d in os.listdir(BASE_IMU_DIR)
        if os.path.isdir(os.path.join(BASE_IMU_DIR, d)) and d.startswith('P')
    ])
    if not participant_dirs:
        print(f"Error: No participant directories (e.g., P1, P2) found in {BASE_IMU_DIR}")
        return
    
    print(f"Found participants: {participant_dirs}")

    print("\nPreprocessing all participant data...")
    all_participant_data = {}
    initial_mem = monitor_memory()
    print(f"Memory usage before loading all data: {initial_mem:.2f} MB")

    for p_id in tqdm(participant_dirs, desc="Loading data"):
        all_participant_data[p_id] = preprocess_participant_data(BASE_IMU_DIR, p_id)
        time.sleep(0.01) 
    
    loaded_mem = monitor_memory()
    print(f"Memory usage after loading all data: {loaded_mem:.2f} MB (Increased by {loaded_mem - initial_mem:.2f} MB)")

    results_summary = []
    for current_p_id_test in participant_dirs:

        accuracy, report = train_participant_model(current_p_id_test, all_participant_data, WANDB_PROJECT_NAME)
        
        if report and 'macro avg' in report: 
            results_summary.append({
                'Participant_Test': current_p_id_test,
                'Accuracy': accuracy,
                'Macro_Avg_Precision': report['macro avg']['precision'],
                'Macro_Avg_Recall': report['macro avg']['recall'],
                'Macro_Avg_F1': report['macro avg']['f1-score'],
                'Weighted_Avg_Precision': report['weighted avg']['precision'],
                'Weighted_Avg_Recall': report['weighted avg']['recall'],
                'Weighted_Avg_F1': report['weighted avg']['f1-score']
            })

            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(os.path.join(RESULTS_OUTPUT_DIR, "classification_reports", f'report_{current_p_id_test}.csv'))

            plt.figure(figsize=(6, 4))
            plt.bar([f'P{current_p_id_test} Test'], [accuracy], color='darkcyan')
            plt.ylim(0, 1)
            plt.ylabel('Accuracy')
            plt.title(f'Test Accuracy (Participant {current_p_id_test} as Test)')
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_OUTPUT_DIR, "accuracy_plots", f'accuracy_{current_p_id_test}.png'))
            plt.close()
        else:
            print(f"Skipping results summary for {current_p_id_test} due to empty or invalid report.")

        
        gc.collect()
        torch.cuda.empty_cache()

    if results_summary:
        results_df = pd.DataFrame(results_summary)
        
        mean_values = results_df.drop(columns=['Participant_Test'], errors='ignore').mean(numeric_only=True)
        mean_row_dict = mean_values.to_dict()
        mean_row_dict['Participant_Test'] = 'Mean_LOSO'
        mean_row_df = pd.DataFrame([mean_row_dict])
        
        results_df = pd.concat([results_df, mean_row_df], ignore_index=True)
        
        results_df.to_csv(os.path.join(RESULTS_OUTPUT_DIR, 'loso_imu_summary_accuracies.csv'), index=False)
        print("\n--- LOSO Summary ---")
        print(results_df)

        avg_accuracy_overall_series = results_df[results_df['Participant_Test'] == 'Mean_LOSO']['Accuracy']
        if not avg_accuracy_overall_series.empty:
            avg_accuracy_overall = avg_accuracy_overall_series.iloc[0]
            
            plt.figure(figsize=(max(8, len(participant_dirs)*0.5) , 5)) 
            participant_accuracies = results_df[results_df['Participant_Test'] != 'Mean_LOSO']['Accuracy']
            participant_names = results_df[results_df['Participant_Test'] != 'Mean_LOSO']['Participant_Test']
            plt.bar(participant_names, participant_accuracies, color='skyblue')
            plt.axhline(avg_accuracy_overall, color='red', linestyle='--', label=f'Mean Accuracy: {avg_accuracy_overall:.2f}')
            plt.xlabel('Test Participant')
            plt.ylabel('Accuracy')
            plt.title('LOSO Cross-Validation Accuracy per Participant')
            plt.xticks(rotation=45, ha="right")
            plt.ylim(0, 1)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_OUTPUT_DIR, 'loso_overall_accuracy_comparison.png'))
            plt.close()

            print(f"\nOverall Mean LOSO Accuracy: {avg_accuracy_overall:.4f}")
        else:
            print("Could not calculate overall mean accuracy.")
    else:
        print("No results were generated. Check for errors in the logs.")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Training will run on CPU, which will be very slow.")
    else:
        print(f"CUDA is available. PyTorch CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        current_cuda_device = os.environ.get("CUDA_VISIBLE_DEVICES", "All (default)")
        print(f"CUDA_VISIBLE_DEVICES is set to: {current_cuda_device}")
    main()