import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
from sklearn.metrics import accuracy_score, classification_report
from transformers import Trainer, TrainingArguments
import wandb
import psutil
import gc
import time
from safetensors.torch import load_file
from pathlib import Path
from typing import Tuple, List 

RANDOM_STATE = 42
# Set random seed
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed(RANDOM_STATE)

# --- Configuration for IMU part ---
SAMPLING_RATE_IMU = 148  # Hz
WINDOW_DURATION_S_IMU = 20  # seconds
WINDOW_LENGTH_IMU = SAMPLING_RATE_IMU * WINDOW_DURATION_S_IMU 
NUM_SENSORS_IMU = 48 
NUM_GESTURES = 5 

# --- Configuration for EMG part (from contrastive learning script) ---
NUM_EMG_SENSORS = 8
FEATURES_PER_EMG_SENSOR = 11
TOTAL_FEATURES_EMG = NUM_EMG_SENSORS * FEATURES_PER_EMG_SENSOR  
EMG_ENCODER_DIM = 128 
EMG_PROJECTION_DIM = 64 

# --- Configuration for Combined Model (Teacher) & Training ---
COMBINED_OUTPUT_DIR = f'./loso_results_combined_IMU_EMG_FIXED_ordered_LSTM_random_state_{RANDOM_STATE}' 
BASE_IMU_DIR = '/home/tih/kirtil/neurips/filtered_IMU_ordered' 
BASE_EMG_FEATURE_DIR = '/home/tih/kirtil/neurips/features_by_window_ordered' 
# --- Configuration for Student Model & Distillation ---
STUDENT_OUTPUT_DIR = f'./loso_results_student_IMU_distilled_FIXED_LSTM_random_state_{RANDOM_STATE}'
WANDB_PROJECT_NAME_STUDENT = f"IMU_Gesture_LOSO_Student_Distilled_FIXED_LSTM_random_state_{RANDOM_STATE}"
KD_ALPHA = 0.3  
KD_TEMPERATURE = 2.0 

# --- Device Configuration ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using CUDA ({torch.cuda.device_count()} device(s))")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU – expect slow training …")
PIN_MEMORY = torch.cuda.is_available()


# --- Utility Functions (from both scripts) ---
def monitor_memory(tag=""):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    rss_mb = memory_info.rss / (1024 * 1024)
    print(f"[MEM] {tag:<25} {rss_mb:8.1f} MB")
    return rss_mb

def normalize_window_data_imu(X_window):
    mean = np.mean(X_window, axis=1, keepdims=True)
    std = np.std(X_window, axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (X_window - mean) / (std + 1e-7)

class IdentityScaler:
    def fit(self, X): return self
    def transform(self, X): return X
    def fit_transform(self, X): return X

# --- EMG Encoder Definition ---
class ContrastiveFatigueEncoder(nn.Module):
    def __init__(self,
                 num_features: int = TOTAL_FEATURES_EMG,
                 encoder_dim: int = EMG_ENCODER_DIM,
                 projection_dim: int = EMG_PROJECTION_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, encoder_dim), nn.BatchNorm1d(encoder_dim),
        )
        self.proj = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim // 2), nn.ReLU(),
            nn.Linear(encoder_dim // 2, projection_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.encoder(x)
        proj = self.proj(enc)
        return enc, proj

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# --- Combined Dataset Class (Unchanged) ---
class CombinedDataset(Dataset):
    def __init__(self, imu_features_list, emg_features_list, label_arrays_list, emg_scaler):
        self.imu_feature_arrays = [arr for arr in imu_features_list if arr.size > 0]
        self.emg_feature_arrays = [arr for arr in emg_features_list if arr.size > 0]
        self.label_arrays = [arr for arr in label_arrays_list if arr.size > 0]
        self.emg_scaler = emg_scaler

        if not (self.imu_feature_arrays and self.emg_feature_arrays and self.label_arrays):
            self.cumulative_lengths = np.array([0])
            self._all_labels_concatenated = np.array([])
        else:
            if not (len(self.imu_feature_arrays) == len(self.emg_feature_arrays) == len(self.label_arrays)):
                raise ValueError("Mismatch in the number of IMU, EMG, and label arrays.")
            self.cumulative_lengths = np.cumsum([len(arr) for arr in self.imu_feature_arrays])
            self._all_labels_concatenated = None

        for i in range(len(self.imu_feature_arrays)):
            if not (len(self.imu_feature_arrays[i]) == len(self.emg_feature_arrays[i]) == len(self.label_arrays[i])):
                raise ValueError(f"Mismatch in lengths for array set {i}: "
                                 f"{len(self.imu_feature_arrays[i])} IMU vs "
                                 f"{len(self.emg_feature_arrays[i])} EMG vs "
                                 f"{len(self.label_arrays[i])} labels.")

    def __len__(self):
        return self.cumulative_lengths[-1] if len(self.cumulative_lengths) > 0 and self.cumulative_lengths[-1] > 0 else 0

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self): raise IndexError("Index out of bounds")
        array_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        sample_idx = idx - (self.cumulative_lengths[array_idx - 1] if array_idx > 0 else 0)

        imu_features = self.imu_feature_arrays[array_idx][sample_idx]
        emg_features_raw = self.emg_feature_arrays[array_idx][sample_idx]
        labels = self.label_arrays[array_idx][sample_idx]
        scaled_emg_features = self.emg_scaler.transform(emg_features_raw.reshape(1, -1)).flatten()

        return {
            'imu_pixel_values': torch.tensor(imu_features, dtype=torch.float32),
            'emg_features': torch.tensor(scaled_emg_features, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

    def get_all_labels(self):
        if self._all_labels_concatenated is None:
            if not self.label_arrays: self._all_labels_concatenated = np.array([])
            else:
                valid_label_arrays = [arr for arr in self.label_arrays if arr.size > 0]
                if not valid_label_arrays: self._all_labels_concatenated = np.array([])
                else: self._all_labels_concatenated = np.concatenate(valid_label_arrays)
        return self._all_labels_concatenated

# --- Combined Model Definition (Teacher Model - Unchanged) ---
class CombinedGestureModel(nn.Module):
    def __init__(self, num_imu_sensors, imu_sequence_length, num_gestures,
                 emg_fatigue_encoder, 
                 emg_embedding_dim,
                 freeze_emg_encoder=True):
        super(CombinedGestureModel, self).__init__()

        # EMG Branch (Pre-trained Encoder)
        self.emg_fatigue_encoder = emg_fatigue_encoder
        if freeze_emg_encoder:
            for param in self.emg_fatigue_encoder.parameters():
                param.requires_grad = False
        self.emg_embedding_dim = emg_embedding_dim

        # IMU Branch (CNN-LSTM from baseline)
        self.imu_conv_block = nn.Sequential(
            nn.Conv1d(num_imu_sensors, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(kernel_size=2, stride=2), nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(kernel_size=2, stride=2), nn.Dropout(0.3),
            nn.Conv1d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(kernel_size=2, stride=2), nn.Dropout(0.3)
        )
        self.imu_LSTM = nn.LSTM(
            input_size=256, hidden_size=128, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.3
        )
        imu_LSTM_output_dim = 128 * 2  

        # Fusion and Classifier
        self.fc = nn.Sequential(
            nn.Linear(imu_LSTM_output_dim + self.emg_embedding_dim, 256), 
            nn.ReLU(), nn.Dropout(0.5), 
            nn.Linear(256, num_gestures)
        )
        self._init_weights_custom() # Initialize IMU branch and FC layers

    def _init_weights_custom(self):
        for m_name, m in self.named_modules():
            if m_name.startswith("emg_fatigue_encoder"): 
                continue
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name: nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name: nn.init.orthogonal_(param.data)
                    elif 'bias' in name: param.data.fill_(0)

    def forward(self, imu_pixel_values=None, emg_features=None, labels=None, **kwargs):
        x_imu = self.imu_conv_block(imu_pixel_values) 
        x_imu = x_imu.permute(0, 2, 1) 
        
        self.imu_LSTM.flatten_parameters()
        _,( hidden_imu,_) = self.imu_LSTM(x_imu) 

        hidden_imu = hidden_imu.view(self.imu_LSTM.num_layers, 2 if self.imu_LSTM.bidirectional else 1,
                                     x_imu.size(0), self.imu_LSTM.hidden_size)
        last_forward_imu = hidden_imu[-1, 0, :, :]  
        last_backward_imu = hidden_imu[-1, 1, :, :] 
        imu_representation = torch.cat([last_forward_imu, last_backward_imu], dim=1) 

        if self.emg_fatigue_encoder.training and not any(p.requires_grad for p in self.emg_fatigue_encoder.parameters()):
            self.emg_fatigue_encoder.eval()
            emg_embedding = self.emg_fatigue_encoder.get_embedding(emg_features) 
            self.emg_fatigue_encoder.train() 
        else:
            emg_embedding = self.emg_fatigue_encoder.get_embedding(emg_features)


        # Fusion
        combined_representation = torch.cat((imu_representation, emg_embedding), dim=1)

        logits = self.fc(combined_representation)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        if loss is not None:
            return {'loss': loss, 'logits': logits}
        return {'logits': logits}



# --- Student Model Definition (IMU-only) ---
class StudentIMUModel(nn.Module):
    def __init__(self, num_imu_sensors, imu_sequence_length, num_gestures):
        super(StudentIMUModel, self).__init__()
        # IMU Branch (similar to teacher's IMU branch)
        self.imu_conv_block = nn.Sequential(
            nn.Conv1d(num_imu_sensors, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(kernel_size=2, stride=2), nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(kernel_size=2, stride=2), nn.Dropout(0.3),
            nn.Conv1d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(kernel_size=2, stride=2), nn.Dropout(0.3)
        )
        self.imu_LSTM = nn.LSTM(
            input_size=256, hidden_size=128, num_layers=2, # Can be made smaller if desired
            batch_first=True, bidirectional=True, dropout=0.3
        )
        imu_LSTM_output_dim = 128 * 2  # Bidirectional

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(imu_LSTM_output_dim, 256), # Only IMU features
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_gestures)
        )
        self._init_weights_custom()

    def _init_weights_custom(self): # Same initialization as teacher's relevant parts
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name: nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name: nn.init.orthogonal_(param.data)
                    elif 'bias' in name: param.data.fill_(0)

    def forward(self, imu_pixel_values=None, labels=None, **kwargs):
        x_imu = self.imu_conv_block(imu_pixel_values)
        x_imu = x_imu.permute(0, 2, 1)

        self.imu_LSTM.flatten_parameters()
        _, (hidden_imu,_) = self.imu_LSTM(x_imu)

        hidden_imu = hidden_imu.view(self.imu_LSTM.num_layers, 2, x_imu.size(0), self.imu_LSTM.hidden_size)
        last_forward_imu = hidden_imu[-1, 0, :, :]
        last_backward_imu = hidden_imu[-1, 1, :, :]
        imu_representation = torch.cat([last_forward_imu, last_backward_imu], dim=1)

        logits = self.fc(imu_representation)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels) # This is the standard CE loss

        if loss is not None:
            return {'loss': loss, 'logits': logits} # Trainer expects loss, but we'll compute it custom
        return {'logits': logits}


# --- Custom Trainer for Distillation ---
class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, alpha=0.5, temperature=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature
        if self.teacher_model:
            self.teacher_model.eval() # Ensure teacher is in eval mode

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # `model` is the student model
        # `inputs` comes from the CombinedDataset, so it has 'imu_pixel_values', 'emg_features', 'labels'
        
        labels = inputs.pop("labels") # Get labels and remove from inputs for student
        emg_features_for_teacher = inputs.get("emg_features") 

        student_outputs = model(**inputs) 
        student_logits = student_outputs['logits']

        # Hard loss (CrossEntropy with true labels)
        loss_ce = F.cross_entropy(student_logits, labels)

        # Soft loss (KL Divergence with teacher's soft targets)
        loss_kd = 0.0
        if self.teacher_model:
            with torch.no_grad():
                # Teacher needs both IMU and EMG
                teacher_inputs = {
                    'imu_pixel_values': inputs['imu_pixel_values'],
                    'emg_features': emg_features_for_teacher
                }
                teacher_outputs = self.teacher_model(**teacher_inputs)
                teacher_logits = teacher_outputs['logits']

            # Soften probabilities
            soft_teacher_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
            log_soft_student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
            
            loss_kd = F.kl_div(log_soft_student_probs, soft_teacher_targets, reduction='batchmean') * (self.temperature ** 2)
        
        # Combined loss
        loss = self.alpha * loss_ce + (1.0 - self.alpha) * loss_kd
        
        if self.args.report_to and "wandb" in self.args.report_to:
             wandb.log({"train/loss_ce": loss_ce.item(), "train/loss_kd": loss_kd.item() if self.teacher_model else 0.0})

        return (loss, student_outputs) if return_outputs else loss


# --- Data Preprocessing Function (Unchanged) ---
def preprocess_participant_data_combined(base_imu_dir, base_emg_feature_dir, participant_id,
                                         window_length_imu, num_sensors_imu, total_features_emg, num_gestures):
    p_imu_dir = os.path.join(base_imu_dir, participant_id)
    p_emg_dir = os.path.join(base_emg_feature_dir, participant_id)
    X_imu_participant, X_emg_participant, y_participant = [], [], []

    for gesture_num in range(1, num_gestures + 1):
        imu_file_path = os.path.join(p_imu_dir, f'Gesture_{gesture_num}.csv')
        emg_file_path = os.path.join(p_emg_dir, f'Gesture_{gesture_num}.csv')

        if not os.path.exists(imu_file_path) or not os.path.exists(emg_file_path): 
            print(f"Warning: File not found for P{participant_id}, G{gesture_num}")
            continue

        try:
            df_imu = pd.read_csv(imu_file_path)
            df_emg = pd.read_csv(emg_file_path)

            if df_imu.shape[1] != num_sensors_imu: continue
            if "Interval" not in df_emg.columns: continue
            if not (total_features_emg <= df_emg.shape[1] -1 <= total_features_emg + 2): continue

            imu_sensor_data = df_imu.values
            df_emg = df_emg.sort_values(by="Interval")
            emg_feature_columns = [col for col in df_emg.columns if col != 'Interval'][:total_features_emg]
            if len(emg_feature_columns) != total_features_emg: continue
            emg_features_for_gesture = df_emg[emg_feature_columns].values.astype(np.float32)

            num_imu_windows = len(imu_sensor_data) // window_length_imu
            num_emg_windows = len(emg_features_for_gesture)
            num_common_windows = min(num_imu_windows, num_emg_windows)
            if num_common_windows == 0: continue

            for i in range(num_common_windows):
                start_idx = i * window_length_imu
                imu_window = imu_sensor_data[start_idx:start_idx + window_length_imu, :]
                imu_window_normalized = normalize_window_data_imu(imu_window.T)
                emg_feature_vector = emg_features_for_gesture[i, :]

                if np.isfinite(emg_feature_vector).all() and np.isfinite(imu_window_normalized).all():
                    X_imu_participant.append(imu_window_normalized)
                    X_emg_participant.append(emg_feature_vector)
                    y_participant.append(gesture_num - 1)
                else:
                    print(f"Warning: NaN/Inf found in data for P{participant_id}, G{gesture_num}, W{i}. Skipping window.")
        except Exception as e:
            print(f"Error processing P{participant_id}, G{gesture_num}: {str(e)}")
            continue
    print(len(X_imu_participant), len(X_emg_participant), len(y_participant))
    if not X_imu_participant: return {'X_imu': np.array([]), 'X_emg': np.array([]), 'y': np.array([])}
    return {
        'X_imu': np.array(X_imu_participant, dtype=np.float32),
        'X_emg': np.array(X_emg_participant, dtype=np.float32),
        'y': np.array(y_participant, dtype=np.int64)
    }


# --- Training Function for Student-Teacher LOSO ---
def train_student_teacher_loso(current_p_id_str, all_participant_data_combined, wandb_project_name):
    print(f"\n--- Training Student Model (Distillation) for test participant: {current_p_id_str} ---")
    monitor_memory(f"Start Student P{current_p_id_str}")

    teacher_emg_encoder_module = ContrastiveFatigueEncoder(
        num_features=TOTAL_FEATURES_EMG, encoder_dim=EMG_ENCODER_DIM, projection_dim=EMG_PROJECTION_DIM
    )
    teacher_model = CombinedGestureModel(
        num_imu_sensors=NUM_SENSORS_IMU, imu_sequence_length=WINDOW_LENGTH_IMU,
        num_gestures=NUM_GESTURES, emg_fatigue_encoder=teacher_emg_encoder_module,
        emg_embedding_dim=EMG_ENCODER_DIM, freeze_emg_encoder=True # EMG encoder is frozen by default
    ).to(DEVICE)

    teacher_checkpoint_path = os.path.join(COMBINED_OUTPUT_DIR, 'checkpoints', f'participant_{current_p_id_str}')
    checkpoints = [d for d in os.listdir(teacher_checkpoint_path) if d.startswith('checkpoint')]
    if not checkpoints:
        print('bruh')
        return 0, {}
    teacher_checkpoint_path = os.path.join(teacher_checkpoint_path, checkpoints[0])
    if not os.path.exists(teacher_checkpoint_path):
        print(f"ERROR: Teacher model checkpoint not found at {teacher_checkpoint_path}. Skipping participant {current_p_id_str}.")
        return 0.0, {}
    try:
        teacher_model.load_state_dict(load_file(os.path.join(teacher_checkpoint_path, "model.safetensors"), device="cpu"))
    except Exception as e:
        print(f"Error loading Teacher's model state_dict for {current_p_id_str}: {e}")
        return 0.0, {}

    teacher_model.eval() # Set teacher to evaluation mode
    for param in teacher_model.parameters(): # Freeze all teacher parameters
        param.requires_grad = False
    print(f"Loaded and froze Teacher model from: {teacher_checkpoint_path}")


    # --- 3. Prepare Train/Test Data Splits (Same as combined model) ---
    train_imu_list, train_emg_list, train_labels_list = [], [], []
    test_imu_list, test_emg_list, test_labels_list = [], [], []
    current_fold_train_emg_features_for_scaling = []

    for p_id, data_dict in all_participant_data_combined.items():
        if not data_dict['X_imu'].size or not data_dict['X_emg'].size: continue
        if p_id == current_p_id_str:
            test_imu_list.append(data_dict['X_imu'])
            test_emg_list.append(data_dict['X_emg'])
            test_labels_list.append(data_dict['y'])
        else:
            train_imu_list.append(data_dict['X_imu'])
            train_emg_list.append(data_dict['X_emg'])
            train_labels_list.append(data_dict['y'])
            if data_dict['X_emg'].size > 0:
                 current_fold_train_emg_features_for_scaling.append(data_dict['X_emg'])
    
    if not train_imu_list or not train_emg_list: return 0.0, {}
    
    if not test_imu_list or not test_emg_list: return 0.0, {}
    
    # Fit EMG Scaler (needed for teacher's EMG input, done by CombinedDataset)
    if current_fold_train_emg_features_for_scaling:
        all_train_emg_np = np.concatenate(current_fold_train_emg_features_for_scaling, axis=0)
        if all_train_emg_np.ndim == 1: all_train_emg_np = all_train_emg_np.reshape(1, -1)
        if all_train_emg_np.shape[0] > 0 and all_train_emg_np.shape[1] == TOTAL_FEATURES_EMG:
             emg_scaler = StandardScaler().fit(all_train_emg_np)
        else: emg_scaler = IdentityScaler()
    else: emg_scaler = IdentityScaler()
    

    full_train_dataset = CombinedDataset(train_imu_list, train_emg_list, train_labels_list, emg_scaler)
    test_dataset = CombinedDataset(test_imu_list, test_emg_list, test_labels_list, emg_scaler)
    
    if len(full_train_dataset) == 0 or len(test_dataset) == 0: return 0.0, {}
    
    train_indices = np.arange(len(full_train_dataset))
    all_train_labels_for_split = full_train_dataset.get_all_labels()
    try:
        val_test_size = 0.15 if len(all_train_labels_for_split) < 100 else 0.2
        unique_labels_train, counts_train = np.unique(all_train_labels_for_split, return_counts=True)
        min_samples_per_class_for_split = 2
        stratify_labels = all_train_labels_for_split if (len(unique_labels_train) >= NUM_GESTURES and all(counts_train >= min_samples_per_class_for_split) and len(all_train_labels_for_split) >= NUM_GESTURES * min_samples_per_class_for_split) else None
        train_idx, val_idx = train_test_split(train_indices, test_size=val_test_size, random_state=42, stratify=stratify_labels)
    except ValueError:
        train_idx, val_idx = train_test_split(train_indices, test_size=0.2, random_state=42)
    print(6)
    train_subset = Subset(full_train_dataset, train_idx)
    val_subset = Subset(full_train_dataset, val_idx)
    if len(val_subset) == 0 and len(train_subset) > 0: val_subset = train_subset
    elif len(train_subset) == 0: return 0.0, {}

    # --- W&B and STUDENT Model Training ---
    run = wandb.init(
        project=wandb_project_name,
        name=f"run_test_P{current_p_id_str}_student_distilled",
        config={
            "learning_rate": 1e-4, "epochs": 75, "batch_size": 32, # Student might need more epochs or adjusted LR
            "model_type": "Student_IMU_Distilled", "test_participant": current_p_id_str,
            "kd_alpha": KD_ALPHA, "kd_temperature": KD_TEMPERATURE,
            "window_length_imu": WINDOW_LENGTH_IMU, "num_sensors_imu": NUM_SENSORS_IMU,
            "num_gestures": NUM_GESTURES
        },
        reinit=True
    )

    student_model = StudentIMUModel(
        num_imu_sensors=NUM_SENSORS_IMU,
        imu_sequence_length=WINDOW_LENGTH_IMU,
        num_gestures=NUM_GESTURES
    ).to(DEVICE)

    training_args_student = TrainingArguments(
        output_dir=os.path.join(STUDENT_OUTPUT_DIR, 'checkpoints', f'participant_{current_p_id_str}'),
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
        greater_is_better=True if len(val_subset) > 0 and "accuracy" in (training_args_student.metric_for_best_model if 'training_args_student' in locals() else "accuracy") else False,
        save_total_limit=1,
        logging_dir=os.path.join(STUDENT_OUTPUT_DIR, 'logs', f'participant_{current_p_id_str}'),
        logging_steps=10,
        remove_unused_columns=False 
    )

    def compute_metrics_student(p):
        preds = np.argmax(p.predictions, axis=1)
        return {"accuracy": accuracy_score(p.label_ids, preds)}

    distillation_trainer = DistillationTrainer(
        model=student_model,
        teacher_model=teacher_model, 
        args=training_args_student,
        train_dataset=train_subset,
        eval_dataset=val_subset if len(val_subset) > 0 else None,
        compute_metrics=compute_metrics_student if len(val_subset) > 0 else None,
        alpha=run.config.kd_alpha,
        temperature=run.config.kd_temperature
    )

    print("Starting student model training with distillation...")
    distillation_trainer.train()
    
    print("Evaluating student model on test set...")
    best_student_model_path = os.path.join(training_args_student.output_dir, "pytorch_model.bin") 
    if distillation_trainer.state.best_model_checkpoint:
         best_student_model_path = os.path.join(distillation_trainer.state.best_model_checkpoint, "pytorch_model.bin")
    
    if Path(best_student_model_path).exists():
        student_model.load_state_dict(torch.load(best_student_model_path, map_location=DEVICE))
        print(f"Loaded best student model from {best_student_model_path} for final evaluation.")
    else:
        print(f"Warning: Best student model checkpoint not found at {best_student_model_path}. Using last state.")

    student_model.eval()
    
    predictions_output_student = distillation_trainer.predict(test_dataset) # Uses the model currently in trainer
    y_pred_test_student = np.argmax(predictions_output_student.predictions, axis=1)
    y_true_test_student = test_dataset.get_all_labels()

    accuracy_test_student = accuracy_score(y_true_test_student, y_pred_test_student)
    report_test_dict_student = classification_report(y_true_test_student, y_pred_test_student, output_dict=True, zero_division=0, labels=np.arange(NUM_GESTURES))

    print(f"Test Accuracy for P{current_p_id_str} (Student Model): {accuracy_test_student:.4f}")
    print(classification_report(y_true_test_student, y_pred_test_student, zero_division=0, labels=np.arange(NUM_GESTURES)))

    wandb.log({ "test_accuracy_student": accuracy_test_student, "test_classification_report_student": report_test_dict_student })
    run.finish()

    del student_model, teacher_model, teacher_emg_encoder_module, distillation_trainer, emg_scaler
    del full_train_dataset, test_dataset, train_subset, val_subset
    del train_imu_list, train_emg_list, train_labels_list
    del test_imu_list, test_emg_list, test_labels_list
    torch.cuda.empty_cache()
    gc.collect()
    monitor_memory(f"End Student P{current_p_id_str}")

    return accuracy_test_student, report_test_dict_student


# --- Main Execution for Student-Teacher ---
def main_student_teacher():
    Path(STUDENT_OUTPUT_DIR, "accuracy_plots").mkdir(parents=True, exist_ok=True)
    Path(STUDENT_OUTPUT_DIR, "classification_reports").mkdir(parents=True, exist_ok=True)
    Path(STUDENT_OUTPUT_DIR, "checkpoints").mkdir(parents=True, exist_ok=True)
    Path(STUDENT_OUTPUT_DIR, "logs").mkdir(parents=True, exist_ok=True)

    participant_dirs = sorted([
        d for d in os.listdir(BASE_IMU_DIR)
        if os.path.isdir(os.path.join(BASE_IMU_DIR, d)) and d.startswith('P')
    ])
    if not participant_dirs:
        print(f"Error: No participant directories found in {BASE_IMU_DIR}")
        return
    print(f"Found participants for student-teacher: {participant_dirs}")

    print("\nPreprocessing all participant data (IMU & EMG) for student-teacher...")
    all_participant_data_combined = {} # Reusing the same data structure
    initial_mem = monitor_memory("Pre-load ST")
    for p_id_str in tqdm(participant_dirs, desc="Loading data for ST"):
        all_participant_data_combined[p_id_str] = preprocess_participant_data_combined(
            BASE_IMU_DIR, BASE_EMG_FEATURE_DIR, p_id_str,
            WINDOW_LENGTH_IMU, NUM_SENSORS_IMU, TOTAL_FEATURES_EMG, NUM_GESTURES
        )
        time.sleep(0.01)
    loaded_mem = monitor_memory("Post-load ST")
    print(f"Memory increase after loading for ST: {loaded_mem - initial_mem:.2f} MB")

    results_summary_student = []
    for current_p_id_test_str in participant_dirs:

        accuracy, report = train_student_teacher_loso(
            current_p_id_test_str, all_participant_data_combined, WANDB_PROJECT_NAME_STUDENT
        )

        if report and 'macro avg' in report:
            results_summary_student.append({
                'Participant_Test': current_p_id_test_str, 'Accuracy': accuracy,
                'Macro_Avg_Precision': report['macro avg']['precision'],
                'Macro_Avg_Recall': report['macro avg']['recall'],
                'Macro_Avg_F1': report['macro avg']['f1-score'],
            })
            pd.DataFrame(report).transpose().to_csv(
                os.path.join(STUDENT_OUTPUT_DIR, "classification_reports", f'report_student_{current_p_id_test_str}.csv')
            )
        else:
            print(f"Skipping results summary for student model, participant {current_p_id_test_str} due to empty/invalid report.")
        
        gc.collect()
        torch.cuda.empty_cache()

    if results_summary_student:
        results_df_student = pd.DataFrame(results_summary_student)
        mean_values_student = results_df_student.drop(columns=['Participant_Test'], errors='ignore').mean(numeric_only=True)
        mean_row_dict_student = mean_values_student.to_dict()
        mean_row_dict_student['Participant_Test'] = 'Mean_LOSO_Student'
        results_df_student = pd.concat([results_df_student, pd.DataFrame([mean_row_dict_student])], ignore_index=True)
        
        results_df_student.to_csv(os.path.join(STUDENT_OUTPUT_DIR, 'loso_student_distilled_summary_accuracies.csv'), index=False)
        print("\n--- LOSO Student Model (Distilled) Summary ---")
        print(results_df_student)

        avg_accuracy_overall_student = results_df_student[results_df_student['Participant_Test'] == 'Mean_LOSO_Student']['Accuracy'].iloc[0]
        plt.figure(figsize=(max(8, len(participant_dirs)*0.5), 5))
        participant_accuracies_student = results_df_student[results_df_student['Participant_Test'] != 'Mean_LOSO_Student']['Accuracy']
        participant_names_student = results_df_student[results_df_student['Participant_Test'] != 'Mean_LOSO_Student']['Participant_Test']
        plt.bar(participant_names_student, participant_accuracies_student, color='deepskyblue')
        plt.axhline(avg_accuracy_overall_student, color='darkorange', linestyle='--', label=f'Mean Accuracy: {avg_accuracy_overall_student:.2f}')
        plt.xlabel('Test Participant'); plt.ylabel('Accuracy'); plt.title('LOSO Student (Distilled) Model Accuracy')
        plt.xticks(rotation=45, ha="right"); plt.ylim(0, 1); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(STUDENT_OUTPUT_DIR, 'loso_student_distilled_overall_accuracy.png'))
        plt.close()
        print(f"\nOverall Mean LOSO Accuracy (Student Distilled Model): {avg_accuracy_overall_student:.4f}")
    else:
        print("No results were generated for the student (distilled) model.")

if __name__ == "__main__":

    print("\n\n--- Running Main Student-Teacher Distillation Training ---")
    main_student_teacher()