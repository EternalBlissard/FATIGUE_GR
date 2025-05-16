import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# TODO: RANDOM SEED
RANDOM_STATE = 41
np.random.seed(RANDOM_STATE)

# TODO: PATHS
FEATURES_BASE_DIR: str = "path/to/features/directory"
MODEL_OUTPUT_DIR: str = f"path/to/output/directory_random{RANDOM_STATE}"
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

TARGET_COLUMN = 'gesture' 
PARTICIPANT_ID_COLUMN = 'participant' 

def load_all_feature_files(base_dir):
    """Loads all feature CSV files from subdirectories into a single DataFrame."""
    all_feature_files = []
    for p_folder in glob.glob(os.path.join(base_dir, 'P*')): 
        participant_files = glob.glob(os.path.join(p_folder, 'Gesture_*_features.csv'))
        all_feature_files.extend(participant_files)

    if not all_feature_files:
        print(f"No feature files found in {base_dir}. Exiting.")
        return None
    print(f"Found {len(all_feature_files)} feature files.")

    df_list = []
    for f_path in all_feature_files:
        try:
            df = pd.read_csv(f_path)
            if PARTICIPANT_ID_COLUMN not in df.columns:
                participant_id_from_path = os.path.basename(os.path.dirname(f_path))
                df[PARTICIPANT_ID_COLUMN] = participant_id_from_path
            df_list.append(df)
        except pd.errors.EmptyDataError:
            print(f"Warning: Skipping empty feature file: {f_path}")
        except Exception as e:
            print(f"Warning: Error loading {f_path}: {e}")

    if not df_list:
        print("No data loaded after attempting to read files. Exiting.")
        return None
    full_df = pd.concat(df_list, ignore_index=True)
    print(f"Combined DataFrame shape: {full_df.shape}")
    return full_df

def get_feature_names_and_cols_to_drop(df_columns, target_col, participant_id_col):
    cols_to_drop = [participant_id_col, 'window_id', 'window_start_time', 'window_end_time', target_col, 'original_file']
    
    if 'gesture' in df_columns and 'gesture' != target_col:
        cols_to_drop.append('gesture')
    
    actual_cols_to_drop = [col for col in cols_to_drop if col in df_columns]
    feature_names = [col for col in df_columns if col not in actual_cols_to_drop]
    feature_names = [fn for fn in feature_names if fn not in [target_col, participant_id_col]]
    
    return feature_names, list(set(actual_cols_to_drop))


def train_random_forest_for_loso(X_train, y_train):
    """Trains a Random Forest classifier for a LOSO fold."""
    model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight='balanced',
                                   max_depth=20, min_samples_leaf=2, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def evaluate_loso_results(all_y_true, all_y_pred, label_encoder, feature_names, all_feature_importances, output_dir):
    """Evaluates aggregated LOSO results and saves them, including gesture-wise accuracy."""
    print(f"\n--- Aggregated LOSO Evaluation ---")

    
    all_y_true_np = np.array(all_y_true)
    all_y_pred_np = np.array(all_y_pred)

    # Overall Accuracy
    accuracy = accuracy_score(all_y_true_np, all_y_pred_np)
    print(f"Overall LOSO Accuracy (Aggregated): {accuracy:.4f}")
    with open(os.path.join(output_dir, "loso_overall_accuracy_AGGREGATED.txt"), "w") as f:
        f.write(f"Overall LOSO Accuracy (Aggregated): {accuracy:.4f}\n")
        
    # Label Encoding Classes
    report_labels_encoded = label_encoder.transform(label_encoder.classes_)
    report_target_names_str = label_encoder.classes_.astype(str)

    # Generate the classification report as a dictionary to easily access metrics
    report_dict = classification_report(
        all_y_true_np, 
        all_y_pred_np,
        labels=report_labels_encoded,      
        target_names=report_target_names_str, 
        zero_division=0,
        output_dict=True
    )

    # Gesture-wise Accuracies (Recalls) (Aggregated)
    gesture_wise_accuracies = {}
    print("\nGesture-wise Accuracies (Recalls) (Aggregated):")
    for gesture_name in report_target_names_str: 
        if gesture_name in report_dict: 
            recall = report_dict[gesture_name]['recall']
            gesture_wise_accuracies[gesture_name] = recall
            print(f"  Accuracy for {gesture_name}: {recall:.4f} (Support: {report_dict[gesture_name]['support']})")
        else:
            gesture_wise_accuracies[gesture_name] = np.nan 
            print(f"  Accuracy for {gesture_name}: N/A (Not found in report dictionary)")


    avg_gesture_wise_accuracy = report_dict['macro avg']['recall']
    print(f"Average Gesture-wise Accuracy (Macro Recall): {avg_gesture_wise_accuracy:.4f}")

    path_gesture_wise_acc = os.path.join(output_dir, "loso_gesture_wise_accuracy_AGGREGATED.txt")
    with open(path_gesture_wise_acc, "w") as f:
        f.write("Gesture-wise Accuracies (Recalls) (Aggregated):\n")
        for gesture_name, acc in gesture_wise_accuracies.items():
            support = report_dict[gesture_name]['support'] if gesture_name in report_dict else 'N/A'
            f.write(f"  Accuracy for {gesture_name}: {acc:.4f} (Support: {support})\n")
        f.write(f"\nAverage Gesture-wise Accuracy (Macro Recall): {avg_gesture_wise_accuracy:.4f}\n")
    print(f"Gesture-wise accuracies saved to: {path_gesture_wise_acc}")


    # Complete Full Classification Report in String Format
    full_report_str = classification_report(
        all_y_true_np, 
        all_y_pred_np,
        labels=report_labels_encoded,
        target_names=report_target_names_str,
        zero_division=0
    )
    print("\nOverall LOSO Classification Report (Aggregated):")
    print(full_report_str)
    with open(os.path.join(output_dir, "loso_classification_report_AGGREGATED.txt"), "w") as f:
        f.write(f"Overall LOSO Accuracy (Aggregated): {accuracy:.4f}\n\n") 
        f.write(full_report_str)
    print(f"Full classification report saved to: {os.path.join(output_dir, 'loso_classification_report_AGGREGATED.txt')}")

    # Confusion Matrix
    cm = confusion_matrix(all_y_true_np, all_y_pred_np, labels=report_labels_encoded)
    plt.figure(figsize=(max(8, len(label_encoder.classes_)*0.9), max(6, len(label_encoder.classes_)*0.7)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=report_target_names_str, 
                yticklabels=report_target_names_str) 
    plt.title('Overall LOSO Confusion Matrix (Aggregated)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loso_confusion_matrix_AGGREGATED.png"))
    plt.close()
    print("Overall LOSO Confusion matrix (Aggregated) saved.")

    # Averaged Feature Importances
    if all_feature_importances: 
        if len(set(len(fi) for fi in all_feature_importances)) > 1:
            print("Warning: Feature importance arrays have inconsistent lengths. Cannot average.")
        elif len(all_feature_importances[0]) != len(feature_names):
             print(f"Warning: Length of feature importances ({len(all_feature_importances[0])}) does not match number of feature names ({len(feature_names)}). Cannot plot feature importances accurately.")
        else:
            avg_importances = np.mean(all_feature_importances, axis=0)
            indices = np.argsort(avg_importances)[::-1]
            top_n = min(30, len(feature_names))

            plt.figure(figsize=(12, max(8, top_n * 0.4))) 
            plt.title(f"Top {top_n} Averaged Feature Importances (LOSO - Aggregated)")
            sns.barplot(x=[feature_names[i] for i in indices[:top_n]], y=avg_importances[indices][:top_n], palette="viridis")
            plt.xticks(rotation=90)
            plt.ylabel("Mean Gini Importance")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "loso_avg_feature_importances_AGGREGATED.png"))
            plt.close()
            print("Averaged LOSO feature importances plot (Aggregated) saved.")

            importance_df = pd.DataFrame({'feature': feature_names, 'importance': avg_importances})
            importance_df = importance_df.sort_values(by='importance', ascending=False).reset_index(drop=True)
            importance_df.to_csv(os.path.join(output_dir, "loso_avg_feature_importances_AGGREGATED.csv"), index=False)
            print("Averaged LOSO feature importances (Aggregated) saved to CSV.")
    else:
        print("No feature importances collected to average for aggregated report.")

def save_per_participant_metrics(per_participant_metrics_list, gesture_class_names, output_dir):
    """Saves per-participant metrics to text and CSV files."""
    if not per_participant_metrics_list:
        print("No per-participant metrics to save.")
        return
        
    print(f"\n--- Saving Per-Participant Metrics ---")
    
    path_txt = os.path.join(output_dir, "loso_per_participant_classification_reports.txt")
    with open(path_txt, "w") as f:
        for item in per_participant_metrics_list:
            f.write(f"--- Participant: {item['participant_id']} ---\n")
            f.write(f"Overall Accuracy for this participant: {item['accuracy']:.4f}\n")
            f.write("Classification Report:\n")
            f.write(item['classification_report_str']) 
            f.write("\n\n")
    print(f"Per-participant classification reports saved to: {path_txt}")

    summary_data = []
    sorted_gesture_names = sorted(list(gesture_class_names)) 

    for item in per_participant_metrics_list:
        row = {'participant_id': item['participant_id'], 'overall_accuracy': item['accuracy']}
        report_dict = item['classification_report_dict']
        
        for gesture_name in sorted_gesture_names:
            if gesture_name in report_dict and isinstance(report_dict[gesture_name], dict):
                class_metrics = report_dict[gesture_name]
                row[f'{gesture_name}_precision'] = class_metrics.get('precision', np.nan)
                row[f'{gesture_name}_recall'] = class_metrics.get('recall', np.nan)
                row[f'{gesture_name}_f1-score'] = class_metrics.get('f1-score', np.nan)
                row[f'{gesture_name}_support'] = class_metrics.get('support', 0)
            else:
                row[f'{gesture_name}_precision'] = np.nan
                row[f'{gesture_name}_recall'] = np.nan
                row[f'{gesture_name}_f1-score'] = np.nan
                row[f'{gesture_name}_support'] = 0
        
        # Add macro and weighted averages with underscore formatting for column names
        for avg_type_key in ['macro avg', 'weighted avg']: 
            col_prefix = avg_type_key.replace(" ", "_") 
            
            if avg_type_key in report_dict and isinstance(report_dict[avg_type_key], dict):
                avg_metrics_dict = report_dict[avg_type_key]
                row[f'{col_prefix}_precision'] = avg_metrics_dict.get('precision', np.nan)
                row[f'{col_prefix}_recall'] = avg_metrics_dict.get('recall', np.nan)
                row[f'{col_prefix}_f1-score'] = avg_metrics_dict.get('f1-score', np.nan)
                if avg_type_key == 'weighted avg': 
                    row[f'{col_prefix}_support'] = avg_metrics_dict.get('support', np.nan)
            else:
                row[f'{col_prefix}_precision'] = np.nan
                row[f'{col_prefix}_recall'] = np.nan
                row[f'{col_prefix}_f1-score'] = np.nan
                if avg_type_key == 'weighted avg':
                    row[f'{col_prefix}_support'] = np.nan
        summary_data.append(row)

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        metric_cols_per_gesture = []
        for gesture_name in sorted_gesture_names:
            metric_cols_per_gesture.extend([f'{gesture_name}_precision', f'{gesture_name}_recall', f'{gesture_name}_f1-score', f'{gesture_name}_support'])
        
        avg_metric_cols = []
        for avg_name_prefix in ['macro_avg', 'weighted_avg']: 
            avg_metric_cols.extend([f'{avg_name_prefix}_precision', f'{avg_name_prefix}_recall', f'{avg_name_prefix}_f1-score'])
            if avg_name_prefix == 'weighted_avg': 
                 avg_metric_cols.append(f'{avg_name_prefix}_support')

        ordered_cols = ['participant_id', 'overall_accuracy'] + metric_cols_per_gesture + avg_metric_cols
        
        final_cols = [col for col in ordered_cols if col in summary_df.columns]
        final_cols.extend([col for col in summary_df.columns if col not in final_cols])
        
        summary_df = summary_df[final_cols]
        
        path_csv = os.path.join(output_dir, "loso_per_participant_metrics_summary.csv")
        summary_df.to_csv(path_csv, index=False, float_format='%.4f')
        print(f"Per-participant metrics summary saved to: {path_csv}")
    else:
        print("No summary data was generated for per-participant metrics CSV.")


def main_gesture_recognition_loso():
    print("--- Starting LOSO Gesture Recognition Pipeline ---")

    full_df = load_all_feature_files(FEATURES_BASE_DIR)
    if full_df is None or full_df.empty:
        print("Failed to load data. Exiting.")
        return

    print("\nSample of loaded data (first 3 rows):")
    print(full_df.head(3))
    print(f"\nColumns in loaded DataFrame: {full_df.columns.tolist()}")

    if TARGET_COLUMN not in full_df.columns:
        print(f"Error: Target column '{TARGET_COLUMN}' not found in DataFrame.")
        return
    
    print(f"\nValue counts for target column '{TARGET_COLUMN}' (before NaN handling):")
    print(full_df[TARGET_COLUMN].value_counts(dropna=False))
            
    if PARTICIPANT_ID_COLUMN not in full_df.columns:
        print(f"Error: Participant ID column '{PARTICIPANT_ID_COLUMN}' not found. Cannot perform LOSO.")
        return

    initial_rows = len(full_df)
    full_df.dropna(subset=[PARTICIPANT_ID_COLUMN], inplace=True)
    if len(full_df) < initial_rows:
        print(f"Dropped {initial_rows - len(full_df)} rows due to missing participant ID.")
    
    if full_df.empty:
        print("DataFrame is empty after dropping rows with missing participant ID. Exiting.")
        return

    participants = full_df[PARTICIPANT_ID_COLUMN].astype(str).unique()
    if len(participants) <= 1:
        print(f"Error: Need at least 2 participants for LOSO. Found: {len(participants)} ({participants}). Exiting.")
        return
    
    print(f"\nFound {len(participants)} unique participants for LOSO: {sorted(participants)}")

    le = LabelEncoder()
    try:
        y_all_for_le_fit = full_df[TARGET_COLUMN].dropna().astype(str)
        if y_all_for_le_fit.empty:
            print(f"Error: Target column '{TARGET_COLUMN}' contains only NaNs or is empty. Cannot fit LabelEncoder.")
            return
        le.fit(y_all_for_le_fit)
        print(f"Label encoder classes ({len(le.classes_)}): {le.classes_}")
    except Exception as e:
        print(f"Error fitting LabelEncoder on '{TARGET_COLUMN}': {e}")
        return

    all_y_true_encoded_aggregated = [] 
    all_y_pred_fold_aggregated = []    
    all_fold_feature_importances = []
    per_participant_metrics_list = []
    
    feature_names, cols_to_drop_for_X = get_feature_names_and_cols_to_drop(
        full_df.columns, TARGET_COLUMN, PARTICIPANT_ID_COLUMN
    )
    if not feature_names:
        print("Error: No feature names identified. Check column dropping logic and data content.")
        return
    print(f"Identified {len(feature_names)} features.")

    logo = LeaveOneGroupOut()
    groups_for_loso = full_df[PARTICIPANT_ID_COLUMN].astype(str) 
    fold_num = 0

    for train_idx, test_idx in logo.split(full_df, groups=groups_for_loso):
        fold_num += 1
        df_train, df_test = full_df.iloc[train_idx].copy(), full_df.iloc[test_idx].copy()
        
        current_test_participant_arr = df_test[PARTICIPANT_ID_COLUMN].astype(str).unique()
        if not current_test_participant_arr.size:
             print(f"Warning: Fold {fold_num} has test data but no participant ID. Skipping.")
             continue
        current_test_participant = current_test_participant_arr[0]

        print(f"\n--- Fold {fold_num}/{len(participants)}: Testing on Participant {current_test_participant} ---")
        print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")

        y_train_raw = df_train[TARGET_COLUMN]
        X_train = df_train[feature_names]
        y_test_raw = df_test[TARGET_COLUMN]
        X_test = df_test[feature_names]

        train_valid_target_idx = y_train_raw.notna()
        X_train = X_train[train_valid_target_idx]
        y_train_raw = y_train_raw[train_valid_target_idx].astype(str)

        test_valid_target_idx = y_test_raw.notna()
        X_test = X_test[test_valid_target_idx]
        y_test_raw = y_test_raw[test_valid_target_idx].astype(str)

        empty_fold_metrics_entry = {
            'participant_id': current_test_participant,
            'accuracy': np.nan,
            'classification_report_dict': {cls: {'precision':0.0,'recall':0.0,'f1-score':0.0,'support':0} for cls in le.classes_},
            'classification_report_str': "No valid test data for this participant in this fold (e.g., all targets were NaN or removed due to not being in LabelEncoder's known classes)."
        }
        empty_fold_metrics_entry['classification_report_dict']['macro avg'] = {'precision':np.nan,'recall':np.nan,'f1-score':np.nan,'support':0}
        empty_fold_metrics_entry['classification_report_dict']['weighted avg'] = {'precision':np.nan,'recall':np.nan,'f1-score':np.nan,'support':0}


        if X_train.empty or y_train_raw.empty:
            print(f"Fold {fold_num}: Training data empty for participant {current_test_participant}. Skipping.")
            per_participant_metrics_list.append(empty_fold_metrics_entry)
            continue
        if X_test.empty or y_test_raw.empty:
            print(f"Fold {fold_num}: Test data empty for participant {current_test_participant}. No predictions.")
            per_participant_metrics_list.append(empty_fold_metrics_entry)
            continue
            
        train_mask = y_train_raw.isin(le.classes_)
        if not train_mask.all():
            print(f"Warning: Fold {fold_num} (Train): {sum(~train_mask)} samples with unseen labels by LabelEncoder. Removing them.")
            y_train_raw = y_train_raw[train_mask]
            X_train = X_train[train_mask]
        
        test_mask = y_test_raw.isin(le.classes_)
        if not test_mask.all():
            print(f"Warning: Fold {fold_num} (Test): {sum(~test_mask)} samples with unseen labels by LabelEncoder. Removing them.")
            y_test_raw = y_test_raw[test_mask]
            X_test = X_test[test_mask]

        if y_train_raw.empty or X_train.empty:
            print(f"Fold {fold_num}: Training data empty after removing unseen labels. Skipping.")
            per_participant_metrics_list.append(empty_fold_metrics_entry)
            continue
        if y_test_raw.empty or X_test.empty:
            print(f"Fold {fold_num}: Test data empty after removing unseen labels. No predictions.")
            per_participant_metrics_list.append(empty_fold_metrics_entry)
            continue

        y_train_encoded = le.transform(y_train_raw)
        y_test_encoded = le.transform(y_test_raw)

        imputer = SimpleImputer(strategy='mean')
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        
        X_train_imputed_df = pd.DataFrame(X_train_imputed, columns=feature_names, index=X_train.index)
        X_test_imputed_df = pd.DataFrame(X_test_imputed, columns=feature_names, index=X_test.index)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed_df)
        X_test_scaled = scaler.transform(X_test_imputed_df)
        
        model = train_random_forest_for_loso(X_train_scaled, y_train_encoded)
        y_pred_this_fold = model.predict(X_test_scaled)
        
        all_y_true_encoded_aggregated.extend(y_test_encoded)
        all_y_pred_fold_aggregated.extend(y_pred_this_fold)
        
        if hasattr(model, 'feature_importances_'):
            if len(model.feature_importances_) == len(feature_names):
                all_fold_feature_importances.append(model.feature_importances_)
            else:
                print(f"Warning: Mismatch FI length ({len(model.feature_importances_)}) vs features ({len(feature_names)}) for fold {fold_num}.")
        
        acc_fold = accuracy_score(y_test_encoded, y_pred_this_fold)
        report_labels_fold = le.transform(le.classes_) 
        report_target_names_fold = le.classes_.astype(str)

        fold_report_dict = classification_report(
            y_test_encoded, y_pred_this_fold,
            labels=report_labels_fold,
            target_names=report_target_names_fold,
            output_dict=True, zero_division=0
        )
        fold_report_str = classification_report(
            y_test_encoded, y_pred_this_fold,
            labels=report_labels_fold,
            target_names=report_target_names_fold,
            zero_division=0 
        )
        print(f"Fold {fold_num} (Participant {current_test_participant}) Accuracy: {acc_fold:.4f}. Full report metrics stored.")

        per_participant_metrics_list.append({
            'participant_id': current_test_participant,
            'accuracy': acc_fold,
            'classification_report_dict': fold_report_dict,
            'classification_report_str': fold_report_str
        })

    
    if not all_y_true_encoded_aggregated:
        print("\nNo valid predictions were made across all LOSO folds. Cannot produce aggregated evaluation.")
    else:
        evaluate_loso_results(all_y_true_encoded_aggregated, all_y_pred_fold_aggregated, le, 
                              feature_names, all_fold_feature_importances, MODEL_OUTPUT_DIR)

    save_per_participant_metrics(per_participant_metrics_list, le.classes_.astype(str), MODEL_OUTPUT_DIR)

    print("\n--- LOSO Gesture Recognition Pipeline Finished ---")

if __name__ == "__main__":
    main_gesture_recognition_loso()