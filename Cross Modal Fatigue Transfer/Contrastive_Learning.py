from __future__ import annotations

import concurrent.futures as cf
import gc
import glob
import os
import re
import time
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import wandb

# Constants
NUM_SENSORS = 8
FEATURES_PER_SENSOR = 11
TOTAL_FEATURES = NUM_SENSORS * FEATURES_PER_SENSOR

#TODO: Random Seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

#TODO: Paths
SUBJECTIVE_CSV_PATH = "/path/to/subjective/csv"
FEATURE_DATA_BASE_DIR = "/path/to/feature/data"
OUTPUT_DIR = f"/path/to/output/directory_seed_{RANDOM_SEED}"

#TODO: WandB
WANDB_ENTITY = "EMG-Gesture-Recognition"
WANDB_PROJECT = "EMG_LOSO_Contrastive_SUB-fixed"

#TODO: Training Params
CONTRASTIVE_LR = 1e-4
CONTRASTIVE_WEIGHT_DECAY = 1e-5
CONTRASTIVE_EPOCHS = 10
CONTRASTIVE_BATCH_SIZE = 256
TRIPLET_MARGIN = 0.5
EVAL_BATCH_SIZE = 1024


CFG = {
    "subjective_csv_path": SUBJECTIVE_CSV_PATH,  
    "feature_data_base_dir": FEATURE_DATA_BASE_DIR,
    "output_dir": OUTPUT_DIR,
    "wandb_entity": WANDB_ENTITY,
    "wandb_project": WANDB_PROJECT,
    "contrastive_lr": CONTRASTIVE_LR,
    "contrastive_weight_decay": CONTRASTIVE_WEIGHT_DECAY,
    "contrastive_epochs": CONTRASTIVE_EPOCHS,
    "contrastive_batch_size": CONTRASTIVE_BATCH_SIZE,
    "triplet_margin": TRIPLET_MARGIN,
    "encoder_dim": 128,
    "projection_dim": 64,
    "hard_frac": 0.2,           
    "patience": 15,             
    "val_split": 0.15,
    "eval_batch_size": EVAL_BATCH_SIZE,
    "num_workers": 4,
}

Path(CFG["output_dir"], "accuracy_plots").mkdir(parents=True, exist_ok=True)


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using CUDA ({torch.cuda.device_count()} device(s))")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU – expect slow training …")

PIN_MEMORY = torch.cuda.is_available()


def triplet_sq_loss(anchor: torch.Tensor,
                    positive: torch.Tensor,
                    negative: torch.Tensor,
                    margin: float = CFG["triplet_margin"]) -> torch.Tensor:
    d_ap = (anchor - positive).pow(2).sum(dim=1)        
    d_an = (anchor - negative).pow(2).sum(dim=1)       
    return F.relu(d_ap - d_an + margin).mean()


def monitor_memory(tag: str = "") -> float:
    mb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    print(f"[MEM] {tag:<25} {mb:8.1f} MB")
    return mb

# ---------------------------------------------------------------- dataset

class FatigueFeatureDataset(Dataset):

    def __init__(self,
                 features: np.ndarray,
                 labels: np.ndarray,
                 pids: np.ndarray,
                 scaler: StandardScaler | None = None):

        if scaler is not None:
            features = scaler.transform(features)
        self._X = features.astype(np.float32, copy=False)
        self._y = labels.astype(np.int64, copy=False)
        self._pid = pids.astype(np.int32, copy=False)

    def __len__(self) -> int:
        return self._X.shape[0]

    def __getitem__(self, idx: int):
        return {
            "features": torch.from_numpy(self._X[idx]),
            "labels":   torch.tensor(self._y[idx], dtype=torch.long),
            "participant_id": torch.tensor(self._pid[idx], dtype=torch.long),
        }


class ContrastiveFatigueEncoder(nn.Module):
    def __init__(self,
                 num_features: int = TOTAL_FEATURES,
                 encoder_dim: int = 128,
                 projection_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, encoder_dim),
            nn.BatchNorm1d(encoder_dim),
        )
        self.proj = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim // 2),
            nn.ReLU(),
            nn.Linear(encoder_dim // 2, projection_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.encoder(x)
        proj = self.proj(enc)
        return enc, proj


def mine_triplets_batch(emb: torch.Tensor,
                        y: torch.Tensor,
                        p: torch.Tensor,
                        hard_frac: float = 0.2) -> Tuple[torch.Tensor, ...]:
    """Adaptive semi‑hard miner (see doc‑string at top)."""
    bs = emb.size(0)
    if bs < 3:
        return None, None, None

    if 0.9 < emb.norm(dim=1).mean().item() < 1.1: 
        dist = 1 - torch.matmul(F.normalize(emb, dim=1), F.normalize(emb, dim=1).T)
    else:
        dist = torch.cdist(emb, emb, p=2)

    dist.fill_diagonal_(float("inf"))
    device = emb.device
    q_pos = max(1, int(hard_frac * (bs - 1)))
    q_neg = q_pos

    a_idx, p_idx, n_idx = [], [], []

    for i in range(bs):
        same_p, same_y = p == p[i], y == y[i]
        pos_mask = same_p & same_y
        pos_mask[i] = False
        if not pos_mask.any():
            continue
        pos_d = dist[i][pos_mask]
        kth = torch.kthvalue(pos_d, min(q_pos, pos_d.numel()))[0]
        hard_pos_mask = pos_mask & (dist[i] >= kth - 1e-6)
        j = torch.multinomial(hard_pos_mask.float(), 1).item()

        pos_d_val = dist[i, j]
        neg_mask = (~same_p) | (~same_y)
        cand = neg_mask & (dist[i] > pos_d_val)
        if not cand.any():
            continue
        neg_d = dist[i][cand]
        kth_neg = torch.kthvalue(neg_d, min(q_neg, neg_d.numel()))[0]
        hard_neg_mask = cand & (dist[i] <= kth_neg + 1e-6)
        k = torch.multinomial(hard_neg_mask.float(), 1).item()

        a_idx.append(i); p_idx.append(j); n_idx.append(k)

    if not a_idx:
        return None, None, None
    return (torch.tensor(a_idx, device=device),
            torch.tensor(p_idx, device=device),
            torch.tensor(n_idx, device=device))

# Training Loop

def train_contrastive(model: nn.Module,
                      tr_loader: DataLoader,
                      va_loader: DataLoader | None,
                      fold_tag: str) -> nn.Module:
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["contrastive_lr"],
                            weight_decay=CFG["contrastive_weight_decay"])

    best_val = float("inf")
    patience = CFG["patience"]
    wait = 0
    for ep in range(1, CFG["contrastive_epochs"] + 1):
        model.train()
        ep_loss = ep_triplets = 0
        for batch in tqdm(tr_loader, leave=False, desc=f"{fold_tag} ▸ train e{ep}"):
            feat = batch["features"].to(DEVICE)
            lab = batch["labels"].to(DEVICE)
            pid = batch["participant_id"].to(DEVICE)

            _, proj = model(feat)
            a, p, n = mine_triplets_batch(proj, lab, pid, hard_frac=CFG["hard_frac"])
            if a is None:
                continue
            loss = triplet_sq_loss(proj[a], proj[p], proj[n])
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * a.size(0)
            ep_triplets += a.size(0)
        ep_loss = ep_loss / max(1, ep_triplets)
        print(f"{fold_tag} e{ep:02d} train‑loss {ep_loss:.4f} (triplets {ep_triplets})")

        # validation
        if va_loader is not None:
            model.eval(); val_loss = val_trip = 0
            with torch.no_grad():
                for batch in va_loader:
                    feat = batch["features"].to(DEVICE)
                    lab = batch["labels"].to(DEVICE)
                    pid = batch["participant_id"].to(DEVICE)
                    _, proj = model(feat)
                    a, p, n = mine_triplets_batch(proj, lab, pid, hard_frac=CFG["hard_frac"])
                    if a is None:
                        continue
                    loss = triplet_sq_loss(proj[a], proj[p], proj[n])
                    val_loss += loss.item() * a.size(0)
                    val_trip += a.size(0)
            val_loss = val_loss / max(1, val_trip)
            print(f"{fold_tag} e{ep:02d}   val‑loss {val_loss:.4f} (triplets {val_trip})")
            improve = val_loss < best_val
            best_val = min(best_val, val_loss)
        else:
            improve = (ep == CFG["contrastive_epochs"])  

        if improve:
            wait = 0
            enc_state = (model.module.encoder if isinstance(model, nn.DataParallel)
                          else model.encoder).state_dict()
            torch.save(enc_state,
                       Path(CFG["output_dir"]) / f"{fold_tag}_best_encoder.pt")
        else:
            wait += 1
            if wait >= patience:
                print(f"{fold_tag} early‑stop (@epoch {ep})")
                break
    best_path = Path(CFG["output_dir"]) / f"{fold_tag}_best_encoder.pt"
    if best_path.exists():
        (model.module if isinstance(model, nn.DataParallel) else model).encoder.load_state_dict(
            torch.load(best_path, map_location=DEVICE))
    return model


def gen_embeddings(loader: DataLoader, encoder: nn.Module, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    encs, labs = np.empty((0, dim), np.float32), np.empty((0,), np.int64)
    with torch.no_grad():
        for batch in loader:
            feat = batch["features"].to(DEVICE)
            lab = batch["labels"].cpu().numpy()
            enc = encoder(feat)
            if isinstance(enc, tuple):
                enc = enc[0]
            enc_out = enc
            encs = np.vstack([encs, enc_out.cpu().numpy()])
            labs = np.concatenate([labs, lab])
    return encs, labs


def linear_probe(train_idx: np.ndarray, test_idx: np.ndarray,
                 raw_X: np.ndarray, raw_y: np.ndarray, raw_pid: np.ndarray,
                 encoder: nn.Module, fold_tag: str, scaler: StandardScaler):
    dataset = FatigueFeatureDataset(raw_X, raw_y, raw_pid, scaler)
    tr_loader = DataLoader(Subset(dataset, train_idx), batch_size=CFG["eval_batch_size"],
                           shuffle=False, num_workers=2, pin_memory=PIN_MEMORY)
    te_loader = DataLoader(Subset(dataset, test_idx), batch_size=CFG["eval_batch_size"],
                           shuffle=False, num_workers=2, pin_memory=PIN_MEMORY)

    encoder.eval()
    tmp = encoder(torch.zeros(1, TOTAL_FEATURES, device=DEVICE))
    dim = tmp[0].shape[1] if isinstance(tmp, tuple) else tmp.shape[1]
    Xtr, ytr = gen_embeddings(tr_loader, encoder, dim)
    Xte, yte = gen_embeddings(te_loader, encoder, dim)

    if len(np.unique(ytr)) < 2 or Xtr.shape[0] < 5:
        print(f"{fold_tag} ‑ linear probe skipped (too few samples/classes)")
        return np.nan

    clf = LogisticRegression(max_iter=2000, solver="saga", n_jobs=-1, C=0.1)
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)
    acc = accuracy_score(yte, ypred)
    print(f"{fold_tag} ‑ probe accuracy {acc:.4f}")

    wandb.log({f"{fold_tag}/probe_acc": acc})
    cls_rep = classification_report(yte, ypred, zero_division=0, output_dict=True)
    for cls, d in cls_rep.items():
        if isinstance(d, dict):
            wandb.log({f"{fold_tag}/f1_{cls}": d.get("f1-score", 0.0)})
    return acc


def preprocess_participant(base_dir: str, pid_str: str, subj_df: pd.DataFrame
                           ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return (X, y, pid_numeric) where X is (m, F) float32."""
    pid_num = int(pid_str[1:])
    p_dir = Path(base_dir) / pid_str
    pattern = re.compile(r"Gesture_(\d+)\.csv$")
    X, y = [], []
    for f in p_dir.glob("Gesture_*.csv"):
        m = pattern.search(f.name)
        if not m:
            continue
        gest_id = int(m.group(1))
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            continue
        if df.empty or "Interval" not in df:
            continue
        for _, row in df.iterrows():
            seg = int(row["Interval"])
            subj_row = (pid_num - 1) * 5 + (gest_id - 1)
            subj_col = seg + 3  # after P_ID, Gesture, Trial
            if (subj_row >= len(subj_df)) or (subj_col >= len(subj_df.columns)):
                continue
            label = subj_df.iat[subj_row, subj_col]
            if pd.isna(label):
                continue
            feat_vec = row.drop("Interval").values.astype(np.float32)
            if np.isfinite(feat_vec).all():
                X.append(feat_vec)
                y.append(int(label))
    if not X:
        return np.empty((0, TOTAL_FEATURES), np.float32), np.empty((0,), np.int64), pid_num
    return np.vstack(X), np.array(y, np.int64), pid_num


def main():
    monitor_memory("start")

    subj_df = pd.read_csv(CFG["subjective_csv_path"])
    participants = sorted([d.name for d in Path(CFG["feature_data_base_dir"]).iterdir()
                           if d.is_dir() and d.name.startswith("P")])
    print("Participants", participants)

    # Preprocess in parallel 
    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_pid: List[np.ndarray] = []
    with cf.ThreadPoolExecutor(max_workers=min(8, os.cpu_count())) as ex:
        fut2pid = {ex.submit(preprocess_participant, CFG["feature_data_base_dir"], p, subj_df): p
                   for p in participants}
        for fut in tqdm(cf.as_completed(fut2pid), total=len(fut2pid), desc="preproc"):
            pid = fut2pid[fut]
            X, y, pid_num = fut.result()
            if X.size == 0:
                print(f"{pid} – no valid windows → skipped")
                continue
            all_X.append(X); all_y.append(y)
            all_pid.append(np.full(len(X), pid_num, np.int32))
    if not all_X:
        raise RuntimeError("No data loaded – aborting")

    raw_X = np.concatenate(all_X, axis=0).astype(np.float32, copy=False)
    raw_y = np.concatenate(all_y, axis=0).astype(np.int64,  copy=False)
    raw_pid = np.concatenate(all_pid, axis=0).astype(np.int32, copy=False)

    del all_X, all_y, all_pid; gc.collect(); monitor_memory("after concat")

    unique_pids = np.unique(raw_pid)
    results = []
    for pid in unique_pids:
        fold = f"P{pid}"
        print("\n======================", fold, "======================")

        test_idx = np.where(raw_pid == pid)[0]
        train_idx = np.where(raw_pid != pid)[0]

        scaler = StandardScaler().fit(raw_X[train_idx])
        dataset_full = FatigueFeatureDataset(raw_X, raw_y, raw_pid, scaler)

        # Split val from train
        tr_idx, va_idx = train_test_split(train_idx, test_size=CFG["val_split"],
                                          stratify=raw_y[train_idx], random_state=42)
        tr_loader = DataLoader(Subset(dataset_full, tr_idx), batch_size=CFG["contrastive_batch_size"],
                               shuffle=True, num_workers=CFG["num_workers"], pin_memory=PIN_MEMORY,
                               drop_last=True)
        va_loader = None
        if len(va_idx):
            va_loader = DataLoader(Subset(dataset_full, va_idx), batch_size=CFG["eval_batch_size"],
                                   shuffle=False, num_workers=CFG["num_workers"], pin_memory=PIN_MEMORY)

        wandb.init(entity=CFG["wandb_entity"], project=CFG["wandb_project"],
                   name=f"{fold}_run", reinit=True, config=CFG)

        net = ContrastiveFatigueEncoder(TOTAL_FEATURES, CFG["encoder_dim"], CFG["projection_dim"])
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
        net.to(DEVICE)

        net = train_contrastive(net, tr_loader, va_loader, fold)

        enc = net.module.encoder if isinstance(net, nn.DataParallel) else net.encoder
        acc = linear_probe(train_idx, test_idx, raw_X, raw_y, raw_pid, enc, fold, scaler)
        results.append((fold, acc))
        wandb.finish()
        monitor_memory(f"after {fold}")

    # Save overall results
    res_df = pd.DataFrame(results, columns=["Participant", "ProbeAcc"])
    res_path = Path(CFG["output_dir"]) / "probe_results.csv"
    res_df.to_csv(res_path, index=False)
    print("\nSaved", res_path)
    print(res_df)

    if not res_df["ProbeAcc"].isna().all():
        mean_acc = res_df["ProbeAcc"].mean()
        std_acc = res_df["ProbeAcc"].std(ddof=0)
        plt.figure(figsize=(5, 4))
        plt.bar(["Avg"], [mean_acc], yerr=[std_acc], capsize=5)
        plt.ylim(0, 1); plt.ylabel("Accuracy"); plt.title("LOSO – Avg Probe Acc")
        plt.tight_layout()
        plt.savefig(Path(CFG["output_dir"], "avg_probe_acc.png"))
        plt.close()
        print(f"Average acc {mean_acc:.4f} ± {std_acc:.4f}")


if __name__ == "__main__":
    main()
