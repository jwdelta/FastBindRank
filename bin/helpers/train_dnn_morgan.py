#!/usr/bin/env python3

import os, sys, csv, argparse, random, time
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class MorganSparseDataset(Dataset):
    """
    指纹：每行 'ligand_id,idx1,idx2,...'（0基）
    标签文件：含 id_col 和 label_col；默认使用“软标签”[0,1]。
    当 --hard_label 打开时，按 --threshold 二值化。
    支持 labels_path 为空 -> 推断模式（无标签）。
    """
    def __init__(self,
                 fp_path: str, # Path to fingerprint file
                 labels_path: Optional[str], # Path to labels file; None/不存在/空文件=>推断模式
                 bits: int, # Fingerprint length (e.g. 1024 or 2048 bits)
                 one_based_indices: bool, # Whether fingerprint indices are 1-based (True) or 0-based (False)
                 id_col: str, # Column name in labels file for ligand IDs
                 label_col: str, # Column name in labels file for target labels
                 hard_label: bool, # If True: convert probability labels to binary {0,1} using threshold
                 threshold: float): # Threshold for hard labels (default 0.5)
        self.fp_path = fp_path
        self.labels_path = labels_path
        self.bits = bits
        self.one_based_indices = one_based_indices
        self.id_col = id_col
        self.label_col = label_col
        self.hard_label = hard_label
        self.threshold = threshold

        # 判定是否有可用标签文件
        self.has_labels = (
                labels_path is not None and
                os.path.exists(labels_path) and
                os.path.getsize(labels_path) > 0)

        # read labels
        self.labels: Dict[str, float] = {}
        if self.has_labels:
            with open(self.labels_path, 'r', newline='') as f:
                reader = csv.DictReader(f, delimiter="\t")
                if self.id_col not in reader.fieldnames or self.label_col not in reader.fieldnames:
                    raise ValueError(f"labels file must has: {self.id_col}, {self.label_col}; Now is: {reader.fieldnames}")
                #row = next(reader)
                for row in reader:
                    lid = str(row[self.id_col])
                    v = row[self.label_col]
                    if lid == '' or v is None or v == '':
                        continue
                    try:
                        y = float(v)
                    except Exception:
                        continue
                    # Soft label: keep 0~1; hard label: binary based on threshold
                    if self.hard_label:
                        y = 1.0 if y > self.threshold else 0.0
                    else:
                        y = float(min(1.0, max(0.0, y)))
                    self.labels[lid] = y

        # read Morgan fingerprint
        self.items: List[Tuple[str, List[int]]] = []
        with open(self.fp_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split(',')
                if len(parts) < 1: continue
                ligand_id = parts[0]
                if len(parts) > 1:
                    try:
                        idxs = [int(x) for x in parts[1:]]
                    except Exception:
                        continue
                if self.one_based_indices:
                    idxs = [i-1 for i in idxs]
                idxs = [i for i in idxs if 0 <= i < self.bits]

                if self.has_labels:
                    if ligand_id in self.labels:
                        self.items.append((ligand_id, idxs))
                else:
                    self.items.append((ligand_id, idxs))

        if not self.items:
            raise ValueError("No samples were read; please check the fingerprint file/label file and column names.")

        if self.has_labels:
            self.y = np.array([self.labels[lig] for lig,_ in self.items], dtype=np.float32)
        else:
            self.y = None  # 推断模式

    def __len__(self): return len(self.items)

    def __getitem__(self, i: int):
        lig, bit_idx = self.items[i]
        if self.has_labels:
            y = float(self.y[i])
        else:
            y = None  # 推断模式下无标签
        return lig, bit_idx, y

def collate_to_dense(batch, bits: int):
    """
    兼容有/无标签：
        - 有标签：返回 (ids, X, y: Tensor[bs])
        - 无标签：返回 (ids, X, None)
    """
    bs = len(batch)
    X = np.zeros((bs, bits), dtype=np.float32)
    ids: List[str] = []
    has_y = (bs > 0 and batch[0][2] is not None)

    if has_y:
        y = np.zeros((bs,), dtype=np.float32)
    else:
        y = None
    
    for i, (lig, idxs, yi) in enumerate(batch):
        ids.append(lig)
        if idxs:
            X[i, idxs] = 1.0
        if has_y:
            y[i] = float(yi)
    
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y) if has_y else None
    return ids, X_t, y_t

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        layers = []; prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev,h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev,1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(1)

def make_loader(fp, labels, args, shuffle):
    """
    - 训练/验证/测试: labels 有效
    - 推断/应用: labels 无效
    """
    # 判定标签是否可用
    has_labels = (
            labels is not None and
            os.path.exists(labels) and
            os.path.getsize(labels) > 0)

    ds = MorganSparseDataset(
        fp_path=fp, labels_path=labels if has_labels else None, bits=args.bits,
        one_based_indices=args.one_based_indices, id_col=args.id_col, label_col=args.label_col,
        hard_label=args.hard_label, threshold=args.threshold
    )

    # 训练才需要 drop_last；验证/测试/推断不丢样本
    drop_last = bool(has_labels and shuffle) # 因为我们指定了train时才shuffle=TRUE

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=lambda b: collate_to_dense(b, args.bits),
        persistent_workers=(args.num_workers > 0),
        drop_last=drop_last
    )
    return ds, loader

# Create an optimizer for `model.parameters()` based on a string, a class, a callable, or a ready instance
def build_optimizer(model, optimizer='adam', lr=1e-3, weight_decay=0.0, momentum=0.9, optimizer_kwargs=None):
    """
    Parameters
    model : The model whose parameters will be optimized.
    optimizer (str or torch.optim.Optimizer): Name of the optimizer ("adam", "adamw", "sgd", "rmsprop", etc.) or an optimizer instance.
    lr (float, default=1e-3): Learning rate. Used by all optimizers.
    weight_decay (float, default=0.0): L2 weight decay. Supported by Adam/AdamW/SGD/RMSprop/Adagrad. Increase if overfitting.
    momentum (float, default=0.9): Momentum factor. Used only by SGD (and RMSprop). If training is unstable, lower momentum slightly (e.g., 0.9 → 0.8); if convergence is slow, increase it (e.g., 0.95).
    optimizer_kwargs (dict, default=None): Extra keyword arguments passed to the optimizer.
        Examples:
            - Adam/AdamW: betas, eps, amsgrad
            - SGD: nesterov, dampening
            - RMSprop: alpha, eps, centered
            - Adagrad: lr_decay, initial_accumulator_value
    """
    if isinstance(optimizer, torch.optim.Optimizer):
        return optimizer
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    # Map common strings to optimizer classes
    opt_map = {
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW,
            'sgd': torch.optim.SGD,
            'rmsprop': torch.optim.RMSprop,
            'adagrad': torch.optim.Adagrad}

    if isinstance(optimizer, str):
        key = optimizer.lower()
        if key not in opt_map:
            raise ValueError(f"Unknown optimizer '{optimizer}'. Choose from {list(opt_map.keys())}.")
        opt_cls = opt_map[key]
    elif callable(optimizer):
        opt_cls = optimizer
    else:
        raise ValueError("`optimizer` must be a string, an optimizer instance, a class, or a callable.")

    # Default/common kwargs per optimizer
    common = dict(lr=lr, weight_decay=weight_decay)
    if opt_cls is torch.optim.SGD:
        common.update(dict(momentum=momentum, nesterov=True))
    if opt_cls is torch.optim.RMSprop:
        common.update(dict(momentum=momentum))

    # User-supplied kwargs override defaults
    common.update(optimizer_kwargs)

    return opt_cls(model.parameters(), **common)


@torch.no_grad()
def evaluate(model, loader, device, criterion, threshold=0.5, use_amp=False, hard_label=False, 
        plot=False, out_prefix=None):
    model.eval()
    total_loss=0.0; total=0; correct=0
    probs_all=[]; labels_all=[]

    for ids, X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda" if use_amp==True else "cpu", enabled=use_amp):
            logits = model(X)
            loss = criterion(logits, y)
        bs = y.size(0)
        total_loss += loss.item() * bs
        total += bs
        probs_all.append(torch.sigmoid(logits).detach().cpu().numpy())
        labels_all.append(y.detach().cpu().numpy())

    avg_loss = total_loss / max(1,total)
    p = np.concatenate(probs_all) # 预测概率
    l = np.concatenate(labels_all) # 标签（可能是0/1，也可能是概率）

    if hard_label:
        # 硬标签任务：分类指标
        preds = (p >= threshold).astype(int)
        acc = accuracy_score(l, preds)
        f1 = f1_score(l, preds)
        auc = roc_auc_score(l, p) if len(np.unique(l)) > 1 else None

        if plot and out_prefix is not None and auc is not None:
            fpr, tpr, _ = roc_curve(l, p)
            plt.figure(figsize=(5,5))
            plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
            plt.plot([0,1],[0,1],'k--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            text = f"ACC={acc:.3f}\nF1={f1:.3f}\nAUC={auc:.3f}"
            #plt.annotate(text, xy=(0.6,0.2), xycoords="axes fraction",
            #        bbox=dict(boxstyle="round", fc="w"))
            plt.legend()
            plt.savefig(f"{out_prefix}_roc_curve.pdf", bbox_inches="tight")
            plt.close()

        return avg_loss, acc, f1, auc, None  # brier=None

    else:
        p = np.asarray(p, dtype=float).reshape(-1)
        l = np.asarray(l, dtype=float).reshape(-1)
        if p.shape[0] != l.shape[0]:
            raise ValueError(f"Shape mismatch: p has {p.shape[0]} elements but l has {l.shape[0]}.")

        # Brier score（= MSE for probabilities)
        brier = float(np.mean((p - l) ** 2))
        # Correlations
        pearson = spearman = None
        if np.std(l) > 1e-8 and np.std(p) > 1e-8:
            pearson = pearsonr(l, p)[0]
            spearman = spearmanr(l, p)[0]
        # Soft PR curve + Soft PR-AUC
        # The area under the precision–recall curve computed using soft labels, measuring how well true binders are enriched toward the top of the ranked list under class imbalance.
        def soft_pr_curve_and_auc(y_soft, y_score):
            order = np.argsort(-y_score)  # high → low
            y_soft_sorted = y_soft[order]
            total_pos = float(np.sum(y_soft_sorted))
            if total_pos <= 1e-12:
                return None, None, None
            soft_tp_cum = np.cumsum(y_soft_sorted)
            k = np.arange(1, len(y_soft_sorted) + 1)
            precision = soft_tp_cum / k
            recall = soft_tp_cum / total_pos
            pr_auc = float(np.trapz(precision, recall))
            return precision, recall, pr_auc
        prec_curve, rec_curve, soft_pr_auc = soft_pr_curve_and_auc(l, p)
        # Soft Precision@K
        # the average soft-label (expected binding probability) among the top K ranked candidates, interpreted as the expected hit rate when selecting K compounds for verification.
        def soft_precision_at_k(y_soft, y_score, K):
            n = len(y_soft)
            if n < K or K <= 0:
                return None
            order = np.argsort(-y_score)
            return float(np.sum(y_soft[order[:K]]) / K)
        K_list = [100, 1000, 10000, 100000]
        n = len(l)
        prec_at_k = {K: soft_precision_at_k(l, p, K) for K in K_list if n >= K}

        # Annotation text (used in both plots)
        textlines = [f"Brier={brier:.3f}"]
        if pearson is not None: textlines.append(f"Pearson={pearson:.3f}")
        if spearman is not None: textlines.append(f"Spearman={spearman:.3f}")
        textlines.append(f"Soft PR-AUC={soft_pr_auc:.3f}" if soft_pr_auc is not None else "Soft PR-AUC=NA")
        for K in K_list:
            if K in prec_at_k:
                textlines.append(f"Prec@{K}={prec_at_k[K]:.3f}")

        anno_text = "\n".join(textlines)

        if plot and out_prefix is not None:
            # 1) Scatter: Prediction vs Label
            plt.figure(figsize=(5,5))
            plt.scatter(l, p, alpha=0.5, s=5)
            plt.xlabel("True Probability")
            plt.ylabel("Predicted Probability")
            plt.title("Prediction vs Soft Label")
            #plt.annotate(anno_text, xy=(0.05,0.95), xycoords="axes fraction", va="top",
            #        bbox=dict(boxstyle="round", fc="w"))
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.savefig(f"{out_prefix}_prediction_scatter.pdf", bbox_inches="tight")
            plt.close()
            # 2) Soft PR Curve
            baseline = float(np.mean(l))
            plt.figure(figsize=(5, 5))
            plt.plot(rec_curve, prec_curve, label=f"Soft PR-AUC={soft_pr_auc:.3f}")
            plt.plot([0,1], [baseline, baseline], 'k--')
            
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Soft Precision–Recall Curve")
            #plt.annotate(anno_text, xy=(0.05,0.95), xycoords="axes fraction", va="top",
            #        bbox=dict(boxstyle="round", fc="w"))
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.legend()
            plt.savefig(f"{out_prefix}_soft_pr_curve.pdf", bbox_inches="tight")
            plt.close()

        return avg_loss, brier, pearson, spearman, soft_pr_auc, prec_at_k


def plot_losses(train_losses, val_losses, output_dir, best_epoch=None):
    """
    绘制训练和验证 Loss 曲线，并保存为 PDF 文件。
    Args:
        train_losses (list): 每个 epoch 的训练 loss
        val_losses (list): 每个 epoch 的验证 loss
        output_dir (str): 输出目录
        best_epoch (int, 可选): 如果提供，则在最佳验证点画竖线
    """
    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure(figsize=(8,6))
    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs, val_losses, label="Validation Loss", marker="s")
    if best_epoch is not None:
        plt.axvline(best_epoch, color="red", linestyle="--", label=f"Best Epoch {best_epoch}")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    #plt.xticks(epochs) # 强制横坐标显示为整数
    plt.xticks(np.arange(1, len(train_losses)+1, 5))

    out_path = os.path.join(output_dir, "loss_curve.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[Info] Saved loss curve -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="DNN on Morgan fingerprints with hard/soft-label BCE (Binary Cross-Entropy loss).")
    ap.add_argument("--train_fp", required=True)
    ap.add_argument("--train_labels", required=True)
    ap.add_argument("--val_fp", required=True)
    ap.add_argument("--val_labels", required=True)
    ap.add_argument("--test_fp", default=None)
    ap.add_argument("--test_labels", default=None)
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--bits", type=int, default=1024)
    ap.add_argument("--hidden_dims", type=str, default="1024,512,256")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--optimizer", type=str, default="adam")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--momentum", type=float, default=0.9)

    ap.add_argument("--one_based_indices", type=lambda s: s.lower() in ["1","true","yes","y"], default=False)
    ap.add_argument("--id_col", type=str, default="CID")
    ap.add_argument("--label_col", type=str, default="affinity_probability_binary")

    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--early_stop_patience", type=int, default=8)
    ap.add_argument("--save_probabilities", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.5, help="分类阈值（评估/导出时）")
    ap.add_argument("--mode", type=str, help="train/test")
    ap.add_argument("--test_out_prefix", type=str)

    ap.add_argument("--finetune_from", type=str, default=None, help="Path to a checkpoint to finetune from. If not set, train from scratch.")

    # 新增：软/硬标签选择
    ap.add_argument("--hard_label", action="store_true", help="将概率阈值化为0/1再训练")
    ap.add_argument("--pos_weight", type=str, default="auto", help="'auto'或具体数字。如果正负样本比例严重失衡，模型会倾向于预测多数类，用pos_weight给正样本的 loss 加权")

    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "ckpts"), exist_ok=True)
    log_path = os.path.join(args.output_dir, "train.log")
    log_f = open(log_path, "a")   # 用 "a" 追加模式，这样不会覆盖
    set_seed(args.seed)
    device = torch.device(args.device)

    # 模型
    hidden_dims = [int(x) for x in args.hidden_dims.split(",") if x.strip()!='']
    model = MLP(args.bits, hidden_dims, args.dropout).to(device)

    # 数据
    train_ds,train_loader = make_loader(args.train_fp, args.train_labels, args, shuffle=True)
    val_ds,val_loader = make_loader(args.val_fp, args.val_labels, args, shuffle=False)
    # pos_weight
    y_train = train_ds.y
    if args.hard_label:
        y_train = (y_train > args.threshold).astype(np.float32)
        pos_rate = float(np.mean(y_train))
        if isinstance(args.pos_weight, str) and args.pos_weight.lower()=="auto":
            pw = (1.0 - pos_rate) / max(1e-8, pos_rate) if pos_rate>0 else 1.0
        else:
            pw = float(args.pos_weight)
        pos_weight = torch.tensor([pw], dtype=torch.float32, device=device)
        msg_pw = f"[Info] Train pos_rate={pos_rate:.4f} -> pos_weight={pos_weight.item():.4f}"
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = build_optimizer(
            model,
            optimizer=args.optimizer,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum)
    use_amp = args.amp and args.device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) # scaler是AMP(Automatic Mixed Precision)的梯度缩放器，如果不用 AMP，它会自动降级成普通训练。AMP 的梯度缩放器 (GradScaler) = 自动把 loss 放大/缩小，保证 FP16 梯度既不会下溢为 0，也不会溢出成 Inf，从而让混合精度训练既快又稳定

    # finetune: 如果提供了路径，就加载 checkpoint
    if args.finetune_from is not None and os.path.exists(args.finetune_from):
        msg_announce = f"[Info] Finetuning from checkpoint -> {args.finetune_from}"
        ckpt = torch.load(args.finetune_from, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
    else:
        msg_announce = "[Info] Training from scratch."


    # 训练
    if args.mode == "train":
        print(msg_announce)
        if args.hard_label:
            print(msg_pw)
            print(msg_pw, file=log_f)
            log_f.flush()

        best_val = float("inf") # record the best validation loss
        patience = args.early_stop_patience; no_improve = 0
        train_losses = []; val_losses = []

        for ep in range(1, args.epochs+1):
            this_ckpt = os.path.join(args.output_dir, "ckpts", "model_"+str(ep)+".pt")
            model.train(); t0=time.time()
            run_loss=0.0; total=0
            for ids, X, y in train_loader: # ids, X, y = next(iter(train_loader))可以提取一个batch来测试forward和loss
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True) # 清空梯度
                # 前向计算
                with torch.amp.autocast(device_type="cuda" if use_amp==True else "cpu", enabled=use_amp): # 用 autocast 开启混合精度 (如果args.amp=True)
                    logits = model(X) # 前向传播
                    loss = criterion(logits, y) # 计算损失（BCEWithLogitsLoss）
                # 反向传播
                scaler.scale(loss).backward() # 缩放 loss，反向传播梯度
                scaler.step(optimizer); scaler.update() # 更新参数和动态调整缩放因子
                # 累计所有batch的训练损失。loss.item()*bs是因为要用平均loss乘以batch size，得到总损失
                bs = y.size(0)
                run_loss += loss.item()*bs; total += bs

            tr_loss = run_loss/max(1,total) # 训练集的平均损失
            if args.hard_label:
                val_loss, val_acc, val_f1, val_auc, val_ = evaluate(model, val_loader, device, criterion, threshold=args.threshold, use_amp=use_amp, hard_label=args.hard_label)
            else:
                val_loss, val_brier, val_pearson, val_spearman, val_soft_pr_auc, val_prec_at_k = evaluate(model, val_loader, device, criterion, use_amp=use_amp, hard_label=args.hard_label)

            # 保存模型并记录loss
            #torch.save({"model": model.state_dict(), "args": vars(args),
            #    "bits": args.bits, "hidden_dims": hidden_dims}, this_ckpt)
            train_losses.append(tr_loss)
            val_losses.append(val_loss)

            # 打印日志
            dt = time.time()-t0
            if args.hard_label:
                msg = f"Epoch {ep:03d} | {dt:.1f}s | train_loss {tr_loss:.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | val_f1 {val_f1:.4f} | val_auc {val_auc:.4f}"
            else:
                def fmt4(x):
                    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"
                msg = f"Epoch {ep:03d} | {dt:.1f}s | train_loss {tr_loss:.4f} | val_loss {val_loss:.4f} | val_brier {val_brier:.4f} | val_pearson {fmt4(val_pearson)} | val_spearman {fmt4(val_spearman)}"
            print(msg)
            print(msg, file=log_f)
            log_f.flush()

            if val_loss + 1e-6 < best_val: # 保存模型&早停。如果验证损失变小,更新最佳值并保存模型checkpoint;否则no_improve+1,直至早停
                best_val = val_loss; no_improve = 0
                torch.save({"model": model.state_dict(), "args": vars(args), 
                    "bits": args.bits, "hidden_dims": hidden_dims}, os.path.join(args.output_dir, "ckpts", "best_model.pt"))
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"[EarlyStop] no improvement for {patience} epochs."); break


        # 训练结束后画图
        #plot_losses(train_losses, val_losses, args.output_dir, best_epoch=np.argmin(val_losses)+1)
        plot_losses(train_losses, val_losses, args.output_dir)

    if args.mode == "test":
        # labels 可以为 None/空文件，此时 make_loader 会进入推断模式
        test_ds,test_loader = make_loader(args.test_fp, args.test_labels, args, shuffle=False)

        print(f"[Info] Test from checkpoint -> best_model.pt")
        ckpt = torch.load(os.path.join(args.output_dir, "ckpts", "best_model.pt"), map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])

        # 统一的导出函数：有标签就同时评估并画图；无标签只导出概率
        if args.save_probabilities:
            @torch.no_grad()
            def dump_probs(ds, loader, out_csv, out_metrics, out_prefix):
                model.eval(); rows=[]
                has_labels = getattr(ds, "has_labels", False)

                for ids, X, y in loader:
                    X = X.to(device, non_blocking=True)
                    probs = torch.sigmoid(model(X)).detach().cpu().numpy().tolist()
                    for i, lig in enumerate(ids):
                        if has_labels:
                            rows.append((lig, float(y[i].item()), float(probs[i])))
                        else:
                            rows.append((lig, float(probs[i])))

                # 保存预测结果
                with open(out_csv, "w", newline="") as f:
                    w = csv.writer(f)
                    if has_labels:
                        w.writerow(["ligand_id","label","prob"])
                    else:
                        w.writerow(["ligand_id", "prob"])
                    w.writerows(rows)
                print(f"[Info] Saved probs -> {out_csv}")

                # 仅在有标签时调用 evaluate 画图并保存 pdf
                if has_labels:
                    metrics = evaluate(model, loader, device, criterion, use_amp=use_amp, hard_label=args.hard_label, plot=True, out_prefix=out_prefix)
                    print(f"[Info] Saved plot -> {out_prefix}_*.pdf")
                    
                    # Write metrics to txt
                    with open(out_metrics, "w") as f:
                        f.write("metric\tvalue\n")
                        if args.hard_label:
                            val_loss, val_acc, val_f1, val_auc, _ = metrics
                            f.write(f"loss\t{val_loss:.6f}\n")
                            f.write(f"accuracy\t{val_acc:.6f}\n")
                            f.write(f"f1\t{val_f1:.6f}\n")
                            f.write(f"roc_auc\t{val_auc:.6f}\n")
                        else:
                            val_loss, val_brier, val_pearson, val_spearman, val_soft_pr_auc, val_prec_at_k = metrics
                            f.write(f"loss\t{val_loss:.6f}\n")
                            f.write(f"brier\t{val_brier:.6f}\n")
                            f.write(f"pearson\t{val_pearson:.6f}\n" if val_pearson is not None else "pearson\tNA\n")
                            f.write(f"spearman\t{val_spearman:.6f}\n" if val_spearman is not None else "spearman\tNA\n")
                            f.write(f"soft_pr_auc\t{val_soft_pr_auc:.6f}\n" if val_soft_pr_auc is not None else "soft_pr_auc\tNA\n")
                            for K in sorted(val_prec_at_k.keys()):
                                f.write(f"prec_at_{K}\t{val_prec_at_k[K]:.6f}\n")
                else:
                    print("[Info] No labels provided; skip evaluation/plot.")

            out_csv = os.path.join(args.output_dir, args.test_out_prefix + "_pred_out.csv")
            out_metrics = os.path.join(args.output_dir, args.test_out_prefix + "_metrics.txt")
            out_prefix = os.path.join(args.output_dir, args.test_out_prefix)
            dump_probs(test_ds, test_loader, out_csv, out_metrics, out_prefix)


if __name__ == "__main__":
    main()
