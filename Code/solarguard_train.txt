"""SolarGuard AI: reproducible CNN / CBAM / Focal-Loss experiments.

This reference implementation follows the method specified in the team's
Checkpoint 2. It reads the original Raptor Maps metadata JSON, performs the
reported exact-duplicate audit, creates a deterministic stratified split,
trains the four planned configurations, and evaluates clean and perturbed
copies of the test set.

Verified against the public dataset: the duplicate audit (22 groups, 6
conflicting, 12 + 16 images removed, 19,972 kept) and the 13,980 / 2,996 /
2,996 split are identical to the Checkpoint 2 notebook. Training results come
only from running this script (see README) and are stored in all_results.csv.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, classification_report

CLASS_NAMES = [
    "Cell", "Cell-Multi", "Cracking", "Diode", "Diode-Multi",
    "Hot-Spot", "Hot-Spot-Multi", "No-Anomaly", "Offline-Module",
    "Shadowing", "Soiling", "Vegetation",
]
NUM_CLASSES = len(CLASS_NAMES)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_metadata(data_root: Path) -> list[dict]:
    """Load Raptor Maps module_metadata.json and validate every image path."""
    meta_path = data_root / "module_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Could not find {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    rows = []
    for image_id, item in raw.items():
        rel_path = item.get("image_filepath")
        label = item.get("anomaly_class")
        if not rel_path or not label:
            raise ValueError(f"Invalid metadata entry for image {image_id}")
        path = data_root / rel_path
        if not path.exists():
            raise FileNotFoundError(f"Image listed in metadata is missing: {path}")
        rows.append({"image_id": str(image_id), "path": path, "label": label})
    if len(rows) != 20000:
        raise ValueError(f"Expected 20,000 source images, found {len(rows)}")
    return rows


def file_hash(path: Path) -> str:
    """SHA-256 of the raw file bytes (same fingerprint as the Colab notebook)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_exact_duplicates(rows: list[dict], expected_clean_count: int = 19972) -> tuple[list[dict], dict]:
    """Remove conflicting-label duplicate groups and collapse same-label copies.

    The Checkpoint 2 audit reports 22 exact-duplicate groups, six of which
    contain conflicting labels (12 images removed), followed by 16 additional
    same-label duplicates (first copy in metadata order is kept). The
    assertions below ensure the same counts occur.
    """
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(file_hash(row["path"]), []).append(row)
    duplicate_groups = [g for g in groups.values() if len(g) > 1]
    conflicting = [g for g in duplicate_groups if len({x["label"] for x in g}) > 1]
    same_label = [g for g in duplicate_groups if len({x["label"] for x in g}) == 1]
    removed_conflicting = sum(len(g) for g in conflicting)
    removed_same = sum(len(g) - 1 for g in same_label)
    if len(duplicate_groups) != 22 or len(conflicting) != 6 or removed_conflicting != 12 or removed_same != 16:
        raise ValueError(
            "Duplicate audit does not match Checkpoint 2: "
            f"groups={len(duplicate_groups)}, conflicting_groups={len(conflicting)}, "
            f"conflicting_images={removed_conflicting}, same_label_removed={removed_same}. "
            "Do not silently continue; inspect the dataset/version and document any difference."
        )
    conflicting_ids = {x["image_id"] for g in conflicting for x in g}
    keep_ids = set()
    for g in same_label:
        keep_ids.add(g[0]["image_id"])  # first in metadata order, as in the notebook
    duplicate_ids = {x["image_id"] for g in duplicate_groups for x in g}
    clean = []
    for row in rows:
        if row["image_id"] in conflicting_ids:
            continue
        if row["image_id"] in duplicate_ids and row["image_id"] not in keep_ids:
            continue
        clean.append(row)
    if len(clean) != expected_clean_count:
        raise ValueError(f"Expected {expected_clean_count} cleaned images, found {len(clean)}")
    summary = {
        "source_images": len(rows),
        "duplicate_groups": len(duplicate_groups),
        "conflicting_groups_removed": len(conflicting),
        "conflicting_images_removed": removed_conflicting,
        "same_label_duplicate_images_removed": removed_same,
        "clean_images": len(clean),
    }
    return clean, summary


def make_splits(rows: list[dict], output_dir: Path, seed: int = 42) -> dict[str, list[dict]]:
    """Create 70/15/15 stratified train/validation/test partitions."""
    labels = np.asarray([r["label"] for r in rows])
    idx = np.arange(len(rows))
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.30, random_state=seed, stratify=labels
    )
    temp_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=seed, stratify=temp_labels
    )
    split = {
        "train": [rows[i] for i in train_idx],
        "validation": [rows[i] for i in val_idx],
        "test": [rows[i] for i in test_idx],
    }
    expected = {"train": 13980, "validation": 2996, "test": 2996}
    for key, n in expected.items():
        if len(split[key]) != n:
            raise ValueError(f"Expected {n} {key} images, found {len(split[key])}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in split.items():
        with (output_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_id", "image_filepath", "anomaly_class"])
            for r in subset:
                writer.writerow([r["image_id"], str(r["path"]), r["label"]])
    return split


class SolarDataset(Dataset):
    def __init__(self, rows: Sequence[dict]):
        self.rows = list(rows)
        self.label_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
        unknown = sorted({r["label"] for r in self.rows} - set(CLASS_NAMES))
        if unknown:
            raise ValueError(f"Unknown classes in data: {unknown}")
        self.images, self.labels = [], []
        for row in self.rows:  # decode once; the whole dataset is only ~20 MB
            with Image.open(row["path"]) as im:
                arr = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
            if arr.shape != (40, 24):
                raise ValueError(f"Expected image array shape (40,24), got {arr.shape}: {row['path']}")
            self.images.append(torch.from_numpy(arr).unsqueeze(0))
            self.labels.append(self.label_to_idx[row["label"]])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index]


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel then spatial attention)."""
    def __init__(self, channels: int, reduction: int = 8, spatial_kernel: int = 7):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                                 padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=(2, 3), keepdim=True)
        mx = torch.amax(x, dim=(2, 3), keepdim=True)
        channel_att = self.sigmoid(self.mlp(avg) + self.mlp(mx))
        x = x * channel_att
        spatial_avg = torch.mean(x, dim=1, keepdim=True)
        spatial_max = torch.amax(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.spatial(torch.cat([spatial_avg, spatial_max], dim=1)))
        return x * spatial_att


class SolarGuardCNN(nn.Module):
    def __init__(self, use_attention: bool = False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.attention = CBAM(32) if use_attention else nn.Identity()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 5 * 3, 64), nn.ReLU(inplace=True),
            nn.Dropout(p=0.30), nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.block1(x))
        x = self.attention(self.block2(x))
        x = self.pool2(x)
        x = self.pool3(self.block3(x))
        if x.shape[1:] != (64, 5, 3):
            raise RuntimeError(f"Unexpected feature shape before classifier: {tuple(x.shape[1:])}")
        return self.classifier(x)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.view(-1, 1)).squeeze(1)
        pt = log_pt.exp()
        return (-((1.0 - pt) ** self.gamma) * log_pt).mean()


@dataclass(frozen=True)
class Config:
    name: str
    attention: bool
    focal: bool


CONFIGS = [
    Config("CNN_CE", False, False),
    Config("CNN_CBAM_CE", True, False),
    Config("CNN_Focal", False, True),
    Config("CNN_CBAM_Focal", True, True),
]


def loader(rows: Sequence[dict], batch_size: int, shuffle: bool, seed: int = 42) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(SolarDataset(rows), batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=torch.cuda.is_available(), generator=generator)


def scores(y_true: list[int], y_pred: list[int]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def train_one(model: nn.Module, train_dl: DataLoader, val_dl: DataLoader,
              focal: bool, device: torch.device, output_dir: Path,
              seed: int, max_epochs: int = 50, patience: int = 8) -> tuple[nn.Module, dict]:
    criterion = FocalLoss(gamma=2.0) if focal else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    start = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        train_n = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(yb)
            train_n += len(yb)
        model.eval()
        val_true, val_pred = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                logits = model(xb.to(device))
                val_pred.extend(logits.argmax(1).cpu().tolist())
                val_true.extend(yb.tolist())
        val_metrics = scores(val_true, val_pred)
        history.append({"epoch": epoch, "train_loss": train_loss / max(1, train_n),
                        "validation_macro_f1": val_metrics["macro_f1"]})
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "validation_macro_f1"])
        writer.writeheader(); writer.writerows(history)
    torch.save(model.state_dict(), output_dir / "best_model.pt")
    return model, {"best_validation_macro_f1": best_f1, "best_epoch": best_epoch,
                   "epochs_trained": len(history), "training_seconds": time.time() - start}


def perturb(x: torch.Tensor, kind: str, strength: float, generator: torch.Generator) -> torch.Tensor:
    if kind == "clean":
        out = x
    elif kind == "brightness":
        out = x + strength
    elif kind == "contrast":
        mean = x.mean(dim=(2, 3), keepdim=True)
        out = (x - mean) * strength + mean
    elif kind == "gaussian_noise":
        noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype) * strength
        out = x + noise
    else:
        raise ValueError(f"Unknown perturbation: {kind}")
    return out.clamp(0.0, 1.0)


def evaluate(model: nn.Module, rows: Sequence[dict], device: torch.device,
             condition: tuple[str, float], seed: int = 2026) -> tuple[dict, list[int], list[int]]:
    kind, strength = condition
    dl = loader(rows, batch_size=64, shuffle=False)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in dl:
            xb = perturb(xb.to(device), kind, strength, generator)
            logits = model(xb)
            y_true.extend(yb.tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())
    return scores(y_true, y_pred), y_true, y_pred


def save_report(output_dir: Path, config_name: str, seed: int,
                metrics: dict, y_true: list[int], y_pred: list[int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = list(range(NUM_CLASSES))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.savetxt(output_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    report = classification_report(y_true, y_pred, labels=labels,
                                   target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    with (output_dir / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def run(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    rows = load_metadata(data_root)
    clean_rows, audit = audit_exact_duplicates(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "duplicate_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    def write_distribution(path: Path, items: Sequence[dict]) -> None:
        counts = Counter(r["label"] for r in items)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["class", "count"])
            for class_name in CLASS_NAMES:
                writer.writerow([class_name, counts.get(class_name, 0)])

    write_distribution(output_root / "source_class_distribution.csv", rows)
    write_distribution(output_root / "clean_class_distribution.csv", clean_rows)
    split = make_splits(clean_rows, output_root / "splits", seed=42)
    with (output_root / "split_class_distribution.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "class", "count"])
        for split_name, subset in split.items():
            counts = Counter(r["label"] for r in subset)
            for class_name in CLASS_NAMES:
                writer.writerow([split_name, class_name, counts.get(class_name, 0)])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    selected_configs = [c for c in CONFIGS if args.config == "all" or c.name == args.config]
    if not selected_configs:
        raise ValueError(f"Unknown configuration {args.config}")
    for cfg in selected_configs:
        for seed in args.seeds:
            seed_everything(seed)
            run_dir = output_root / cfg.name / f"seed_{seed}"
            model = SolarGuardCNN(use_attention=cfg.attention).to(device)
            train_dl = loader(split["train"], batch_size=64, shuffle=True, seed=seed)
            val_dl = loader(split["validation"], batch_size=64, shuffle=False, seed=seed)
            model, train_info = train_one(model, train_dl, val_dl, cfg.focal,
                                          device, run_dir, seed,
                                          max_epochs=args.max_epochs, patience=8)
            conditions = [("clean", 0.0)]
            conditions += [("brightness", s) for s in (-0.4, -0.2, 0.2, 0.4)]
            conditions += [("contrast", s) for s in (0.6, 0.8, 1.2, 1.4)]
            conditions += [("gaussian_noise", s) for s in (0.05, 0.10, 0.20)]
            for kind, strength in conditions:
                metrics, y_true, y_pred = evaluate(model, split["test"], device,
                                                   (kind, strength), seed=1000 + seed)
                condition_name = f"{kind}_{strength:g}"
                save_report(run_dir / condition_name, cfg.name, seed, metrics, y_true, y_pred)
                row = {"configuration": cfg.name, "seed": seed,
                       "condition": condition_name, **train_info, **metrics}
                results.append(row)
                print(row)
    results_path = output_root / "all_results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as f:
        fields = list(results[0].keys()) if results else []
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(results)
    print(f"Saved results to {results_path}")


def smoke_test() -> None:
    """Fast architecture/loss test; no dataset or GPU required."""
    for attention in (False, True):
        model = SolarGuardCNN(use_attention=attention)
        x = torch.rand(2, 1, 40, 24)
        logits = model(x)
        assert logits.shape == (2, NUM_CLASSES), logits.shape
        loss = FocalLoss()(logits, torch.tensor([0, 1]))
        assert torch.isfinite(loss).item()
    print("Smoke test passed: baseline/CBAM shapes and Focal Loss are valid.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate SolarGuard AI configurations.")
    parser.add_argument("--data-root", type=str, help="Folder containing module_metadata.json and images/")
    parser.add_argument("--output-dir", default="outputs", type=str)
    parser.add_argument("--config", default="all", choices=["all"] + [c.name for c in CONFIGS])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if not args.smoke_test and not args.data_root:
        parser.error("--data-root is required unless --smoke-test is used")
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.smoke_test:
        smoke_test()
    else:
        run(cli_args)
