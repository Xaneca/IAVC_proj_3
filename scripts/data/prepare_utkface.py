""" 
- Extract labels
- Check outliers - images wrongly labeled
- Split data into train, val and test: 70 / 15 / 15 -> data/splits/utkface.csv
  (path, age, gender, age_bin, age_class, split)
- Data statistics: age histograms, gender distribution -> saves in docs/figs/utkface
"""

from logging import config
import os
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # save figures without opening windows
import matplotlib.pyplot as plt

DATA_DIR = None
REPO = Path(__file__).resolve().parents[2]
FIGURES_DIR = REPO / "docs/figs/utkface"
SPLITS_CSV = REPO / "data/splits/utkface.csv"

# Figure style
GENDER_NAMES = {0: "Masculino", 1: "Feminino"}
GENDER_COLORS = {0: "#2a78d6", 1: "#eb6834"}  # validated categorical pair (colorblind-safe)
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"

def get_path():
    """Lê o ficheiro config.yaml"""
    config_path = os.path.join(os.path.dirname(__file__), "../../config.yaml")

    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    data_config = config.get("data", {})
    age_gender_config = data_config.get("age_gender", {})

    images_path = age_gender_config.get(
        "path",
        os.path.join("data", "UTKFace", "UTKFace"),
    )

    config_dir = os.path.dirname(config_path)
    images_path = os.path.join(config_dir, images_path)

    global DATA_DIR
    DATA_DIR = images_path

    return images_path

def get_labels(images_path):
    # Get Labels from image names
    with os.scandir(images_path) as entries:
        labels = []
        for entry in entries:
            if entry.is_file() and entry.name.endswith(".jpg"):
                # Extract age, gender, and ethnicity from the filename
                filename = entry.name
                age, gender = filename.split("_")[:2]
                labels.append({"path": filename, "age": int(age), "gender": int(gender)})
    labels = pd.DataFrame(labels)

    # Sort numerically by age (scandir returns alphabetical order: 1, 10, 100, 101, ...)
    labels = labels.sort_values(["age", "gender", "path"]).reset_index(drop=True)

    # Age discretization: [0-5), [5-10), then 10-year bins up to 80, and 80+
    age_bins = [0, 5] + list(np.arange(10, 90, 10)) + [np.inf]
    age_bin_names = ["0-4", "5-9"] + [f"{a}-{a + 9}" for a in range(10, 80, 10)] + ["80+"]

    labels["age_bin"] = pd.cut(labels["age"], bins=age_bins, labels=age_bin_names, right=False)
    labels["age_class"] = labels["age_bin"].cat.codes

    return labels

def data_split(labels_df, test_size=0.15, val_size=0.15, random_state=42):

    first_split_size = test_size + val_size
    second_split_size = val_size / first_split_size

    # Split the data into train, validation, and test sets
    train_df, temp_df = train_test_split(labels_df, test_size=first_split_size, random_state=42, stratify=labels_df[["age_class", "gender"]])
    val_df, test_df = train_test_split(temp_df, test_size=second_split_size, random_state=42, stratify=temp_df[["age_class", "gender"]])

    # One CSV with a split column (same format as affectnet.csv / btfer7.csv)
    splits = pd.concat([train_df.assign(split="train"), val_df.assign(split="val"),
                        test_df.assign(split="test")])
    splits = splits.sort_values(["age", "gender", "path"])
    SPLITS_CSV.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(SPLITS_CSV, index=False)
    print(f"Saved {SPLITS_CSV} ({len(splits)} images: "
          + ", ".join(f"{k}={v}" for k, v in splits["split"].value_counts().items()) + ")")
    return splits

#### FIGURES #####

def _style_axes(ax, title, xlabel, ylabel="Nº de imagens"):
    """Recessive axes: no top/right spines, light horizontal grid, muted ticks."""
    ax.set_title(title, loc="left", fontsize=12, color=INK_PRIMARY, pad=10)
    ax.set_xlabel(xlabel, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.tick_params(colors=INK_MUTED, length=0)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)


def _save(fig, out_dir, name):
    fig.tight_layout()
    fig.savefig(out_dir / name, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def data_statistics(labels_df, out_dir=FIGURES_DIR):
    """Saves age/gender distribution figures (per year and per age bin) to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ages = np.arange(labels_df["age"].min(), labels_df["age"].max() + 1)
    bin_names = labels_df["age_bin"].cat.categories
    x_bins = np.arange(len(bin_names))

    # counts per (age, gender) and (bin, gender), with zeros for missing combinations
    age_counts = labels_df.groupby(["age", "gender"]).size().unstack(fill_value=0).reindex(ages, fill_value=0)
    bin_counts = labels_df.groupby(["age_bin", "gender"], observed=False).size().unstack(fill_value=0)

    # 1) Age per year - all data
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(ages, age_counts.sum(axis=1), width=1.0, color=GENDER_COLORS[0],
           edgecolor=SURFACE, linewidth=0.5)
    _style_axes(ax, f"Distribuição de idades (n = {len(labels_df)})", "Idade (anos)")
    ax.set_xlim(ages[0] - 1, ages[-1] + 1)
    _save(fig, out_dir, "age_hist.png")

    # 2) Age per bin - all data
    fig, ax = plt.subplots(figsize=(10, 4.5))
    totals = bin_counts.sum(axis=1)
    bars = ax.bar(x_bins, totals, width=0.7, color=GENDER_COLORS[0])
    ax.bar_label(bars, labels=[f"{v}" for v in totals], padding=3, color=INK_SECONDARY, fontsize=9)
    _style_axes(ax, "Distribuição por classe de idade", "Classe de idade (anos)")
    ax.set_xticks(x_bins, bin_names)
    _save(fig, out_dir, "age_bin_hist.png")

    # 3) Age per year by gender
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for g, name in GENDER_NAMES.items():
        ax.step(ages, age_counts.get(g, 0), where="mid", color=GENDER_COLORS[g],
                linewidth=2, label=f"{name} (n = {age_counts.get(g, pd.Series(0)).sum()})")
    _style_axes(ax, "Distribuição de idades por género", "Idade (anos)")
    ax.set_xlim(ages[0] - 1, ages[-1] + 1)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY)
    _save(fig, out_dir, "age_gender_hist.png")

    # 4) Age bin by gender (grouped bars)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.38
    for i, (g, name) in enumerate(GENDER_NAMES.items()):
        offset = (i - 0.5) * width
        ax.bar(x_bins + offset, bin_counts.get(g, 0), width=width, color=GENDER_COLORS[g],
               edgecolor=SURFACE, linewidth=1.5, label=name)
    _style_axes(ax, "Distribuição por classe de idade e género", "Classe de idade (anos)")
    ax.set_xticks(x_bins, bin_names)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY)
    _save(fig, out_dir, "age_bin_gender_hist.png")

    # 5) Gender distribution
    fig, ax = plt.subplots(figsize=(5, 4.5))
    gender_counts = labels_df["gender"].value_counts().reindex(GENDER_NAMES.keys(), fill_value=0)
    bars = ax.bar(list(GENDER_NAMES.values()), gender_counts.values, width=0.6,
                  color=[GENDER_COLORS[g] for g in GENDER_NAMES])
    pct = gender_counts / gender_counts.sum() * 100
    ax.bar_label(bars, labels=[f"{c} ({p:.1f}%)" for c, p in zip(gender_counts, pct)],
                 padding=3, color=INK_SECONDARY, fontsize=9)
    _style_axes(ax, "Distribuição por género", "")
    _save(fig, out_dir, "gender_dist.png")


def prepare_utkface():
    images_path = get_path()

    # Get Labels from image names
    labels_df = get_labels(images_path)

    # Data statistics
    data_statistics(labels_df)

    # Wrong Labels Check
    # TODO

    # DATA SPLIT
    data_split(labels_df)

if __name__ == "__main__":
    prepare_utkface()
