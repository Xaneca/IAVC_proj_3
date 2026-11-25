"""Treino multitarefa: idade + género (UTKFace) e expressão (AffectNet).

    crop → stem + stages congelados → stages afinados (partilhados pelas 3 cabeças)
             ├─ MLP idade     → regressão (idade normalizada), L1
             ├─ MLP género    → 2 logits, CrossEntropy
             └─ MLP expressão → 7 logits, CrossEntropy com pesos por classe

Quais os stages afinados e de que stage saem as features vem do config.yaml
(train.trainable_stages, train.feature_stage), p. ex. [3]/3 ou [4]/4.

Cada passo junta um batch do UTKFace e um do AffectNet num único forward.
As losses de idade e género só usam a parte UTKFace e a de expressão só a
parte AffectNet. A loss total é a soma ponderada (train.loss_weights do
config.yaml), com um único backward: os stages afinados recebem a soma dos
gradientes das três cabeças e cada MLP só o da sua loss.

Early stopping por componente:
  - backbone (stages afinados): pela loss conjunta de validação. Enquanto
    treina, as cabeças não param (as features ainda mudam). Quando para, o
    modelo inteiro volta à melhor época do backbone, que fica congelado.
  - cada MLP: pela sua loss de validação, só depois de o backbone parar (ou
    logo desde o início, sem stages afinados). Quando para, volta aos seus
    melhores pesos e congela; as outras continuam.
O treino acaba quando todas as cabeças pararam (ou ao fim de train.epochs).
O checkpoint junta os melhores pesos de cada componente.

Correr: python -m train.train_multitask [--epochs N] [--max-steps N]
Saída:  train.output (melhor época) e results/train_multitask/<nome>_history.csv
"""
import argparse
import copy
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from app.backbone import preprocess
from app.heads import EXPRESSIONS, MultiTaskModel, age_to_class
from app.iresnet import load_arcface

REPO = Path(__file__).resolve().parents[1]
SPLITS = REPO / "data/splits"
RESULTS = REPO / "results/train_multitask"
HEADS = ("age", "gender", "expression")
NAMES = {"backbone": "backbone", "age": "idade", "gender": "género", "expression": "expressão"}


class FaceDataset(Dataset):
    """Crops 112×112 de um split e as suas labels.

    O crop de cada linha é <crops_dir>/<path do CSV> com extensão .jpg (o mesmo
    layout de scripts/data/preprocess_faces.py). Linhas sem crop nessa pasta
    (p. ex. imagens eliminadas por outro pré-processamento) são ignoradas.
    """

    def __init__(self, csv: Path, crops_dir: Path, split: str, label_columns: list[str], flip: bool):
        df = pd.read_csv(csv)
        df = df[df["split"] == split].reset_index(drop=True)
        if "label" in label_columns:
            df["label"] = df["label"].map({name: i for i, name in enumerate(EXPRESSIONS)})
            df = df[df["label"].notna()].reset_index(drop=True)   # retira classes fora das 7
        paths = [(crops_dir / rel).with_suffix(".jpg") for rel in df["path"]]
        exists = np.array([p.is_file() for p in paths], dtype=bool)
        if not exists.any():
            raise FileNotFoundError(f"Nenhum crop do split {split} de {csv.name} em {crops_dir}")
        if not exists.all():
            print(f"{csv.stem} {split}: {(~exists).sum()} sem crop em "
                  f"{crops_dir.relative_to(REPO)} (ignoradas)")
        df = df[exists].reset_index(drop=True)
        self.paths = [p for p, ok in zip(paths, exists) if ok]
        self.labels = df[label_columns].to_numpy(dtype=np.float32)
        self.flip = flip

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        image = cv2.imread(str(self.paths[i]))
        if image is None:
            raise OSError(f"Não abre: {self.paths[i]}")
        if self.flip and np.random.rand() < 0.5:
            image = image[:, ::-1]
        return preprocess([image])[0], torch.from_numpy(self.labels[i])


def loader(dataset: Dataset, batch_size: int, train: bool, workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=train,
                      num_workers=workers, pin_memory=True, persistent_workers=workers > 0)


def encoder_stages(model: MultiTaskModel):
    """(nº do stage, módulo) para cada stage copiado no encoder."""
    return zip(range(model.first_stage, model.feature_stage + 1), model.encoder)


def set_trainable(model: MultiTaskModel) -> None:
    for stage, module in encoder_stages(model):
        module.requires_grad_(stage in model.trainable_stages)


def train_mode(model: MultiTaskModel, stopped: list[nn.Module]) -> None:
    """Cabeças e stages afinados em train; congelados e parados em eval (BN fixa)."""
    model.train()
    for stage, module in encoder_stages(model):
        module.train(stage in model.trainable_stages)
    for module in stopped:
        module.eval()


class EarlyStop:
    """Melhor valor e melhores pesos de um componente (backbone ou uma cabeça)."""

    def __init__(self, module: nn.Module, patience: int):
        self.module, self.patience = module, patience
        self.best, self.bad, self.state, self.stopped = float("inf"), 0, None, False

    def snapshot(self, value: float) -> None:
        self.best, self.bad = value, 0
        self.state = copy.deepcopy(self.module.state_dict())

    def step(self, value: float) -> bool:
        """Regista a época; devolve True se melhorou. Para ao fim de patience épocas sem melhorar."""
        if value < self.best:
            self.snapshot(value)
            return True
        self.bad += 1
        if self.bad >= self.patience:
            self.stop()
        return False

    def restore(self) -> None:
        self.module.load_state_dict(self.state)

    def stop(self) -> None:
        self.restore()
        self.module.requires_grad_(False)
        self.stopped = True


def losses(pred: dict, n_utk: int, age_z, gender, expression, class_weights) -> dict:
    return {
        "age": F.l1_loss(pred["age_z"][:n_utk].float(), age_z),
        "gender": F.cross_entropy(pred["gender"][:n_utk].float(), gender),
        "expression": F.cross_entropy(pred["expression"][n_utk:].float(), expression,
                                      weight=class_weights),
    }


@torch.no_grad()
def evaluate(model, arcface, utk_loader, aff_loader, device, amp, class_weights, weights) -> dict:
    model.eval()
    age_true, age_pred, g_true, g_pred, e_true, e_pred, e_loss = [], [], [], [], [], [], []
    for x, y in utk_loader:
        with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
            pred = model(arcface.trunk(x.to(device), model.first_stage - 1))
        age_true.append(y[:, 0].numpy())
        age_pred.append(model.age_years(pred["age_z"].float()).cpu().numpy())
        g_true.append(y[:, 1].long().numpy())
        g_pred.append(pred["gender"].float().cpu().numpy())
    for x, y in aff_loader:
        with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
            pred = model(arcface.trunk(x.to(device), model.first_stage - 1))
        logits = pred["expression"].float()
        target = y[:, 0].long().to(device)
        e_loss.append(F.cross_entropy(logits, target, weight=class_weights, reduction="sum").item())
        e_true.append(y[:, 0].long().numpy())
        e_pred.append(logits.argmax(1).cpu().numpy())

    age_true, age_pred = np.concatenate(age_true), np.concatenate(age_pred)
    g_true, g_logits = np.concatenate(g_true), np.concatenate(g_pred)
    e_true, e_pred = np.concatenate(e_true), np.concatenate(e_pred)
    age_std = model.age_std.item()
    g_loss = F.cross_entropy(torch.from_numpy(g_logits), torch.from_numpy(g_true)).item()
    loss = {"age": float(np.abs(age_pred - age_true).mean()) / age_std,
            "gender": g_loss,
            "expression": sum(e_loss) / len(e_true)}
    return {
        "loss": sum(weights[k] * v for k, v in loss.items()),
        **{f"loss_{k}": v for k, v in loss.items()},
        "age_mae": float(np.abs(age_pred - age_true).mean()),
        "age_bin_acc": float((age_to_class(age_pred) == age_to_class(age_true)).mean()),
        "gender_acc": float((g_logits.argmax(1) == g_true).mean()),
        "expression_acc": float((e_pred == e_true).mean()),
        "expression_f1": float(f1_score(e_true, e_pred, average="macro")),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO / "config.yaml"))
    parser.add_argument("--epochs", type=int, help="sobrepõe train.epochs")
    parser.add_argument("--max-steps", type=int, help="limita os passos por época (teste rápido)")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    tc = cfg["train"]
    epochs = args.epochs or tc["epochs"]
    weights = tc["loss_weights"]
    torch.manual_seed(tc["seed"])
    np.random.seed(tc["seed"])

    device = tc["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA indisponível: a treinar em CPU (lento).")
        device = "cpu"
    amp = device == "cuda"   # bfloat16 na GPU

    # Dados
    bs, workers = tc["batch_size"], tc["num_workers"]
    utk_csv, aff_csv = SPLITS / "utkface.csv", SPLITS / "affectnet.csv"
    utk_dir, aff_dir = REPO / tc["crops"]["utkface"], REPO / tc["crops"]["affectnet"]
    print(f"Crops: UTKFace {tc['crops']['utkface']} | AffectNet {tc['crops']['affectnet']}")
    utk_train = FaceDataset(utk_csv, utk_dir, "train", ["age", "gender"], flip=True)
    aff_train = FaceDataset(aff_csv, aff_dir, "train", ["label"], flip=True)
    utk_val = loader(FaceDataset(utk_csv, utk_dir, "val", ["age", "gender"], flip=False),
                     bs * 2, False, workers)
    aff_val = loader(FaceDataset(aff_csv, aff_dir, "val", ["label"], flip=False),
                     bs * 2, False, workers)
    utk_loader = loader(utk_train, bs, True, workers)
    aff_loader = loader(aff_train, bs, True, workers)
    steps = max(len(utk_loader), len(aff_loader))
    if args.max_steps:
        steps = min(steps, args.max_steps)
    print(f"UTKFace {len(utk_train)} treino | AffectNet {len(aff_train)} treino | "
          f"{steps} passos/época | {device}")

    # Modelo
    arcface = load_arcface(REPO / cfg["backbone"]["weights"], device)
    model = MultiTaskModel(arcface, tc["feature_stage"], tc["trainable_stages"]).to(device)
    set_trainable(model)
    print(f"Features do stage {model.feature_stage} | stages afinados {model.trainable_stages} | "
          f"congelados: stem + stages 1-{model.first_stage - 1}")
    ages = utk_train.labels[:, 0]
    age_mean, age_std = float(ages.mean()), float(ages.std())
    model.age_mean.fill_(age_mean)
    model.age_std.fill_(age_std)

    counts = np.bincount(aff_train.labels[:, 0].astype(int), minlength=len(EXPRESSIONS))
    class_weights = torch.tensor(counts.sum() / (len(EXPRESSIONS) * counts),
                                 dtype=torch.float32, device=device)

    backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [p for m in (model.age, model.gender, model.expression) for p in m.parameters()]
    groups = [{"params": head_params, "lr": tc["lr_heads"]}]
    if backbone_params:   # vazio quando só as MLP treinam
        groups.append({"params": backbone_params, "lr": tc["lr_backbone"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=tc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps)

    output = REPO / tc["output"]
    history_csv = RESULTS / f"{output.stem}_history.csv"
    RESULTS.mkdir(parents=True, exist_ok=True)
    history = []
    patience = tc["patience"]
    backbone = EarlyStop(model.encoder, patience) if backbone_params else None
    heads = {k: EarlyStop(getattr(model, k), patience) for k in HEADS}

    def save_best() -> None:
        """Checkpoint com os melhores pesos de cada componente."""
        ckpt = model.checkpoint()
        for name, tracker in [("encoder", backbone), *heads.items()]:
            if tracker is not None and tracker.state is not None:
                ckpt["state_dict"].update({f"{name}.{k}": v for k, v in tracker.state.items()})
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, output)

    def batches(loader_, it):
        """Próximo batch, recomeçando o loader quando acaba (o mais pequeno dá várias voltas)."""
        batch = next(it, None)
        if batch is None:
            it = iter(loader_)
            batch = next(it)
        return batch, it

    utk_iter, aff_iter = iter(utk_loader), iter(aff_loader)
    for epoch in range(1, epochs + 1):
        stopped = [t.module for t in [backbone, *heads.values()] if t is not None and t.stopped]
        train_mode(model, stopped)
        start, totals = time.time(), {"loss": 0.0, "age": 0.0, "gender": 0.0, "expression": 0.0}
        for _ in range(steps):
            (xu, yu), utk_iter = batches(utk_loader, utk_iter)
            (xa, ya), aff_iter = batches(aff_loader, aff_iter)
            x = torch.cat([xu, xa]).to(device, non_blocking=True)
            age_z = ((yu[:, 0] - age_mean) / age_std).to(device)
            gender = yu[:, 1].long().to(device)
            expression = ya[:, 0].long().to(device)

            with torch.no_grad():
                trunk = arcface.trunk(x, model.first_stage - 1)   # stages congelados
            with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
                pred = model(trunk)
            parts = losses(pred, len(xu), age_z, gender, expression, class_weights)
            loss = sum(weights[k] * v for k, v in parts.items())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            totals["loss"] += loss.item()
            for k, v in parts.items():
                totals[k] += v.item()

        val = evaluate(model, arcface, utk_val, aff_val, device, amp, class_weights, weights)
        row = {"epoch": epoch, **{f"train_{k}": v / steps for k, v in totals.items()},
               **{f"val_{k}": v for k, v in val.items()}, "seconds": time.time() - start}
        history.append(row)

        backbone_training = backbone is not None and not backbone.stopped
        improved, just_stopped = [], []
        if backbone_training:
            # Só o backbone tem early stop; as cabeças acompanham a melhor época dele.
            if backbone.step(val["loss"]):
                improved.append("backbone")
                for k, t in heads.items():
                    t.snapshot(val[f"loss_{k}"])
            if backbone.stopped:
                just_stopped.append("backbone")
                for t in heads.values():   # volta à época do melhor backbone
                    t.restore()
                    t.bad = 0
        else:
            for k, t in heads.items():
                if t.stopped:
                    continue
                if t.step(val[f"loss_{k}"]):
                    improved.append(k)
                if t.stopped:
                    just_stopped.append(k)
        if improved:
            save_best()
        row["improved"] = " ".join(improved)
        row["stopped"] = " ".join(just_stopped)
        pd.DataFrame(history).to_csv(history_csv, index=False)

        def parts_str(get):
            return " ".join(f"{NAMES[k]} {get(k):.3f}" for k in HEADS)

        state = []
        for name, t in [("backbone", backbone), *heads.items()]:
            if t is None:
                continue
            if name in improved:
                mark = "*"
            elif name in just_stopped:
                mark = "parou"
            elif t.stopped:
                mark = "-"
            else:
                mark = f"{t.bad}/{patience}"
            state.append(f"{NAMES[name]} {mark}")
        print(f"época {epoch:2d} | {row['seconds']:.0f}s | " + " | ".join(state))
        print(f"    treino  total {row['train_loss']:.3f} | "
              + parts_str(lambda k: row[f"train_{k}"]))
        print(f"    val     total {val['loss']:.3f} | "
              + parts_str(lambda k: val[f"loss_{k}"]))
        print(f"    MAE {val['age_mae']:.2f} anos, bins {val['age_bin_acc']:.1%} | "
              f"género {val['gender_acc']:.1%} | expressão acc {val['expression_acc']:.1%} "
              f"F1 {val['expression_f1']:.3f}")
        for name in just_stopped:
            print(f"    Early stopping {NAMES[name]}: {patience} épocas sem melhorar"
                  + (" (modelo volta à melhor época do backbone; as MLP continuam)"
                     if name == "backbone" else ""))
        if all(t.stopped for t in heads.values()):
            print("Todas as cabeças pararam.")
            break

    summary = [f"backbone {backbone.best:.3f}"] if backbone is not None else []
    summary += [f"{NAMES[k]} {t.best:.3f}" for k, t in heads.items()]
    print(f"Melhores losses de validação: {', '.join(summary)} -> {output.relative_to(REPO)}")


if __name__ == "__main__":
    main()
