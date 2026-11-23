"""Inventory local releases and build manifests from reviewed source annotations."""

import argparse
import csv
import hashlib
import json
import struct
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile

from PIL import Image, ImageDraw, ImageFont, ImageOps


REPO = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLITS = {"affectnet": ("train", "val", "test"), "btfer7": ("test",)}
PRIORITY = {("affectnet", "test"): 0, ("btfer7", "test"): 1,
            ("affectnet", "train"): 2}


@dataclass(frozen=True)
class Sample:
    dataset: str
    path: str
    source_label: str
    source_split: str
    label: str | None
    split: str
    file_hash: str
    pixel_hash: str


def inventory(root):
    if not root.is_dir():
        raise ValueError(f"Diretório de origem indisponível: {root}")
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Ligação na origem: {path}")
        if path.is_file():
            yield path.relative_to(root).as_posix(), path.suffix.lower() in IMAGE_SUFFIXES


def pixel_hash(stream):
    with Image.open(stream) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("imagem com múltiplos frames")
        pixels = ImageOps.exif_transpose(image).convert("RGBA")
        digest = hashlib.sha256(struct.pack(">II", *pixels.size))
        digest.update(pixels.tobytes())
    return digest.hexdigest()


def hashes(path):
    try:
        with path.open("rb") as stream:
            file_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            stream.seek(0)
            return file_hash, pixel_hash(stream)
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise ValueError(f"Falha ao ler {path}: {error}") from error


def read_source(dataset, root, index, review, classes):
    for field in ("url", "version", "provenance", "license", "restrictions",
                  "authorization_evidence", "label_evidence", "split_evidence"):
        if not isinstance(review.get(field), str) or not review[field].strip():
            raise ValueError(f"{dataset}: falta preencher {field}")
    labels = review["labels"]
    splits = review["splits"]
    if set(labels.values()) - (set(classes) | {None}):
        raise ValueError(f"{dataset}: mapeamento para classes desconhecidas")
    expected = {"train", "test"} if dataset == "affectnet" else {"test"}
    if set(splits.values()) != expected:
        raise ValueError(f"{dataset}: partições de destino esperadas: {sorted(expected)}")
    if dataset == "affectnet" and list(splits.values()).count("test") != 1:
        raise ValueError("AffectNet: indicar uma única origem para a validação oficial")

    remaining = {path for path, is_image in inventory(root) if is_image}
    samples = []
    with index.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        if rows.fieldnames != ["path", "source_label", "source_split"]:
            raise ValueError(f"Cabeçalho inválido em {index}")
        for row in rows:
            path, source_label, source_split = (
                row[key] for key in ("path", "source_label", "source_split")
            )
            if None in row or path not in remaining:
                raise ValueError(f"{dataset}: linha inválida, imagem repetida ou ausente: {path}")
            if source_label not in labels or source_split not in splits:
                raise ValueError(f"{dataset}: classe ou split por rever: {path}")
            remaining.remove(path)
            samples.append(Sample(dataset, path, source_label, source_split,
                                  labels[source_label], splits[source_split],
                                  *hashes(root / path)))
    if remaining:
        raise ValueError(f"{dataset}: {len(remaining)} imagens sem anotação, por exemplo {min(remaining)}")
    return samples


def deduplicate(samples):
    groups = defaultdict(list)
    for sample in samples:
        groups[sample.pixel_hash].append(sample)
    retained = []
    conflicts = []
    for digest, group in sorted(groups.items()):
        group.sort(key=lambda s: (PRIORITY[s.dataset, s.split], s.path))
        first = group[0]
        preferred_split = (first.dataset, first.split)
        labels = {s.label for s in group if s.label is not None}
        preferred = [s for s in group if (s.dataset, s.split) == preferred_split
                     and s.label is not None]
        # External annotations cannot override an unambiguous official label.
        decision_labels = {s.label for s in preferred} if preferred_split == (
            "affectnet", "test"
        ) else labels
        winner = preferred[0] if preferred and len(decision_labels) == 1 else None
        if winner:
            retained.append(winner)
        if len(group) < 2:
            continue
        sources = Counter((s.dataset, s.source_split) for s in group)
        file_counts = Counter(s.file_hash for s in group)
        for sample in group:
            conflicts.append({
                "dataset": sample.dataset, "path": sample.path,
                "source_label": sample.source_label, "source_split": sample.source_split,
                "label": sample.label, "target_before_validation": sample.split,
                "sha256": sample.file_hash, "pixel_sha256": digest,
                "equal_file_bytes": file_counts[sample.file_hash] > 1,
                "within_source_split": sources[sample.dataset, sample.source_split] > 1,
                "across_source_splits": any(
                    ds == sample.dataset and split != sample.source_split for ds, split in sources
                ),
                "across_datasets": len({ds for ds, _ in sources}) > 1,
                "label_conflict": len(labels) > 1,
                "decision": ("kept" if sample == winner else
                             "excluded_label" if sample.label is None else
                             "conflicting_labels" if len(labels) > 1 else
                             "duplicate" if winner else "reserved_for_evaluation"),
            })
    return retained, conflicts


def assign_splits(samples, classes, seed, percent):
    rows = defaultdict(list)
    training = defaultdict(list)
    for sample in samples:
        if sample.split == "train":
            training[sample.label].append(sample)
        else:
            rows[sample.dataset].append((sample.path, sample.label, sample.split))

    sizes = {label: len(training[label]) for label in classes}
    target = (sum(sizes.values()) * percent + 50) // 100
    if min(sizes.values()) < 2 or not len(classes) <= target <= sum(sizes.values()) - len(classes):
        raise ValueError("AffectNet: treino insuficiente para validação estratificada com todas as classes")
    quotas = {label: min(size - 1, max(1, size * percent // 100))
              for label, size in sizes.items()}
    while sum(quotas.values()) < target:
        label = max((c for c in classes if quotas[c] < sizes[c] - 1),
                    key=lambda c: sizes[c] * percent - quotas[c] * 100)
        quotas[label] += 1
    while sum(quotas.values()) > target:
        label = max((c for c in classes if quotas[c] > 1),
                    key=lambda c: quotas[c] * 100 - sizes[c] * percent)
        quotas[label] -= 1
    for label in classes:
        ranked = sorted(training[label],
                        key=lambda s: (hashlib.sha256(f"{seed}:{s.path}".encode()).digest(), s.path))
        for index, sample in enumerate(ranked):
            split = "val" if index < quotas[label] else "train"
            rows["affectnet"].append((sample.path, sample.label, split))
    for dataset, splits in SPLITS.items():
        observed = {split for _, _, split in rows[dataset]}
        missing = set(splits) - observed
        if missing:
            raise ValueError(f"{dataset}: partições vazias após a deduplicação: {sorted(missing)}")
    return rows


def plot_distribution(dataset, rows, classes, dest, splits=None, suffix=""):
    counts = Counter((label, split) for _, label, split in rows)
    splits = splits or SPLITS[dataset]
    font = ImageFont.load_default(size=16)
    title_font = ImageFont.load_default(size=22)
    panel_width, bar_width = 440, 260
    for percentage in (False, True):
        axis_bottom = 96 + 32 * len(classes)
        figure = Image.new("RGB", (panel_width * len(splits), axis_bottom + 45), "white")
        draw = ImageDraw.Draw(figure)
        unit = "Percentagem (%)" if percentage else "Imagens"
        title = f"{dataset} (origem)" if suffix else dataset
        draw.text((20, 15), f"{title} · {unit}", fill="#162435", font=title_font)
        limit = 100 if percentage else max(counts.values())
        for index, split in enumerate(splits):
            left = index * panel_width + 20
            axis = left + 85
            total = sum(counts[label, split] for label in classes)
            draw.text((left, 58), f"{split}  (n={total})", fill="#162435", font=font)
            for tick in range(5):
                x = axis + bar_width * tick // 4
                draw.line((x, 90, x, axis_bottom), fill="#d9e1e9")
                draw.text((x, axis_bottom + 12), f"{limit * tick / 4:g}", anchor="mt",
                          fill="#354454", font=font)
            for row, label in enumerate(classes):
                y = 97 + 32 * row
                count = counts[label, split]
                value = 100 * count / total if percentage else count
                end = axis + round(bar_width * value / limit)
                draw.text((left, y), label, fill="#162435", font=font)
                if value:
                    draw.rectangle((axis, y, max(axis, end - 1), y + 18), fill="#28638e")
                caption = f"{value:.1f}%" if percentage else str(count)
                draw.text((end + 5, y), caption, fill="#162435", font=font)
        metric = "percentagem" if percentage else "contagem"
        figure.save(dest / f"{dataset}{suffix}_{metric}.png")


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)


def preliminary(archives, output, clean=False):
    """Audit the two inspected ZIP layouts without assigning evaluation roles."""
    rows = defaultdict(list)
    groups = {"sha256": defaultdict(list), "pixel_sha256": defaultdict(list)}
    decode_errors = []
    archive_hashes = {}
    for dataset, archive in archives.items():
        with archive.open("rb") as stream:
            archive_hashes[dataset] = hashlib.file_digest(stream, "sha256").hexdigest()
        with ZipFile(archive) as source:
            seen = set()
            for entry in sorted(source.infolist(), key=lambda item: item.filename):
                if entry.is_dir() or Path(entry.filename).suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                path = entry.filename
                parts = path.split("/")
                if path in seen or any(part in {"", ".", ".."} or "\\" in part for part in parts):
                    raise ValueError(f"{dataset}: caminho inválido ou repetido: {path}")
                seen.add(path)
                if dataset == "affectnet" and len(parts) == 4 and parts[0] == "AffectNetCustom":
                    _, split, label, _ = parts
                    if split not in {"train", "val", "test"} or label not in "01234567" or len(label) != 1:
                        raise ValueError(f"AffectNet: partição ou código inesperado: {path}")
                elif dataset == "btfer7" and len(parts) == 3 and parts[0] == "Batch_Ready 7":
                    _, label, _ = parts
                    split = "unsplit"
                else:
                    raise ValueError(f"{dataset}: estrutura inesperada: {path}")
                rows[dataset].append((path, label, split))
                member = (dataset, path, label, split)
                with source.open(entry) as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    groups["sha256"][digest].append(member)
                    stream.seek(0)
                    try:
                        digest = pixel_hash(stream)
                    except (OSError, ValueError, Image.DecompressionBombError) as error:
                        decode_errors.append({"dataset": dataset, "path": path, "error": str(error)})
                    else:
                        groups["pixel_sha256"][digest].append(member)
        if not rows[dataset]:
            raise ValueError(f"{dataset}: ZIP sem imagens")

    conflicts = {}
    for method, by_hash in groups.items():
        matches = []
        for digest, members in sorted(by_hash.items()):
            if len(members) < 2:
                continue
            by_dataset = defaultdict(set)
            labels = defaultdict(set)
            locations = Counter()
            for dataset, _, label, split in members:
                by_dataset[dataset].add(split)
                labels[dataset].add(label)
                locations[dataset, split] += 1
            matches.append({
                "hash": digest,
                "within_split": any(count > 1 for count in locations.values()),
                "across_splits": any(len(splits) > 1 for splits in by_dataset.values()),
                "across_datasets": len(by_dataset) > 1,
                "source_label_conflict": any(len(values) > 1 for values in labels.values()),
                "members": [dict(zip(("dataset", "path", "label", "split"), member))
                            for member in members],
            })
        conflicts[method] = matches

    manifests = output / "data/splits"
    figures = output / "docs/figs/expression-provisoria"
    audit = {"archive_sha256": archive_hashes, "image_counts": {ds: len(entries)
             for ds, entries in rows.items()}, "pixel_decode_errors": decode_errors,
             "conflicts": conflicts}
    suffix = "_preliminar"
    if clean:
        if decode_errors:
            raise ValueError("Há imagens por descodificar; a deduplicação por píxeis está incompleta")
        suffix = "_auditado"
        config = json.loads((REPO / "configs/expression_classes.yaml").read_text(encoding="utf-8"))
        affectnet_labels = config["affectnet_labels"]
        priority = {"test": 0, "val": 1, "unsplit": 2, "train": 3}
        excluded = {}
        # With all images decoded, equal file bytes also produce equal pixel hashes.
        for group in conflicts["pixel_sha256"]:
            members = group["members"]
            if group["source_label_conflict"] or group["across_datasets"]:
                reason = ("conflicting_labels" if group["source_label_conflict"]
                          else "cross_dataset_duplicate")
                drop = members
            else:
                reason = "exact_duplicate"
                drop = sorted(members, key=lambda m: (priority[m["split"]], m["path"]))[1:]
            for member in drop:
                excluded[member["dataset"], member["path"]] = reason
        for path, label, _ in rows["affectnet"]:
            if affectnet_labels[label] == "contempt":
                excluded.setdefault(("affectnet", path), "contempt")

        audit["excluded"] = [
            {"dataset": ds, "path": path, "label": label, "split": split,
             "reason": excluded[ds, path]}
            for ds, entries in rows.items() for path, label, split in entries
            if (ds, path) in excluded
        ]
        audit["excluded_counts"] = {
            ds: dict(Counter(row["reason"] for row in audit["excluded"] if row["dataset"] == ds))
            for ds in rows
        }
        for ds, entries in rows.items():
            rows[ds] = [(path, label, split) for path, label, split in entries
                        if (ds, path) not in excluded]
            if {split for _, _, split in rows[ds]} != {split for _, _, split in entries}:
                raise ValueError(f"{ds}: partição vazia após filtragem")
        rows["affectnet"] = [(path, affectnet_labels[label], split)
                            for path, label, split in rows["affectnet"]]
        btfer_labels = {label.capitalize(): label for label in config["classes"] if label != "anger"}
        btfer_labels["Angry"] = "anger"
        if any(label not in btfer_labels for _, label, _ in rows["btfer7"]):
            raise ValueError("BTFER7: rótulo de origem desconhecido")
        rows["btfer7"] = [(path, btfer_labels[label], split) for path, label, split in rows["btfer7"]]
        audit["retained_counts"] = {
            ds: dict(Counter(split for _, _, split in entries)) for ds, entries in rows.items()
        }
        audit["label_status"] = {"affectnet": "mapping_provided_by_project", "btfer7": "canonical_labels"}
        audit["affectnet_labels"] = affectnet_labels
    manifests.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=manifests) as temp:
        stage = Path(temp)
        for dataset, entries in rows.items():
            write_csv(stage / f"{dataset}{suffix}.csv", ("path", "label", "split"), entries)
            classes = config["classes"] if clean else sorted({label for _, label, _ in entries})
            splits = tuple(split for split in ("train", "val", "test", "unsplit")
                           if any(source_split == split for _, _, source_split in entries))
            plot_distribution(dataset, entries, classes, stage, splits, suffix)
        (stage / f"expression{suffix}_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        for path in stage.iterdir():
            path.replace((figures if path.suffix == ".png" else manifests) / path.name)


def prepare(roots, indexes, review_path, output):
    config = json.loads((REPO / "configs/expression_classes.yaml").read_text(encoding="utf-8"))
    reviews = json.loads(review_path.read_text(encoding="utf-8-sig"))
    classes = config["classes"]
    output = output.resolve()
    destinations = [output / "data/splits", output / "docs/figs"]
    output_names = [
        *(destinations[0] / f"{dataset}.csv" for dataset in SPLITS),
        destinations[0] / "expression_audit.json",
        destinations[0] / "expression_class_mapping.csv",
        *(destinations[1] / f"{dataset}_{metric}.png"
          for dataset in SPLITS for metric in ("contagem", "percentagem")),
    ]
    for path in [output, *destinations, *output_names]:
        if any(path.resolve().is_relative_to(root) for root in roots.values()):
            raise ValueError("Os resultados têm de ficar fora dos datasets originais")
    inputs = {file.resolve() for file in [*indexes.values(), review_path]}
    if any(path.resolve() in inputs for path in output_names):
        raise ValueError("Um resultado substituiria um ficheiro de entrada")

    samples = [s for dataset in SPLITS
               for s in read_source(dataset, roots[dataset], indexes[dataset], reviews[dataset], classes)]
    retained, conflicts = deduplicate(samples)
    rows = assign_splits(retained, classes, config["seed"], config["validation_percent"])
    final_splits = {(dataset, path): split for dataset, entries in rows.items()
                    for path, _, split in entries}
    for conflict in conflicts:
        conflict["final_split"] = final_splits.get((conflict["dataset"], conflict["path"]))
    input_counts = Counter((s.dataset, s.source_split, s.source_label) for s in samples)
    audit = {
        "sources": reviews,
        "parameters": config,
        "input_counts": [{"dataset": ds, "source_split": split, "source_label": label, "count": n}
                         for (ds, split, label), n in sorted(input_counts.items())],
        "excluded_labels": sum(s.label is None for s in samples),
        "retained_counts": {dataset: {split: dict(Counter(
            label for _, label, row_split in rows[dataset] if row_split == split
        )) for split in splits} for dataset, splits in SPLITS.items()},
        "conflicts": conflicts,
    }
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as temp:
        stage = Path(temp)
        for dataset in SPLITS:
            write_csv(stage / f"{dataset}.csv", ("path", "label", "split"), sorted(rows[dataset]))
            plot_distribution(dataset, rows[dataset], classes, stage)
        write_csv(stage / "expression_class_mapping.csv", ("dataset", "source_label", "label"),
                  ((dataset, source_label, label)
                   for dataset in SPLITS
                   for source_label, label in sorted(reviews[dataset]["labels"].items())))
        (stage / "expression_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        for path in output_names:
            path.parent.mkdir(parents=True, exist_ok=True)
            (stage / path.name).replace(path)


def export_local_splits(roots):
    """Organiza os manifestos existentes ao lado das imagens extraídas."""
    exports = {}
    fields = ("path", "label", "split")
    for dataset, root in roots.items():
        manifest = REPO / "data/splits" / f"{dataset}.csv"
        with manifest.open(encoding="utf-8", newline="") as stream:
            rows = [tuple(row[field] for field in fields) for row in csv.DictReader(stream)]
        for path, _, split in rows:
            relative = Path(path)
            if relative.anchor or ".." in relative.parts or not (root / relative).is_file():
                raise ValueError(f"{dataset}: imagem em falta ou caminho inválido: {path}")
            if split not in SPLITS[dataset]:
                raise ValueError(f"{dataset}: split inesperado: {split}")
        exports[dataset] = rows
    for dataset, rows in exports.items():
        root = roots[dataset]
        write_csv(root / "all_labels.csv", fields, rows)
        for split in SPLITS[dataset]:
            write_csv(root / f"{split}_labels.csv", fields,
                      (row for row in rows if row[2] == split))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("root", type=Path)
    local = commands.add_parser("setup-local", help="Exportar os splits junto das imagens extraídas")
    local.add_argument("--affectnet-root", type=Path, default=REPO / "data/AffectNET")
    local.add_argument("--btfer7-root", type=Path, default=REPO / "data/BTFER7")
    preview = commands.add_parser("preliminary")
    for dataset in SPLITS:
        preview.add_argument(f"--{dataset}-zip", type=Path, required=True)
    preview.add_argument("--output", type=Path, default=REPO)
    preview.add_argument("--clean", action="store_true",
                         help="Gerar os manifestos deduplicados sem alterar partições ou rótulos")
    build = commands.add_parser("build")
    for dataset in SPLITS:
        build.add_argument(f"--{dataset}-root", type=Path, required=True)
        build.add_argument(f"--{dataset}-index", type=Path, required=True)
    build.add_argument("--review", type=Path, required=True)
    build.add_argument("--output", type=Path, default=REPO)
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            writer = csv.writer(sys.stdout, lineterminator="\n")
            writer.writerow(("path", "kind"))
            writer.writerows((path, "image" if is_image else "metadata_or_other")
                             for path, is_image in inventory(args.root.resolve()))
        elif args.command == "setup-local":
            export_local_splits({ds: getattr(args, f"{ds}_root").resolve() for ds in SPLITS})
        elif args.command == "preliminary":
            archives = {ds: getattr(args, f"{ds}_zip") for ds in SPLITS}
            preliminary(archives, args.output, clean=args.clean)
        else:
            roots = {ds: getattr(args, f"{ds}_root").resolve() for ds in SPLITS}
            indexes = {ds: getattr(args, f"{ds}_index") for ds in SPLITS}
            prepare(roots, indexes, args.review, args.output)
    except (OSError, TypeError, ValueError, KeyError, csv.Error, BadZipFile) as error:
        parser.exit(2, f"Erro: {error}\n")


if __name__ == "__main__":
    main()
