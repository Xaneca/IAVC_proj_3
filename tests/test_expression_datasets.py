import csv
import json
import shutil
import tempfile
import unittest
from zipfile import ZipFile
from collections import Counter
from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from scripts.data import prepare_expression_datasets as module


class PreparationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.roots = {ds: self.base / ds for ds in module.SPLITS}
        self.indexes = {ds: self.base / f"{ds}.csv" for ds in module.SPLITS}
        self.rows = {ds: [] for ds in module.SPLITS}
        self.colors = iter(range(256))
        self.classes = json.loads(
            (module.REPO / "configs/expression_classes.yaml").read_text(encoding="utf-8")
        )["classes"]
        for ds in module.SPLITS:
            self.roots[ds].mkdir()
        for label in self.classes:
            for i in range(10):
                self.add("affectnet", f"{label}-train-{i}.png", label, "source-train")
            self.add("affectnet", f"{label}-official.png", label, "source-validation")
            self.add("btfer7", f"{label}.png", label, "collection")
        self.review_path = self.base / "review.json"
        fields = ("url", "version", "provenance", "license", "restrictions",
                  "authorization_evidence", "label_evidence", "split_evidence")
        self.review = {ds: dict.fromkeys(fields, "synthetic fixture") for ds in module.SPLITS}
        for review in self.review.values():
            review["labels"] = {**{label: label for label in self.classes}, "contempt": None}
        self.review["affectnet"]["splits"] = {"source-train": "train", "source-validation": "test"}
        self.review["btfer7"]["splits"] = {"collection": "test"}
        self.output = self.base / "output"

    def add(self, dataset, name, label, split, source=None, reencode=False):
        path = self.roots[dataset] / name
        self.rows[dataset].append((name, label, split))
        if source is None:
            Image.new("RGB", (4, 6), (next(self.colors), 20, 40)).save(path)
        elif reencode:
            info = PngInfo()
            info.add_text("comment", "Re-encoded fixture")
            with Image.open(source) as image:
                image.save(path, pnginfo=info)
        else:
            shutil.copyfile(source, path)
        return path

    def run_build(self):
        for ds, path in self.indexes.items():
            module.write_csv(path, ("path", "source_label", "source_split"), self.rows[ds])
        self.review_path.write_text(json.dumps(self.review), encoding="utf-8")
        module.prepare(self.roots, self.indexes, self.review_path, self.output)

    def test_outputs_preserve_evaluation_and_prevent_leakage(self):
        official = self.roots["affectnet"] / "happy-official.png"
        duplicate = self.add("affectnet", "dup-official-as-train.png", "happy", "source-train", official)
        self.add("btfer7", "dup-official-reencoded.png", "anger", "collection", official, reencode=True)
        self.add("affectnet", "z-within.png", "neutral", "source-train",
                 self.roots["affectnet"] / "neutral-train-0.png")
        self.add("affectnet", "excluded.png", "contempt", "source-train")
        contradiction = self.add("affectnet", "contradiction.png", "anger", "source-train")
        self.add("btfer7", "wrong_label.png", "sad", "collection", contradiction)
        original = duplicate.read_bytes()
        self.run_build()
        self.assertEqual(duplicate.read_bytes(), original)

        manifests = self.output / "data/splits"
        audit = json.loads((manifests / "expression_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["excluded_labels"], 1)
        conflicts = {row["path"]: row for row in audit["conflicts"]}
        self.assertTrue(conflicts[official.name]["equal_file_bytes"])
        self.assertTrue(conflicts[official.name]["across_source_splits"])
        self.assertEqual(conflicts[official.name]["decision"], "kept")
        self.assertTrue(conflicts["z-within.png"]["within_source_split"])
        reencoded = conflicts["dup-official-reencoded.png"]
        self.assertTrue(reencoded["across_datasets"])
        self.assertFalse(reencoded["equal_file_bytes"])
        self.assertTrue(reencoded["label_conflict"])
        self.assertEqual(conflicts["wrong_label.png"]["decision"], "conflicting_labels")

        file_hashes = set()
        pixel_hashes = set()
        for ds in module.SPLITS:
            with (manifests / f"{ds}.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            counts = Counter(row["split"] for row in rows)
            self.assertEqual(counts, {"train": 63, "val": 7, "test": 7}
                             if ds == "affectnet" else {"test": 7})
            for row in rows:
                self.assertIn(row["label"], self.classes)
                if ds == "affectnet" and row["split"] == "test":
                    self.assertTrue(row["path"].endswith("-official.png"))
                file_hash, pixel_hash = module.hashes(self.roots[ds] / row["path"])
                self.assertNotIn(file_hash, file_hashes)
                self.assertNotIn(pixel_hash, pixel_hashes)
                file_hashes.add(file_hash)
                pixel_hashes.add(pixel_hash)
        self.assertEqual(len(list((self.output / "docs/figs").glob("*.png"))), 4)
        with (manifests / "expression_class_mapping.csv").open(encoding="utf-8", newline="") as stream:
            mapping = {(row["dataset"], row["source_label"]): row["label"]
                       for row in csv.DictReader(stream)}
        self.assertEqual(mapping["affectnet", "contempt"], "")
        self.assertEqual(mapping["btfer7", "anger"], "anger")

        before = {p: p.read_bytes() for p in self.output.rglob("*") if p.is_file()}
        for rows in self.rows.values():
            rows.reverse()
        self.run_build()
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_unmapped_labels_and_unindexed_images_fail(self):
        self.add("btfer7", "unknown.png", "unknown", "collection")
        with self.assertRaisesRegex(ValueError, "por rever"):
            self.run_build()
        self.rows["btfer7"].pop()
        with self.assertRaisesRegex(ValueError, "sem anotação"):
            self.run_build()
        self.assertFalse(self.output.exists())

    def test_preliminary_reports_duplicates_and_keeps_source_rows(self):
        archives = {ds: self.base / f"{ds}.zip" for ds in module.SPLITS}
        image = (self.roots["affectnet"] / "happy-official.png").read_bytes()
        with ZipFile(archives["affectnet"], "w") as archive:
            archive.writestr("AffectNetCustom/train/0/a.png", image)
            archive.writestr("AffectNetCustom/train/0/b.png", image)
            archive.writestr("AffectNetCustom/test/7/c.png", image)
        with ZipFile(archives["btfer7"], "w") as archive:
            archive.writestr("Batch_Ready 7/Angry/d.png", image)
        module.preliminary(archives, self.output)
        manifests = self.output / "data/splits"
        audit = json.loads((manifests / "expression_preliminar_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["image_counts"], {"affectnet": 3, "btfer7": 1})
        for matches in audit["conflicts"].values():
            self.assertEqual(len(matches), 1)
            match = matches[0]
            for flag in ("within_split", "across_splits", "across_datasets", "source_label_conflict"):
                self.assertTrue(match[flag])
            self.assertEqual(len(match["members"]), 4)
        with (manifests / "affectnet_preliminar.csv").open(encoding="utf-8", newline="") as stream:
            self.assertEqual([(row["label"], row["split"]) for row in csv.DictReader(stream)],
                             [("7", "test"), ("0", "train"), ("0", "train")])

    def test_clean_archives_preserve_splits_and_exclude_all_conflicting_labels(self):
        archives = {ds: self.base / f"{ds}.zip" for ds in module.SPLITS}

        def distinct(label):
            return (self.roots["affectnet"] / f"{label}-official.png").read_bytes()

        with ZipFile(archives["affectnet"], "w") as archive:
            archive.writestr("AffectNetCustom/test/0/keep.png", distinct("happy"))
            archive.writestr("AffectNetCustom/train/0/copy.png", distinct("happy"))
            archive.writestr("AffectNetCustom/train/0/keep.png", distinct("sad"))
            archive.writestr("AffectNetCustom/val/7/keep.png", distinct("anger"))
            archive.writestr("AffectNetCustom/val/3/keep.png", distinct("neutral"))
            archive.writestr("AffectNetCustom/train/0/conflict.png", distinct("fear"))
            archive.writestr("AffectNetCustom/test/1/conflict.png", distinct("fear"))
        with ZipFile(archives["btfer7"], "w") as archive:
            archive.writestr("Batch_Ready 7/Angry/keep.png", distinct("disgust"))
        originals = {ds: path.read_bytes() for ds, path in archives.items()}

        module.preliminary(archives, self.output, clean=True)

        manifests = self.output / "data/splits"
        self.assertFalse((manifests / "affectnet.csv").exists())
        with (manifests / "affectnet_auditado.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([(row["path"].rsplit("/", 1)[-1], row["label"], row["split"])
                          for row in rows], [("keep.png", "neutral", "test"),
                                            ("keep.png", "neutral", "train"),
                                            ("keep.png", "surprise", "val")])
        audit = json.loads((manifests / "expression_auditado_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["excluded_counts"]["affectnet"],
                         {"exact_duplicate": 1, "conflicting_labels": 2, "contempt": 1})
        with (manifests / "btfer7_auditado.csv").open(encoding="utf-8", newline="") as stream:
            self.assertEqual(list(csv.DictReader(stream)), [
                {"path": "Batch_Ready 7/Angry/keep.png", "label": "anger", "split": "unsplit"}
            ])
        self.assertEqual(originals, {ds: path.read_bytes() for ds, path in archives.items()})


if __name__ == "__main__":
    unittest.main()
