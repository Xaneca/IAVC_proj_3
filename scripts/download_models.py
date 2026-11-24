"""Obtém os pesos das distribuições oficiais InsightFace v0.7.

- det_10g.onnx   (buffalo_l):  detetor SCRFD-10G
- det_500m.onnx  (buffalo_sc): detetor SCRFD-500M (benchmark de deteção)
- w600k_r50.onnx (buffalo_l):  ArcFace iResNet-50 (WebFace600K), backbone partilhado

Correr: python -m scripts.download_models
"""

from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve
from zipfile import ZipFile


DETECTORS = {"buffalo_l": "det_10g.onnx", "buffalo_sc": "det_500m.onnx"}
MODELS = {"buffalo_l": ["det_10g.onnx", "w600k_r50.onnx"], "buffalo_sc": ["det_500m.onnx"]}
RELEASE = "https://github.com/deepinsight/insightface/releases/download/v0.7"


def main():
    models = Path(__file__).resolve().parents[1] / "models"
    models.mkdir(exist_ok=True)
    for package, names in MODELS.items():
        missing = [name for name in names if not (models / name).exists()]
        for name in sorted(set(names) - set(missing)):
            print(f"Já disponível: {models / name}")
        if not missing:
            continue
        with TemporaryDirectory(dir=models) as temp:
            archive = Path(temp) / f"{package}.zip"
            print(f"A descarregar {package}.zip ...")
            urlretrieve(f"{RELEASE}/{package}.zip", archive)
            with ZipFile(archive) as bundle:
                for name in missing:
                    member, = (item for item in bundle.namelist() if Path(item).name == name)
                    extracted = Path(temp) / name
                    with bundle.open(member) as source, extracted.open("wb") as output:
                        copyfileobj(source, output)
                    extracted.replace(models / name)
                    print(f"Guardado: {models / name}")


if __name__ == "__main__":
    main()
