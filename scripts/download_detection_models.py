"""Obtém os pesos de deteção das distribuições oficiais InsightFace v0.7."""

from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve
from zipfile import ZipFile


MODELS = {"buffalo_l": "det_10g.onnx", "buffalo_sc": "det_500m.onnx"}
RELEASE = "https://github.com/deepinsight/insightface/releases/download/v0.7"


def main():
    models = Path(__file__).resolve().parents[1] / "models"
    models.mkdir(exist_ok=True)
    for package, name in MODELS.items():
        target = models / name
        if target.exists():
            print(f"Já disponível: {target}")
            continue
        with TemporaryDirectory(dir=models) as temp:
            archive = Path(temp) / f"{package}.zip"
            urlretrieve(f"{RELEASE}/{package}.zip", archive)
            extracted = Path(temp) / name
            with ZipFile(archive) as bundle:
                member, = (item for item in bundle.namelist() if Path(item).name == name)
                with bundle.open(member) as source, extracted.open("wb") as output:
                    copyfileobj(source, output)
            extracted.replace(target)
        print(f"Guardado: {target}")


if __name__ == "__main__":
    main()
