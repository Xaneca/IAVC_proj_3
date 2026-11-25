"""ArcFace iResNet-50 (stages 3-4-14-3) em PyTorch, com os pesos do w600k_r50.onnx.

Arquitetura do arcface_torch (InsightFace), na forma em que vem no ONNX: as
BatchNorm que se seguem às convoluções já estão fundidas nelas (convoluções
com bias). Sobram a BN de entrada de cada bloco, a bn2 final e a BN do embedding.

    entrada (B, 3, 112, 112)  RGB, (x/255 - 0,5) / 0,5
    stem    (B,  64, 112, 112)
    layer1  (B,  64,  56,  56)   stage 1
    layer2  (B, 128,  28,  28)   stage 2
    layer3  (B, 256,  14,  14)   stage 3
    layer4  (B, 512,   7,   7)   stage 4
    saída   (B, 512)             embedding de identidade (sem normalizar)

Os pesos são convertidos uma vez por scripts/export_backbone.py para
models/arcface_r50.pt.
"""
from pathlib import Path
import torch
import torch.nn as nn

STAGES = (3, 4, 14, 3)
CHANNELS = (64, 128, 256, 512)


class Block(nn.Module):
    """Bloco residual iResNet: BN → conv3×3 → PReLU → conv3×3 (stride) + atalho."""

    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        # A ordem dos atributos segue a ordem dos nós no ONNX (usada ao carregar os pesos).
        self.bn1 = nn.BatchNorm2d(cin, eps=1e-5)
        self.conv1 = nn.Conv2d(cin, cout, 3, 1, 1)
        self.prelu = nn.PReLU(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride, 1)
        self.downsample = nn.Conv2d(cin, cout, 1, stride) if stride != 1 or cin != cout else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.prelu(self.conv1(self.bn1(x))))
        return out + (x if self.downsample is None else self.downsample(x))


def _stage(cin: int, cout: int, blocks: int) -> nn.Sequential:
    return nn.Sequential(Block(cin, cout, 2), *[Block(cout, cout, 1) for _ in range(blocks - 1)])


class IResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1), nn.PReLU(64))
        cins = (64, *CHANNELS[:-1])
        self.layer1, self.layer2, self.layer3, self.layer4 = (
            _stage(cin, cout, n) for cin, cout, n in zip(cins, CHANNELS, STAGES))
        self.bn2 = nn.BatchNorm2d(512, eps=1e-5)
        self.fc = nn.Linear(512 * 7 * 7, 512)
        self.features = nn.BatchNorm1d(512, eps=1e-5)

    @property
    def stages(self) -> list[nn.Sequential]:
        return [self.layer1, self.layer2, self.layer3, self.layer4]

    def trunk(self, x: torch.Tensor, last_stage: int = 2) -> torch.Tensor:
        """Stem + stages 1..last_stage (a parte partilhada por identidade e atributos)."""
        x = self.stem(x)
        for stage in self.stages[:last_stage]:
            x = stage(x)
        return x

    def embed(self, trunk: torch.Tensor, last_stage: int = 2) -> torch.Tensor:
        """Restantes stages originais + fc → embedding de identidade (512, sem normalizar)."""
        x = trunk
        for stage in self.stages[last_stage:]:
            x = stage(x)
        return self.features(self.fc(self.bn2(x).flatten(1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(self.trunk(x))


def load_arcface(weights: str | Path, device: str = "cpu") -> IResNet50:
    """IResNet50 com os pesos de models/arcface_r50.pt, congelado e em modo eval."""
    if not Path(weights).is_file():
        raise FileNotFoundError(
            f"Backbone em falta: {weights}. Executa python -m scripts.download_models "
            "e python -m scripts.export_backbone.")
    model = IResNet50()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.requires_grad_(False)
    return model.eval().to(device)


def weights_from_onnx(model: IResNet50, onnx_path: str | Path) -> None:
    """Copia os pesos do w600k_r50.onnx, emparelhando os nós pela ordem do grafo."""
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(str(onnx_path)).graph
    init = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    ops = {nn.Conv2d: "Conv", nn.Linear: "Gemm", nn.PReLU: "PRelu",
           nn.BatchNorm2d: "BatchNormalization", nn.BatchNorm1d: "BatchNormalization"}
    nodes = [(n.op_type, [init[i] for i in n.input[1:]]) for n in graph.node
             if n.op_type in set(ops.values())]
    modules = [m for m in model.modules() if type(m) in ops]
    if len(nodes) != len(modules):
        raise ValueError(f"{len(nodes)} nós com pesos no ONNX, {len(modules)} módulos em PyTorch")

    with torch.no_grad():
        for module, (op, arrays) in zip(modules, nodes):
            if ops[type(module)] != op:
                raise ValueError(f"Ordem diferente: {type(module).__name__} vs {op}")
            tensors = [torch.from_numpy(a.copy()) for a in arrays]
            if isinstance(module, nn.PReLU):
                targets = [module.weight]
                tensors = [tensors[0].flatten()]
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                targets = [module.weight, module.bias, module.running_mean, module.running_var]
            else:
                targets = [module.weight, module.bias]
            for target, tensor in zip(targets, tensors, strict=True):
                if target.shape != tensor.shape:
                    raise ValueError(f"{type(module).__name__}: {tuple(target.shape)} vs {tuple(tensor.shape)}")
                target.copy_(tensor)
