"""Cria o backbone partilhado: ArcFace R50 com a saída da layer3 exposta.

O w600k_r50.onnx (iResNet-50, stages 3-4-14-3) só declara como saída o
embedding de identidade (512). A layer3 é calculada em cada passagem, mas
descartada. Este script acrescenta-a como segunda saída, sem mudar pesos nem
operações:

    entrada "input"      (B, 3, 112, 112)  BGR→RGB, (x/255 - 0,5) / 0,5
    saída   "layer3"     (B, 256, 14, 14)  fim do stage 3 → cabeças de atributos
    saída   "embedding"  (B, 512)          embedding ArcFace → identidade

A layer3 é a última soma residual com 256×14×14 (16.ª Add do grafo).
O script verifica que o embedding é igual ao do modelo original.

Correr: python -m scripts.export_backbone
Saída:  models/arcface_r50_l3.onnx
"""
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, shape_inference

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "models/w600k_r50.onnx"
TARGET = REPO / "models/arcface_r50_l3.onnx"
LAYER3_SHAPE = (256, 14, 14)
STAGES = (3, 4, 14, 3)


def find_layer3(model: onnx.ModelProto) -> str:
    """Saída da última soma residual do stage 3 (contando os blocos por stage)."""
    adds = [node.output[0] for node in model.graph.node if node.op_type == "Add"]
    if len(adds) != sum(STAGES):
        raise ValueError(f"Esperava {sum(STAGES)} blocos residuais (iResNet-50), encontrei {len(adds)}")
    return adds[sum(STAGES[:3]) - 1]


def export() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Falta {SOURCE}. Executa python -m scripts.download_models.")
    model = onnx.load(str(SOURCE))
    graph = model.graph
    layer3 = find_layer3(model)
    embedding = graph.output[0].name

    # Nomes legíveis e batch dinâmico na entrada e nas saídas.
    old_input = graph.input[0].name
    for node in graph.node:
        node.input[:] = ["input" if name == old_input else name for name in node.input]
    graph.input[0].name = "input"
    graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch"

    graph.node.extend([helper.make_node("Identity", [layer3], ["layer3"]),
                       helper.make_node("Identity", [embedding], ["embedding"])])
    del graph.output[:]
    graph.output.extend([
        helper.make_tensor_value_info("layer3", onnx.TensorProto.FLOAT, ["batch", *LAYER3_SHAPE]),
        helper.make_tensor_value_info("embedding", onnx.TensorProto.FLOAT, ["batch", 512]),
    ])
    del graph.value_info[:]
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(TARGET))
    print(f"Guardado: {TARGET} (layer3 = tensor '{layer3}')")


def verify() -> None:
    """Mesmo embedding que o original; layer3 com a forma certa; batch > 1."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (4, 3, 112, 112)).astype(np.float32)
    providers = ["CPUExecutionProvider"]
    original = ort.InferenceSession(str(SOURCE), providers=providers)
    new = ort.InferenceSession(str(TARGET), providers=providers)

    reference = np.concatenate([original.run(None, {original.get_inputs()[0].name: x[i:i + 1]})[0]
                                for i in range(len(x))])
    layer3, embedding = new.run(["layer3", "embedding"], {"input": x})
    if layer3.shape != (len(x), *LAYER3_SHAPE):
        raise AssertionError(f"layer3 com forma {layer3.shape}")
    diff = float(np.abs(embedding - reference).max())
    if diff > 1e-4:
        raise AssertionError(f"Embedding diferente do original (máx. {diff:.2e})")
    print(f"Verificado: layer3 {layer3.shape}, embedding {embedding.shape}, "
          f"diferença máx. para o original {diff:.1e}")


if __name__ == "__main__":
    export()
    verify()
