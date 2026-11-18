"""Avaliação da deteção no WIDER Face (split de validação).

Correr: python -m eval.eval_detection
Saída:  results/detection/

TODO:
- AP em Easy / Medium / Hard (toolbox oficial do WIDER Face)
- Tempo de inferência por imagem e por resolução de entrada (320 / 640)
- Comparar pelo menos 2 detetores (ex.: SCRFD-500M vs SCRFD-10G)
- Exemplos qualitativos: caras pequenas, pose, oclusão, muitas caras
"""


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
