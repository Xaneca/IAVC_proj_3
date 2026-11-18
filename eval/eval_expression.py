"""Avaliação da expressão: intra-dataset (AffectNet) e cross-dataset (BTFER).

Correr: python -m eval.eval_expression
Saída:  results/expression/

TODO:
- Intra: split de validação oficial do AffectNet
- Cross: BTFER sem re-treino, só com as classes comuns aos dois datasets
- Métricas: accuracy, balanced accuracy, macro precision / recall / F1
- Matriz de confusão para cada caso → erros mais frequentes
"""


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
