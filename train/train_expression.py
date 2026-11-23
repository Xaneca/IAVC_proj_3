"""Treina a cabeça de expressão no AffectNet (backbone congelado).

Correr: python -m train.train_expression
Saída:  models/expression.pt

TODO:
1. Dataset a partir de data/splits/affectnet.csv (coluna processed_112; linhas vazias = sem face, ignorar)
   (só as 7 classes de app/heads.py: retirar "contempt")
2. Para cada batch: backbone(crops)["layer3"] → ExpressionHead
3. Loss = CrossEntropy com pesos por classe (o AffectNet é muito desequilibrado)
4. Early stopping pelo macro-F1 no split de validação
5. Guardar o state_dict em models/expression.pt
"""


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
