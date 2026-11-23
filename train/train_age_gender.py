"""Treina a cabeça de idade/género no UTKFace (backbone congelado).

Correr: python -m train.train_age_gender
Saída:  models/age_gender.pt

TODO:
1. Dataset a partir de data/splits/utkface.csv (coluna processed_112; linhas vazias = sem face, ignorar)
2. Para cada batch: backbone(crops)["layer3"] → AgeGenderHead
3. Loss = L1(idade) + CrossEntropy(género)
4. Early stopping pelo MAE no split de validação
5. Guardar o state_dict em models/age_gender.pt
"""


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
