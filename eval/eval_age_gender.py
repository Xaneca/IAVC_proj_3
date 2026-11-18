"""Avaliação de idade e género no split de teste do UTKFace.

Correr: python -m eval.eval_age_gender
Saída:  results/age_gender/

TODO:
- Idade: MAE, RMSE, MAE por grupo etário (0-9, 10-19, ...)
- Género: precision / recall / F1 por classe + matriz de confusão
- Resultados por grupo demográfico (coluna race do UTKFace) → discussão de enviesamento
- Efeito do threshold de confiança (quantas previsões ficam "incertas")
"""


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
