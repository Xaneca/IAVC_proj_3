"""Avaliação da identificação na galeria de voluntários.

Correr: python -m eval.eval_identity
Saída:  results/identity/

Imagens dos voluntários divididas em 3 grupos SEPARADOS:
  enrol → criar os templates
  val   → calibrar o threshold (escolhido aqui, nunca olhando para o teste)
  test  → resultados finais (inclui pessoas NÃO registadas, que devem dar Unknown)

TODO:
- Curvas FAR / FRR em função do threshold (val) → escolher o threshold → config.yaml
- No teste: Rank-1, FAR, FRR, taxa de rejeição correta de Unknown
- Efeito do nº de imagens de registo (1, 3, 5, 10) e de pose / luz / distância / oclusão
"""


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
