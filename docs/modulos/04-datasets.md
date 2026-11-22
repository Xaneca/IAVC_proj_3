# Módulo 4 — Datasets de expressão

## 1. Objetivo

Harmonizar sete classes. Usar AffectNet para treino e avaliação intra-dataset e BTFER7 apenas para avaliação cross-dataset. [Issue #3](https://github.com/Xaneca/IAVC_proj_3/issues/3)

## 2. Decisões e alternativas

Na experiência local, as pastas AffectNet `train` e `val` formam o conjunto de onde é extraída a validação interna. A pasta local `test` fica reservada para avaliação. O BTFER7 é usado exclusivamente para teste.

## 3. Modelo, fonte e licença

A etapa não usa um modelo. O AffectNet oficial tem acordo de uso académico. O Kaggle classifica o subconjunto AffectNet Aligned v1 como `Other` e o BTFER7 v1 como CC0. [Acordo oficial](https://mohammadmahoor.com/wp-content/uploads/2024/06/AffectNet-Agreement-v2.1-10May2024.pdf) · [Metadados AffectNet Aligned](https://www.kaggle.com/api/v1/datasets/view/yakhyokhuja/affectnetaligned) · [Metadados BTFER7](https://www.kaggle.com/api/v1/datasets/view/gipinze/btfer7)

## 4. Dados e pré-processamento

O mapeamento usa as classes de `configs/expression_classes.yaml`, exclui `contempt` e normaliza `Angry` para `anger`. A deduplicação compara SHA-256 dos ficheiros e dos píxeis normalizados, antes de separar a validação interna.

## 5. Parâmetros

Validação interna estratificada de 10%, seed 42. Os manifestos usam `path,label,split`, com caminhos relativos à raiz de cada dataset.

## 6. Resultados

A auditoria dos ZIPs detetou 659 grupos de duplicados e conservou no máximo uma entrada por grupo. Os manifestos finais da experiência têm 30 396 imagens AffectNet para treino, 3 377 para validação e 2 772 para teste. O teste BTFER7 tem 1 916 imagens. [Figuras de distribuição](../figs/)

## 7. Limitações e ética

**Proveniência.** A experiência usa os rótulos atuais e as partições locais do AffectNet Aligned. A correspondência com a validação oficial e a autorização da cópia não foram verificadas. O publicador do BTFER7 declara CC0, com direitos por imagem por verificar.

## 8. Como reproduzir

Executar `python -m unittest discover -s tests -p test_expression_datasets.py -v`. Para a auditoria, executar `python -m scripts.data.prepare_expression_datasets preliminary --affectnet-zip data/downloads/affectnetaligned.zip --btfer7-zip data/downloads/btfer7.zip --clean`.

Os CSVs finais e a auditoria estão em `data/splits`. A auditoria regista os hashes dos ZIPs, o protocolo local e a seed usados na atribuição determinística das partições.
