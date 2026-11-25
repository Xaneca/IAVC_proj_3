# Módulo 4 — Datasets

## 1. Objetivo

**Idade e género.** Preparar o UTKFace para treinar e avaliar as cabeças de idade e género: labels, splits e crops 112×112 obtidos por redimensionamento das imagens já recortadas e alinhadas do dataset. [Issue #2](https://github.com/Xaneca/IAVC_proj_3/issues/2) · [Issue #7](https://github.com/Xaneca/IAVC_proj_3/issues/7)

**Expressão.** Harmonizar sete classes. Usar AffectNet para treino e avaliação intra-dataset e BTFER7 apenas para avaliação cross-dataset. [Issue #3](https://github.com/Xaneca/IAVC_proj_3/issues/3)

## 2. Decisões e alternativas

**Idade e género.** Usa-se só o UTKFace; o FG-NET, previsto para avaliação cross-dataset, foi abandonado. A versão descarregada já vem recortada e alinhada (200×200), por isso as imagens são apenas redimensionadas para 112×112, sem deteção nem alinhamento. A alternativa, realinhar cada imagem com o SCRFD-10G e o template ArcFace da aplicação ([Módulo 3](03-alinhamento.md)), dava ao treino o mesmo enquadramento da webcam, mas exigia uma margem preta para o SCRFD detetar a face, deixava uma faixa preta no topo dos crops e eliminava 382 imagens (1,6%). O redimensionamento é mais simples e mantém todas as imagens, à custa de um enquadramento diferente do da aplicação. Os splits ficam num único CSV com coluna `split`, no mesmo formato da expressão, em vez de pastas por split, para não duplicar imagens.

**Expressão.** Na experiência local, as pastas AffectNet `train` e `val` formam o conjunto de onde é extraída a validação interna. A pasta local `test` fica reservada para avaliação. O BTFER7 é usado exclusivamente para teste.

## 3. Modelo, fonte e licença

**Idade e género.** O pré-processamento não usa um modelo. O UTKFace "Aligned & Cropped" foi obtido no Kaggle. O UTKFace destina-se apenas a investigação não comercial e os direitos das imagens pertencem aos seus autores. [UTKFace cropped (Kaggle)](https://www.kaggle.com/datasets/moritzm00/utkface-cropped) · [UTKFace](https://susanqq.github.io/UTKFace/)

**Expressão.** A etapa não usa um modelo. O AffectNet oficial tem acordo de uso académico. O Kaggle classifica o subconjunto AffectNet Aligned v1 como `Other` e o BTFER7 v1 como CC0. [Acordo oficial](https://mohammadmahoor.com/wp-content/uploads/2024/06/AffectNet-Agreement-v2.1-10May2024.pdf) · [Metadados AffectNet Aligned](https://www.kaggle.com/api/v1/datasets/view/yakhyokhuja/affectnetaligned) · [Metadados BTFER7](https://www.kaggle.com/api/v1/datasets/view/gipinze/btfer7)

## 4. Dados e pré-processamento

**Idade e género.** As labels vêm do nome do ficheiro, `[idade]_[género]_[etnia]_[data].jpg.chip.jpg`, com género 0 = masculino e 1 = feminino (categorias do dataset). A idade é discretizada em 10 classes, `0-4`, `5-9`, `10-19`, …, `70-79`, `80+`, com classes de 5 anos nas crianças. O split 70/15/15 é estratificado por classe de idade e género. Cada imagem é depois redimensionada de 200×200 para 112×112. O CSV de splits ganha a coluna `processed_112` com o caminho do crop, vazia só se a imagem não abrir.

**Expressão.** O mapeamento usa as classes de `configs/expression_classes.yaml`, exclui `contempt` e normaliza `Angry` para `anger`. A deduplicação compara SHA-256 dos ficheiros e dos píxeis normalizados, antes de separar a validação interna.

## 5. Parâmetros

**Idade e género.** Split 70/15/15, seed 42. Redimensionamento 200×200 → 112×112 com `cv2.INTER_AREA` (224×224 com `INTER_CUBIC`, para ViT, com `--size 224`), crops em JPEG de qualidade 95.

**Expressão.** Validação interna estratificada de 10%, seed 42. Os manifestos usam `path,label,split`, com caminhos relativos à raiz de cada dataset.

## 6. Resultados

**Idade e género.** O UTKFace tem 23 708 imagens, divididas em 16 595 de treino, 3 556 de validação e 3 557 de teste. Todas as imagens abrem, por isso ficam 23 708 crops (100%). [Distribuições](../figs/utkface/)

**Expressão.** A auditoria dos ZIPs detetou 659 grupos de duplicados e conservou no máximo uma entrada por grupo. Os manifestos finais da experiência têm 30 396 imagens AffectNet para treino, 3 377 para validação e 2 772 para teste. O teste BTFER7 tem 1 916 imagens. [Figuras de distribuição](../figs/)

## 7. Limitações e ética

**Idade e género.** As idades do UTKFace são aproximadas e há erros de anotação, por exemplo grandes planos de olhos e adultos marcados como crianças; o pré-processamento não remove nenhum. O género é binário e corresponde às categorias do dataset, não à identidade de género. A distribuição é desequilibrada: metade das imagens tem entre 20 e 39 anos, e as classes `70-79` e `80+` têm só 699 e 673. O enquadramento do UTKFace (alinhamento próprio, com mais cabelo e pescoço) difere do template ArcFace que a aplicação usa na webcam, o que pode baixar a exatidão em tempo real face à avaliada no teste. Os perfis, que a aplicação filtra, ficam no treino e no teste.

**Expressão — proveniência.** A experiência usa os rótulos atuais e as partições locais do AffectNet Aligned. A correspondência com a validação oficial e a autorização da cópia não foram verificadas. O publicador do BTFER7 declara CC0, com direitos por imagem por verificar.

## 8. Como reproduzir

**Idade e género.** Descarregar o UTKFace cropped e extrair as imagens para `data/UTKFace/UTKFace/`. Na raiz do projeto, executar:

```bash
python -m scripts.data.prepare_utkface                      # data/splits/utkface.csv + figuras em docs/figs/utkface/
python -m scripts.data.preprocess_faces --dataset utkface   # crops em data/processed/utkface_112/
```

O primeiro comando reescreve o CSV sem a coluna `processed_112`; voltar a correr o segundo repõe-na sem refazer os crops já gravados. As imagens que não abrem ficam em `data/processed/utkface_112/removed.csv` (`unreadable`). No treino, usar as linhas com `processed_112` preenchido.

**Expressão.** Descarregar os dois ZIPs e guardá-los em `data/downloads/`. Extrair o conteúdo do AffectNet para `data/AffectNET/`, preservando a pasta `AffectNetCustom/`, e o do BTFER7 para `data/BTFER7/`, preservando `Batch_Ready 7/`. [AffectNet Aligned](https://www.kaggle.com/datasets/yakhyokhuja/affectnetaligned) · [BTFER7](https://www.kaggle.com/datasets/gipinze/btfer7)

Na raiz do projeto, executar:

```bash
python -m pip install -r requirements-datasets.txt
python -m scripts.data.prepare_expression_datasets setup-local
```

Como no UTKFace, os CSVs ficam ao lado da pasta das imagens. O AffectNet recebe `all_labels.csv`, `train_labels.csv`, `val_labels.csv` e `test_labels.csv`. O BTFER7 recebe apenas `all_labels.csv` e `test_labels.csv`. O comando reutiliza os splits versionados.

Executar `python -m unittest discover -s tests -p test_expression_datasets.py -v`. Para a auditoria, executar `python -m scripts.data.prepare_expression_datasets preliminary --affectnet-zip data/downloads/affectnetaligned.zip --btfer7-zip data/downloads/btfer7.zip --clean`.

Os CSVs finais e a auditoria estão em `data/splits`. A auditoria regista os hashes dos ZIPs, o protocolo local e a seed usados na atribuição determinística das partições.
