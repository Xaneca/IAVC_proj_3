# Módulo 3 — Alinhamento facial

## 1. Objetivo

Alinhar faces a partir de cinco landmarks e partilhar o mesmo crop entre a aplicação e a API offline. [Issue #6](https://github.com/Xaneca/IAVC_proj_3/issues/6)

## 2. Decisões e alternativas

Usar a transformação de semelhança do InsightFace. Para 224 píxeis, aplicar o template ArcFace com coordenadas duplicadas diretamente à imagem. Os datasets já recortados (UTKFace, AffectNet) são realinhados com o mesmo detetor, para que treino e aplicação vejam o mesmo crop. Como o SCRFD a 640 px não deteta faces que ocupam a imagem toda, a API offline repete a deteção com uma margem preta (`retry_pad`). Reduzir o `input_size` do detetor também resolvia, mas deixava de ser o detetor da aplicação. Para preencher a margem no crop, a reflexão criava olhos espelhados e a replicação criava riscas, por isso ficou o preto.

## 3. Modelo, fonte e licença

SCRFD-10G fornece os landmarks. O código InsightFace é MIT e os pesos disponibilizados destinam-se a investigação não comercial. [InsightFace](https://github.com/deepinsight/insightface)

## 4. Dados e pré-processamento

A entrada é BGR. A aplicação envia as faces analisáveis num único lote e mantém as restantes deteções no overlay. A grelha usa a fotografia NASA de Eileen Collins distribuída pelo scikit-image. [Fonte da imagem](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut)

## 5. Parâmetros

Crops de 112×112 e 224×224, caixa mínima de 40 px por lado, confiança mínima de 0,5 e desvio lateral máximo do nariz de 0,25 da distância interocular. No `train.prepare_data`, `retry_pad=0.5` (margem de metade do lado em cada borda).

## 6. Resultados

Testes confirmaram dimensões, posição dos landmarks, filtragem e seleção offline. Com SCRFD real, duas faces seguiram num lote e o crop offline coincidiu píxel a píxel com o da aplicação. [Grelha](../figs/alinhamento.png)

Numa amostra de 300 imagens por dataset, sem margem o SCRFD não detetou nenhuma face. Com `retry_pad=0.5`, ficaram alinhadas 99,0% das imagens do UTKFace e 97,3% das do AffectNet. As falhas ficam registadas em `data/aligned/<dataset>/report.csv`.

## 7. Limitações e ética

O desvio do nariz é uma estimativa de pose. A grelha mostra inclinações no plano. A aplicação processa as imagens localmente. A webcam com os pesos finais fica por validar. Nos datasets já recortados, a zona do crop ArcFace fora da imagem original fica preta, o que na webcam só acontece junto à borda do frame.

## 8. Como reproduzir

Executar `python -m unittest discover -s tests -p test_align.py -v`. Para gerar a grelha, executar `python -m scripts.demo_alignment caminho/astronaut.png docs/figs/alinhamento.png`.

A API comunicada ao André é `detect_and_align(image, size, pick="largest")`. Devolve a maior face analisável ou `None` e aceita um detetor existente via `detector=`. Para realinhar os datasets, executar `python -m train.prepare_data` (opções `--size 224` e `--datasets utkface affectnet`).
