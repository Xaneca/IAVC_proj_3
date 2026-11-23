# Módulo 3 — Alinhamento facial

## 1. Objetivo

Alinhar faces a partir de cinco landmarks e partilhar o mesmo crop entre a aplicação e a API offline. [Issue #6](https://github.com/Xaneca/IAVC_proj_3/issues/6)

## 2. Decisões e alternativas

Usar a transformação de semelhança do InsightFace. Para 224 píxeis, aplicar o template ArcFace com coordenadas duplicadas diretamente à imagem.

## 3. Modelo, fonte e licença

SCRFD-10G fornece os landmarks. O código InsightFace é MIT e os pesos disponibilizados destinam-se a investigação não comercial. [InsightFace](https://github.com/deepinsight/insightface)

## 4. Dados e pré-processamento

A entrada é BGR. A aplicação envia as faces analisáveis num único lote e mantém as restantes deteções no overlay. A grelha usa a fotografia NASA de Eileen Collins distribuída pelo scikit-image. [Fonte da imagem](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut)

## 5. Parâmetros

Crops de 112×112 e 224×224, caixa mínima de 40 px por lado, confiança mínima de 0,5 e desvio lateral máximo do nariz de 0,25 da distância interocular.

## 6. Resultados

Testes confirmaram dimensões, posição dos landmarks, filtragem e seleção offline. Com SCRFD real, duas faces seguiram num lote e o crop offline coincidiu píxel a píxel com o da aplicação. [Grelha](../figs/alinhamento.png)

## 7. Limitações e ética

O desvio do nariz é uma estimativa de pose. A grelha mostra inclinações no plano. A aplicação processa as imagens localmente. A webcam com os pesos finais fica por validar.

## 8. Como reproduzir

Executar `python -m unittest discover -s tests -p test_align.py -v`. Para gerar a grelha, executar `python -m scripts.demo_alignment caminho/astronaut.png docs/figs/alinhamento.png`.

A API comunicada ao André é `detect_and_align(image, size, pick="largest")`. Devolve a maior face analisável ou `None` e aceita um detetor existente via `detector=`.
