# Módulo 2 — Deteção facial

## 1. Objetivo

Detetar faces em vídeo e devolver caixa, confiança e cinco landmarks por face. [Issue #5](https://github.com/Xaneca/IAVC_proj_3/issues/5)

## 2. Decisões e alternativas

O SCRFD-10G é o modelo principal e o SCRFD-500M é a variante leve. A regressão de caixas estima a posição de cada face. O score mede a confiança do detetor. Os landmarks localizam olhos, nariz e cantos da boca. A supressão não máxima (NMS) elimina caixas sobrepostas de menor confiança.

## 3. Modelo, fonte e licença

Pesos dos pacotes InsightFace `buffalo_l` e `buffalo_sc`, para investigação não comercial. Os pesos são descarregados localmente. [InsightFace](https://github.com/deepinsight/insightface)

## 4. Dados e pré-processamento

A demo recebe frames BGR de webcam ou vídeo. O SCRFD devolve as coordenadas na escala do frame original. O overlay desenha caixa, score e landmarks.

## 5. Parâmetros

Entrada padrão de 640 px, score mínimo de 0,5 e NMS de 0,4. O ONNX Runtime usa CUDA quando disponível, com CPU como alternativa.

## 6. Resultados

Tempos preliminares em CPU, com 30 medições por configuração. [CSV](../../results/detection/timings.csv)

| Modelo | Entrada | Mediana (ms) | P95 (ms) |
| --- | ---: | ---: | ---: |
| SCRFD-10G | 320 | 51,438 | 53,956 |
| SCRFD-10G | 640 | 202,977 | 213,508 |
| SCRFD-500M | 320 | 7,682 | 9,836 |
| SCRFD-500M | 640 | 27,926 | 32,417 |

## 7. Limitações e ética

A medição cobre latência preliminar. A estabilidade com várias pessoas na webcam do portátil fica por validar. A demo processa vídeo localmente.

## 8. Como reproduzir

**Instalar e descarregar os pesos.** A partir da raiz do projeto, com o ambiente Python ativo:

```bash
python -m pip install -r requirements.txt
python -m scripts.download_detection_models
```

O script coloca `det_10g.onnx` e `det_500m.onnx` em `models/`. A demo usa o modelo definido em `detector.model_path` no `config.yaml`. Para experimentar o modelo leve, definir esse caminho como `models/det_500m.onnx`.

**Correr a demo.** Usar a webcam de índice 0 ou passar o caminho para um vídeo:

```bash
python -m scripts.demo_detection --source 0
python -m scripts.demo_detection --source "caminho/video.mp4"
```

A janela mostra caixas, confiança, cinco landmarks, contagem de faces e FPS. Carregar em `q` ou `Esc` para sair.

**Medir a latência.** Passar uma ou mais imagens locais. Por omissão, o script faz cinco iterações de aquecimento e 30 medições por configuração. O CSV inclui média, mediana, P95 e provider para cada combinação de modelo e resolução.

```bash
python -m eval.eval_detection "caminho/imagem.jpg" --output results/detection/timings_local.csv
```
