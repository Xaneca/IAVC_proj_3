# Real-Time Facial Analysis

Aplicação local que analisa a webcam em tempo real: deteção de caras, idade aparente,
género (categoria do dataset), expressão facial e identificação numa galeria de voluntários.

## Arquitetura

```
webcam → detector (SCRFD) → alinhamento 112×112 → backbone ArcFace partilhado
                                                    ├── cabeça idade/género
                                                    ├── cabeça expressão
                                                    └── embedding → galeria (cosine + threshold → nome / Unknown)
```

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `app/` | A aplicação em tempo real |
| `train/` | Preparação dos dados e treino das cabeças |
| `eval/` | Avaliações que geram os resultados do relatório |
| `data/` | Datasets (não incluídos no repositório) |
| `models/` | Pesos dos modelos (não incluídos no repositório) |
| `gallery/` | Embeddings dos voluntários (não incluídos no repositório) |
| `results/` | Métricas e figuras |
| `report/` | Relatório técnico |

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Modelos

Os pesos não são redistribuídos. Descarregar para `models/`:

| Ficheiro | Origem | Licença |
|---|---|---|
| `det_10g.onnx` (SCRFD) | InsightFace (`buffalo_l`) | Investigação não comercial |
| `arcface_r50.pth` (iResNet-50) | InsightFace `arcface_torch` | Investigação não comercial |
| `age_gender.pt`, `expression.pt` | Treinados com `train/` | — |

## Datasets

Colocar em `data/raw/`: WIDER Face, UTKFace, AffectNet, BTFER (links no enunciado).

## Como correr

Para executar só a deteção facial, consultar o [guia de deteção](docs/modulos/02-detecao.md#8-como-reproduzir).

Todos os comandos a partir da raiz do projeto:

```bash
python -m train.prepare_data          # alinhar datasets + criar splits
python -m train.train_age_gender
python -m train.train_expression
python -m app.main                    # aplicação
python -m eval.eval_expression        # (e restantes scripts de eval/)
```

### Teclas na aplicação

| Tecla | Ação |
|---|---|
| `e` | Registar pessoa (pede o nome na consola) |
| `d` | Apagar pessoa |
| `c` | Limpar a galeria |
| `1` / `2` / `3` | Ligar/desligar idade-género / expressão / identidade |
| `q` | Sair |

## Ética e privacidade

- Tudo corre localmente; nada é enviado para a cloud.
- A galeria guarda apenas embeddings (não fotografias), só de voluntários com consentimento.
- `c` na aplicação apaga todos os dados biométricos.
- A idade é *aparente*, o género é uma *categoria do dataset* e a expressão descreve a
  *configuração visível da cara*, não o estado emocional da pessoa.
- Não usar para vigilância, controlo de acessos, presenças ou outras decisões com consequências.

## Limitações conhecidas

_A preencher._

## Dados

- UTKFace (Aligned and Cropped): https://www.kaggle.com/datasets/moritzm00/utkface-cropped?resource=download

- WIDER Face (test): http://shuoyang1213.me/WIDERFACE/
