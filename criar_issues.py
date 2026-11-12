#!/usr/bin/env python3
"""
Cria milestones, labels e as 24 issues do Assignment 3 (Real-Time Facial Analysis)
no repositório GitHub, com descrição, lista de tarefas, dependências e responsáveis.

Requisitos: GitHub CLI (`gh`) autenticado com permissão de escrita no repositório.

Uso:
    python criar_issues.py --repo Xaneca/IAVC_proj_3 --a-user Xaneca --b-user <login-do-andre>
    python criar_issues.py --dry-run          # gera apenas issues_preview.md, sem tocar no GitHub
"""
import argparse
import json
import subprocess
import sys

YEAR = "2026"

MEMBERS = {"A": "Xaneca", "B": "André"}  # nomes mostrados no texto das issues

MILESTONES = {
    "M1": ("Etapa 1 – Pipeline mínimo", "2026-11-16",
           "Repositório, datasets, captura, deteção e alinhamento a funcionar no portátil."),
    "M2": ("Etapa 2 – Análise facial", "2026-11-22",
           "Backbone partilhado, cabeças MLP de idade/género e expressão integradas no pipeline."),
    "M3": ("Etapa 3 – Galeria de identidades", "2026-11-22",
           "Consentimento, enrolamento, embeddings, matching e rejeição 'Unknown'."),
    "M4": ("Etapa 4 – Otimização e avaliação", "2026-11-26",
           "Avaliações por componente, interface, desempenho e testes de sistema."),
    "M5": ("Etapa 5 – Entrega final", "2026-11-29",
           "README, relatório técnico, vídeo de demonstração e submissão."),
}

LABELS = {
    "stage-1": ("0e8a16", "Etapa 1 – pipeline mínimo"),
    "stage-2": ("1d76db", "Etapa 2 – análise facial"),
    "stage-3": ("5319e7", "Etapa 3 – galeria de identidades"),
    "stage-4": ("fbca04", "Etapa 4 – otimização e avaliação"),
    "stage-5": ("d93f0b", "Etapa 5 – entrega final"),
    "idade-genero": ("c5def5", "Classificador de idade e género"),
    "expressao": ("f9d0c4", "Classificador de expressão facial"),
    "identidade": ("d4c5f9", "Reconhecimento de identidade"),
    "sistema": ("bfdadc", "Pipeline, aplicação e infraestrutura"),
    "dados": ("fef2c0", "Datasets e pré-processamento"),
    "modelo": ("0052cc", "Modelos, treino e exportação"),
    "avaliacao": ("e99695", "Experiências e métricas"),
    "docs": ("0075ca", "Documentação e relatório"),
    "etica": ("b60205", "Ética, consentimento e privacidade"),
}

DOC_NOTE = (
    "Documentar em `{doc}` seguindo o template (objetivo · decisões e alternativas · "
    "modelo/fonte/licença · dados e pré-processamento · parâmetros · resultados · "
    "limitações e ética · como reproduzir)"
)

# ---------------------------------------------------------------------------
# Issues. "deps" referem-se ao número local (1..24); o script substitui pelos
# números reais do GitHub no momento da criação.
# ---------------------------------------------------------------------------
ISSUES = [
    # 1 -----------------------------------------------------------------------
    dict(
        title="Estrutura do repositório, ambiente, estruturas de dados e template de documentação",
        owner="A", due="11-13", milestone="M1", labels=["stage-1", "sistema", "docs"], deps=[],
        doc="docs/modulos/00-arquitetura.md",
        desc=(
            "Criar a base do projeto para que os dois membros possam trabalhar em paralelo a partir de 14 nov. "
            "O enunciado recomenda um pipeline modular em que os módulos comunicam por estruturas de dados bem "
            "definidas (frame, deteção, crop, embedding, previsão de atributos), para que um modelo possa ser "
            "trocado sem redesenhar a aplicação.\n\n"
            "Arquitetura acordada: **um backbone congelado partilhado** extrai features de cada face alinhada; "
            "a partir dessas features, **MLPs independentes** classificam idade, género e expressão. "
            "A identidade usa embeddings ArcFace."
        ),
        tasks=[
            "Criar a estrutura de pastas: `src/iavc/{capture,detection,alignment,backbone,heads,identity,ui,monitoring}`, `scripts/`, `eval/`, `configs/`, `docs/modulos/`, `docs/figs/`, `tests/`",
            "Criar `requirements.txt` com versões fixas (opencv-python, onnxruntime, insightface, torch, torchvision, timm, open_clip_torch, scikit-learn, pandas, matplotlib, PySide6, psutil, pyyaml, cryptography)",
            "Criar `.gitignore` que exclua `data/`, `models/`, `gallery/`, `*.onnx`, `*.pt`, `*.npy` e vídeos",
            "Definir dataclasses em `src/iavc/types.py`: `Frame`, `Detection` (bbox, score, 5 landmarks), `FaceCrop` (imagem, deteção, analisável), `Embedding`, `AttributePrediction` (idade, intervalo, rótulo de género + confiança, probabilidades de expressão, flags de incerteza), `IdentityMatch` (nome ou Unknown, similaridade)",
            "Definir interfaces base (Protocol/ABC) para Detector, Aligner, Backbone, Head e Gallery",
            "Criar `configs/default.yaml` com caminhos de modelos, limiares e flags de ativação dos módulos",
            "Criar o template de documentação e os ficheiros vazios em `docs/modulos/`: 00-arquitetura, 01-captura, 02-detecao, 03-alinhamento, 04-datasets, 05-backbone, 06-idade-genero, 07-expressao, 08-identidade, 09-interface, 10-desempenho, 11-testes-sistema, 12-etica",
            "Criar um esqueleto do README",
            "Combinar convenções: um branch por issue (`issue-N-nome`), PR com `Closes #N`, revisão pelo outro membro",
        ],
        done=[
            "Os dois membros clonam o repositório, criam o ambiente e importam `iavc` sem erros",
            "Os ficheiros de documentação existem com o template",
            "`00-arquitetura.md` tem o diagrama do pipeline e a descrição das estruturas de dados",
        ],
    ),
    # 2 -----------------------------------------------------------------------
    dict(
        title="Datasets de idade/género (UTKFace, FG-NET) e WIDER Face val",
        owner="A", due="11-13", milestone="M1", labels=["stage-1", "dados", "idade-genero"], deps=[],
        doc="docs/modulos/04-datasets.md",
        desc=(
            "Descarregar e organizar os datasets de idade/género e o WIDER Face (usado no benchmark de deteção, #17). "
            "O UTKFace é usado para treinar as cabeças de idade e género. O FG-NET serve só para avaliação "
            "cross-dataset da idade. Como os datasets vão ser pré-processados com o nosso próprio detetor e "
            "alinhamento (#7), é preferível a versão *in the wild* (raw) do UTKFace; a versão cropped/aligned "
            "fica como alternativa."
        ),
        tasks=[
            "Descarregar o UTKFace raw (*in the wild*) e a versão cropped (Kaggle) como alternativa",
            "Descarregar o FG-NET (Kaggle)",
            "Descarregar o WIDER Face val (imagens + anotações `wider_face_split`) e a ferramenta oficial de avaliação",
            "`scripts/data/prepare_utkface.py`: extrair do nome do ficheiro `[age]_[gender]_[race]_[datetime]`, descartar ficheiros mal formados e gerar CSV (path, age, gender, race)",
            "Split 70/15/15 estratificado por faixa etária e género, seed 42 → `data/splits/utkface_{train,val,test}.csv`",
            "`scripts/data/prepare_fgnet.py`: extrair a idade do nome do ficheiro e gerar CSV (só teste)",
            "Gerar estatísticas: histograma de idades e distribuição de género e etnia, guardadas em `docs/figs/` (serve a discussão de enviesamento no relatório)",
            "Verificar e registar as licenças de uso (académico, sem redistribuição)",
        ],
        done=[
            "Os CSVs de split são gerados por script, de forma reprodutível",
            "Os dados ficam fora do git",
            "As figuras de distribuição estão em `docs/figs/`",
        ],
    ),
    # 3 -----------------------------------------------------------------------
    dict(
        title="Datasets de expressão (AffectNet, BTFER) e mapeamento das 7 classes",
        owner="B", due="11-13", milestone="M1", labels=["stage-1", "dados", "expressao"], deps=[],
        doc="docs/modulos/04-datasets.md",
        desc=(
            "Descarregar e organizar os datasets de expressão. O AffectNet é usado para treino e avaliação "
            "intra-dataset (partição oficial); o BTFER serve só para avaliação cross-dataset, sem re-treino. "
            "O enunciado obriga a harmonizar as categorias e a considerar apenas as classes comuns aos dois "
            "datasets, e a garantir que nenhuma imagem é partilhada entre treino e avaliação."
        ),
        tasks=[
            "Descarregar o AffectNet aligned (Kaggle `yakhyokhuja/affectnetaligned`) e verificar a estrutura de pastas, as classes e a partição oficial",
            "Descarregar o BTFER7 (Kaggle `gipinze/btfer7`)",
            "Criar a tabela de mapeamento das classes comuns (neutral, happy, sad, surprise, fear, disgust, anger), removendo *contempt* se existir → `configs/expression_classes.yaml`",
            "Gerar CSVs (path, label, split) para os dois datasets",
            "Separar uma validação interna do treino do AffectNet (~10%, estratificada) para afinar hiperparâmetros; a partição oficial fica reservada para teste",
            "Verificar duplicados entre partições com hash das imagens",
            "Gerar a distribuição de classes dos dois datasets (o AffectNet é muito desbalanceado) → `docs/figs/`",
            "Verificar e registar as licenças de uso",
        ],
        done=[
            "Os CSVs e o mapeamento de classes são gerados por script",
            "Está confirmado que não há imagens em comum entre treino e avaliação",
            "As figuras de distribuição estão em `docs/figs/`",
        ],
    ),
    # 4 -----------------------------------------------------------------------
    dict(
        title="Aquisição de vídeo (webcam e ficheiro de vídeo)",
        owner="B", due="11-14", milestone="M1", labels=["stage-1", "sistema"], deps=[1],
        doc="docs/modulos/01-captura.md",
        desc=(
            "Módulo 1 do pipeline: adquirir frames da webcam do portátil, com ficheiro de vídeo como alternativa "
            "para testes repetíveis. A saída são objetos `Frame` (imagem, timestamp, índice) e a taxa de frames."
        ),
        tasks=[
            "Classe `VideoSource` com fonte configurável (índice de webcam ou caminho de ficheiro) e `read()` → `Frame`",
            "Captura numa thread separada, com buffer de 1 frame que descarta frames antigos para manter a latência baixa",
            "Resolução configurável (640×480, 1280×720) e medição do FPS da fonte",
            "Modo ficheiro com dois comportamentos: tempo real (salta frames) e todos os frames (para testes repetíveis); loop opcional",
            "`start()`, `stop()` e troca de fonte sem bloquear e libertando a câmara",
            "Script `scripts/demo_capture.py` que mostra o vídeo e o FPS",
            "Gravar 2–3 vídeos de teste curtos com pessoas que consentiram (guardados localmente, fora do git)",
        ],
        done=[
            "Captura estável da webcam e de ficheiro de vídeo, com o FPS visível",
        ],
    ),
    # 5 -----------------------------------------------------------------------
    dict(
        title="Deteção de faces com SCRFD + landmarks + overlay básico",
        owner="B", due="11-14", milestone="M1", labels=["stage-1", "sistema", "modelo"], deps=[1, 4],
        doc="docs/modulos/02-detecao.md",
        desc=(
            "Módulo 2: detetar todas as faces visíveis com um detetor pré-treinado e devolver caixa, confiança "
            "e 5 landmarks por face. Modelo principal: **SCRFD-10G** (InsightFace `buffalo_l`). Variante leve "
            "para comparação: **SCRFD-500M**. Isto conclui a Etapa 1 (caixas e keypoints visíveis na webcam)."
        ),
        tasks=[
            "Obter `det_10g.onnx` e `det_500m.onnx` (InsightFace) com script de download; registar a origem e a licença (uso não comercial, sem redistribuição)",
            "Classe `SCRFDDetector` que implementa a interface Detector, com tamanho de entrada, limiar de score e limiar de NMS configuráveis",
            "Devolver `list[Detection]` com bbox, score e 5 landmarks",
            "Escolher o provider do ONNX Runtime automaticamente (CUDA se houver, senão CPU)",
            "Overlay com caixa, score e landmarks",
            "Script `scripts/demo_detection.py` com a webcam e várias pessoas",
            "Medir o tempo de inferência por frame (2 variantes × entradas 320 e 640) e fazer uma tabela preliminar",
            "Documentar os conceitos pedidos na secção 4.1: regressão de caixas, confiança, landmarks, NMS",
        ],
        done=[
            "A deteção é estável com várias faces na webcam do portátil",
            "A tabela preliminar de tempos está na documentação",
        ],
    ),
    # 6 -----------------------------------------------------------------------
    dict(
        title="Alinhamento facial e filtro de faces visíveis",
        owner="A", due="11-15", milestone="M1", labels=["stage-1", "sistema"], deps=[5],
        doc="docs/modulos/03-alinhamento.md",
        desc=(
            "Módulo 3: produzir crops normalizados a partir dos 5 landmarks, por transformação de semelhança "
            "para o template ArcFace. É crítico porque bloqueia o pré-processamento dos datasets (#7 e #8): "
            "o treino tem de ver exatamente os mesmos crops que a webcam. Inclui o critério de face "
            "\"suficientemente visível\" pedido no enunciado."
        ),
        tasks=[
            "`align_face(image, landmarks5, size)` com transformação de semelhança (`cv2.estimateAffinePartial2D` ou skimage) para o template ArcFace 112×112",
            "Variante 224×224 (template escalado) para backbones ViT",
            "Filtro de visibilidade: tamanho mínimo da caixa (ex. 40 px), score mínimo e estimativa grosseira de yaw pelos landmarks → `FaceCrop.analysable=False`",
            "Processamento em lote de várias faces por frame",
            "API offline `detect_and_align(image, size, pick='largest')` para datasets com uma face por imagem (usada em #7 e #8)",
            "Grelha visual de crops alinhados com várias poses → `docs/figs/`",
            "Testes unitários básicos (dimensões, landmarks na posição do template)",
        ],
        done=[
            "Crops alinhados corretos em tempo real e através da API offline",
            "A API está estável e comunicada ao André para #8",
        ],
    ),
    # 7 -----------------------------------------------------------------------
    dict(
        title="Pré-processar UTKFace e FG-NET com o detetor e alinhamento do pipeline",
        owner="A", due="11-16", milestone="M2", labels=["stage-2", "dados", "idade-genero"], deps=[2, 6],
        doc="docs/modulos/04-datasets.md",
        desc=(
            "Gerar os crops de treino e teste com o mesmo detetor e alinhamento usados na aplicação, para "
            "eliminar a diferença de pré-processamento entre treino e webcam."
        ),
        tasks=[
            "Script genérico `scripts/data/preprocess_faces.py --dataset utkface|fgnet|affectnet|btfer` que use `detect_and_align` (partilhar com o André)",
            "Correr sobre o UTKFace raw; quando houver várias faces, escolher a maior ou a mais central",
            "Gerar crops 112×112 e 224×224 em `data/processed/utkface/`",
            "Registar a taxa de falha de deteção; nas falhas, usar a versão cropped do dataset e documentar",
            "Repetir para o FG-NET",
            "Atualizar os CSVs com os caminhos dos crops",
            "Grelha de exemplos por faixa etária → `docs/figs/`",
        ],
        done=[
            "Crops e CSVs prontos para extração de features",
            "Taxa de falhas registada na documentação",
        ],
    ),
    # 8 -----------------------------------------------------------------------
    dict(
        title="Pré-processar AffectNet e BTFER com o detetor e alinhamento do pipeline",
        owner="B", due="11-16", milestone="M2", labels=["stage-2", "dados", "expressao"], deps=[3, 6],
        doc="docs/modulos/04-datasets.md",
        desc=(
            "Igual a #7, para os datasets de expressão. Como o AffectNet aligned e o BTFER já vêm recortados, "
            "o detetor pode falhar em crops apertados; é preciso testar padding/upscale e ter uma alternativa."
        ),
        tasks=[
            "Usar o script genérico de #7 (`--dataset affectnet|btfer`)",
            "Testar padding (ex. 30%) e upscale antes da deteção nos crops apertados",
            "Alternativa quando não há deteção: redimensionamento direto do crop original (documentar e medir a percentagem)",
            "Gerar crops 112×112 e 224×224 em `data/processed/{affectnet,btfer}/`",
            "Calcular a taxa de falha **por classe**, para garantir que as falhas não enviesam as classes",
            "Atualizar os CSVs com os caminhos dos crops",
            "Grelha de exemplos por classe → `docs/figs/`",
        ],
        done=[
            "Crops e CSVs prontos para extração de features",
            "Taxa de falhas por classe documentada",
        ],
    ),
    # 9 -----------------------------------------------------------------------
    dict(
        title="Backbone partilhado: comparar 3 candidatos e escolher um",
        owner="AB", due="11-18", milestone="M2", labels=["stage-2", "modelo", "avaliacao", "idade-genero", "expressao"], deps=[7, 8],
        doc="docs/modulos/05-backbone.md",
        desc=(
            "Escolher **um** backbone congelado que extrai features de cada face, uma passagem por face, "
            "partilhadas pelas MLPs de idade, género e expressão.\n\n"
            "| Candidato | Entrada | Dim. | Nota |\n|---|---|---|---|\n"
            "| ArcFace R50 (`w600k_r50`, InsightFace) | 112 | 512 | Já é usado na identidade; se ganhar, há uma única passagem para tudo. Provavelmente fraco na expressão |\n"
            "| DINOv2 ViT-S/14 | 224 | 384 | Features genéricas fortes e leve (22M parâmetros) |\n"
            "| CLIP ViT-B/16 (OpenCLIP) | 224 | 512 | Bom em atributos faciais, mais pesado |\n\n"
            "**Divisão:** Xaneca avalia idade e género, André avalia expressão e latência. A decisão é tomada em conjunto."
        ),
        tasks=[
            "Wrapper comum `src/iavc/backbone/` com `extract(crops) -> np.ndarray` para os 3 candidatos, com a normalização de entrada correta de cada um",
            "Definir subconjuntos de comparação (ex. 5k treino / 2k validação por tarefa)",
            "[Xaneca] MLP pequena por candidato: MAE de idade e F1 de género na validação do UTKFace",
            "[André] MLP pequena por candidato: F1 macro de expressão na validação interna do AffectNet",
            "[André] Medir a latência em CPU por face (batch 1 e 4) e a memória de cada candidato",
            "Tabela comparativa e decisão justificada com critério explícito (desempenho nas 3 tarefas vs latência)",
            "Exportar o backbone escolhido para ONNX e verificar a equivalência numérica com PyTorch (diferença máx. < 1e-3)",
        ],
        done=[
            "Decisão registada com a tabela de evidência",
            "Backbone ONNX disponível para #10, #11 e #14",
        ],
    ),
    # 10 ----------------------------------------------------------------------
    dict(
        title="Features em cache e MLP de idade e género",
        owner="A", due="11-20", milestone="M2", labels=["stage-2", "modelo", "idade-genero"], deps=[9],
        doc="docs/modulos/06-idade-genero.md",
        desc=(
            "Treinar as cabeças de idade e género sobre as features do backbone congelado. Como o backbone não "
            "é treinado, as features são extraídas uma única vez e guardadas em disco, e cada treino leva minutos. "
            "A idade é tratada como **idade aparente**; o género prevê só as categorias do dataset."
        ),
        tasks=[
            "Extrair as features de UTKFace train/val/test e FG-NET → `data/features/*.npy` (incluindo crops espelhados horizontalmente como aumento de dados)",
            "MLP de idade (ex. d→256→1, dropout, loss L1)",
            "Variante de idade por distribuição (classificação em 0–100 + valor esperado, estilo DEX) para comparação",
            "MLP de género (ex. d→128→2, cross-entropy)",
            "Comparação opcional: MLP multitarefa (camada oculta partilhada → idade + género) vs duas MLP separadas",
            "Early stopping na validação, seeds fixas, configurações e checkpoints guardados",
            "Calibração da confiança do género (temperature scaling na validação) e limiar para marcar \"incerto\"",
            "Definir o intervalo de idade mostrado (ex. década ou ±MAE)",
            "Exportar as cabeças para ONNX",
        ],
        done=[
            "Cabeças ONNX disponíveis para #14",
            "Resultados de validação na documentação",
        ],
    ),
    # 11 ----------------------------------------------------------------------
    dict(
        title="Features em cache e MLP de expressão",
        owner="B", due="11-20", milestone="M2", labels=["stage-2", "modelo", "expressao"], deps=[9],
        doc="docs/modulos/07-expressao.md",
        desc=(
            "Treinar a cabeça de expressão (7 classes) sobre as features do backbone congelado. As previsões "
            "representam a **configuração facial visível**, não o estado emocional da pessoa."
        ),
        tasks=[
            "Extrair as features de AffectNet train/val interna/teste oficial e BTFER → `data/features/*.npy` (incluindo crops espelhados)",
            "MLP (ex. d→512→256→7, dropout)",
            "Tratar o desbalanceamento: pesos por classe ou weighted sampler; label smoothing",
            "Afinar hiperparâmetros só na validação interna (a partição oficial fica para #13)",
            "Temperature scaling e limiar de confiança para \"incerto\"",
            "Saída com top-k probabilidades",
            "Exportar a cabeça para ONNX",
        ],
        done=[
            "Cabeça ONNX disponível para #14",
            "Resultados de validação na documentação",
        ],
    ),
    # 12 ----------------------------------------------------------------------
    dict(
        title="Avaliação de idade e género",
        owner="A", due="11-22", milestone="M4", labels=["stage-4", "avaliacao", "idade-genero"], deps=[10],
        doc="docs/modulos/06-idade-genero.md",
        desc=(
            "Avaliação dos componentes conforme as secções 4.2 e 6.1: MAE (e RMSE) para a idade; precisão, "
            "recall, F1 por classe e matriz de confusão para o género, e não só accuracy. Inclui a discussão "
            "de limitações de anotação, diferenças demográficas e riscos."
        ),
        tasks=[
            "Script `eval/eval_age_gender.py` que gera automaticamente todas as tabelas e figuras",
            "Idade no UTKFace test: MAE, RMSE e CS@5 (opcional)",
            "Idade por faixa etária (0–12, 13–19, 20–29, …, 70+), por etnia e por género",
            "Cross-dataset FG-NET: MAE total e por faixa etária, sem re-treino",
            "Género: precisão, recall e F1 por classe, matriz de confusão, e desagregação por etnia e faixa etária",
            "Curva de cobertura vs desempenho do limiar de \"incerto\" (percentagem retida vs ganho)",
            "Comparar com o modelo pré-treinado `genderage` do InsightFace no mesmo conjunto de teste",
            "Robustez a degradações sintéticas (blur, baixa resolução, brilho)",
            "Discussão crítica: idade aparente vs cronológica, limitações de anotação, riscos de inferir atributos sensíveis",
        ],
        done=[
            "Tabelas e figuras finais na documentação, prontas para o relatório",
        ],
    ),
    # 13 ----------------------------------------------------------------------
    dict(
        title="Avaliação da expressão (intra AffectNet e cross BTFER)",
        owner="B", due="11-23", milestone="M4", labels=["stage-4", "avaliacao", "expressao"], deps=[11],
        doc="docs/modulos/07-expressao.md",
        desc=(
            "Protocolo obrigatório da secção 4.5: (1) avaliação intra-dataset na partição oficial do AffectNet; "
            "(2) avaliação cross-dataset no BTFER, sem treino adicional, só nas classes comuns; "
            "(3) métricas exigidas e matriz de confusão, com discussão de limitações e generalização."
        ),
        tasks=[
            "Script `eval/eval_expression.py` que gera automaticamente todas as tabelas e figuras",
            "Intra-dataset: AffectNet, partição oficial",
            "Cross-dataset: BTFER com classes harmonizadas",
            "Métricas: accuracy, balanced accuracy, precisão/recall/F1 macro e recall por classe",
            "Matrizes de confusão normalizadas para os dois conjuntos",
            "Identificar os erros mais frequentes e mostrar exemplos visuais",
            "Discussão: ambiguidade dos rótulos, desbalanceamento, diferença de domínio entre datasets, expressão ≠ emoção",
        ],
        done=[
            "Tabelas e figuras finais na documentação, prontas para o relatório",
        ],
    ),
    # 14 ----------------------------------------------------------------------
    dict(
        title="Integração do backbone e das cabeças no pipeline (ONNX, overlay, \"incerto\")",
        owner="B", due="11-22", milestone="M2", labels=["stage-2", "sistema", "modelo"], deps=[6, 10, 11],
        doc="docs/modulos/00-arquitetura.md",
        desc=(
            "Módulo 4 na aplicação: para cada face analisável, uma passagem pelo backbone e depois as três "
            "cabeças. Mostrar idade, género e expressão no vídeo, distinguindo previsões incertas ou "
            "indisponíveis. Conclui a Etapa 2."
        ),
        tasks=[
            "`AttributeAnalyzer`: recebe `list[FaceCrop]`, faz uma passagem em lote pelo backbone e aplica as cabeças de idade, género e expressão",
            "Se o backbone escolhido for o ArcFace, partilhar o embedding com a identidade (#16), sem segunda passagem",
            "Executar os atributos a cada N frames (configurável)",
            "Estabilização simples sem tracking (ex. só limiares de confiança). Confirmar com o docente se uma associação por IoU entre frames consecutivos conta como tracking, que está excluído",
            "Overlay: identidade (placeholder até #16), intervalo de idade, \"género (rótulo do dataset)\", expressão + confiança, top-3 opcional",
            "Estilo visual diferente para previsões incertas ou indisponíveis",
            "Flags para ligar/desligar cada módulo",
            "Medir o custo de inferência por módulo",
        ],
        done=[
            "A webcam mostra idade, género e expressão para várias faces em tempo real",
            "A documentação está atualizada (00, 06, 07)",
        ],
    ),
    # 15 ----------------------------------------------------------------------
    dict(
        title="Consentimento, ética e recolha de voluntários",
        owner="B", due="11-17", milestone="M3", labels=["stage-3", "etica", "identidade"], deps=[],
        doc="docs/modulos/12-etica.md",
        desc=(
            "A galeria tem de ser explicitamente baseada em consentimento: menos de 20 voluntários, nenhum "
            "bystander, e nenhuma imagem pública usada como template (secção 13). Esta issue prepara o "
            "consentimento, a política de dados e as sessões de recolha de que o Xaneca precisa para #16 e #18."
        ),
        tasks=[
            "Formulário de consentimento informado: o que é recolhido, para quê, onde é guardado, durante quanto tempo, como apagar e direito de retirar o consentimento",
            "Política de dados: só os embeddings ficam no disco local; as imagens de enrolamento são apagadas após a avaliação; nada no git",
            "Procedimento de eliminação (pessoa individual e galeria inteira), alinhado com #16",
            "Recrutar 8–15 voluntários enrolados e alguns impostores consentidos (não enrolados) para testar \"Unknown\"",
            "Planear as sessões de captura: enrolamento, validação e teste **em sessões separadas**, com variações de pose, luz, distância e oclusão (óculos, mão, máscara)",
            "Combinar o calendário com o Xaneca (#16, #18)",
            "Escrever os riscos: atributos sensíveis, género ≠ identidade de género, expressão ≠ emoção, usos proibidos",
        ],
        done=[
            "Consentimentos assinados (guardados fora do repositório)",
            "Sessões de captura agendadas",
            "`12-etica.md` escrito",
        ],
    ),
    # 16 ----------------------------------------------------------------------
    dict(
        title="Identidade: embeddings ArcFace, galeria protegida, matching e \"Unknown\"",
        owner="A", due="11-20", milestone="M3", labels=["stage-3", "identidade", "sistema"], deps=[6, 9],
        doc="docs/modulos/08-identidade.md",
        desc=(
            "Módulo 5: identificação closed-set com menos de 20 identidades. Usa embeddings ArcFace pré-treinados "
            "(sem treinar redes de reconhecimento), templates por identidade e similaridade cosseno. Abaixo do "
            "limiar validado devolve \"Unknown\". Os embeddings são tratados como dados biométricos sensíveis. "
            "(Depende de #9 só para decidir se o embedding é partilhado com os atributos.)"
        ),
        tasks=[
            "Extrair embeddings ArcFace R50 normalizados (L2), ou reutilizar a passagem do backbone se for o ArcFace",
            "`Gallery.enrol(name, crops)`: template por média normalizada; opção de vários templates por pessoa",
            "Enrolamento guiado: capturar ~10 imagens com pose e expressão variadas e rejeitar crops de baixa qualidade",
            "`match(embedding)` → melhor similaridade cosseno; abaixo do limiar → \"Unknown\"",
            "Persistência local cifrada (ex. Fernet do `cryptography` com chave local) fora do repositório",
            "`remove(name)` e `clear()`, que apagam embeddings e quaisquer imagens associadas",
            "Testes unitários (enrolar, procurar, remover, limpar)",
        ],
        done=[
            "Identidades enroladas aparecem na webcam e pessoas não enroladas aparecem como \"Unknown\" (limiar provisório)",
        ],
    ),
    # 17 ----------------------------------------------------------------------
    dict(
        title="Benchmark de deteção no WIDER Face (SCRFD-10G vs SCRFD-500M)",
        owner="B", due="11-21", milestone="M4", labels=["stage-4", "avaliacao", "sistema"], deps=[2, 5],
        doc="docs/modulos/02-detecao.md",
        desc=(
            "A secção 4.1 pede comparar detetores em precisão, tempo de inferência e robustez a faces pequenas, "
            "pose, oclusão parcial e várias faces, usando o WIDER Face na avaliação."
        ),
        tasks=[
            "Correr SCRFD-10G e SCRFD-500M (entradas 320 e 640) no WIDER val e guardar as previsões no formato oficial",
            "Calcular AP Easy, Medium e Hard com a ferramenta oficial (ou port em Python) e as curvas PR",
            "Medir o tempo de inferência por imagem em CPU (e GPU se houver)",
            "Análise de robustez usando os atributos das anotações do WIDER (oclusão, pose, blur, escala)",
            "Exemplos qualitativos de falhas: faces pequenas, multidões, perfil",
            "(Opcional) comparar com o RetinaFace",
            "Justificar a variante e a resolução usadas na aplicação",
        ],
        done=[
            "Tabela AP × tempo e figuras na documentação",
        ],
    ),
    # 18 ----------------------------------------------------------------------
    dict(
        title="Calibração do limiar e avaliação da identificação",
        owner="A", due="11-24", milestone="M4", labels=["stage-4", "avaliacao", "identidade"], deps=[15, 16],
        doc="docs/modulos/08-identidade.md",
        desc=(
            "O difícil não é a pesquisa, é escolher um limiar que controle os falsos matches sem perder "
            "reconhecimento. O limiar **tem de ser calibrado num conjunto de validação separado**, nunca depois "
            "de ver o teste (secção 4.6)."
        ),
        tasks=[
            "Montar os conjuntos a partir das sessões de #15: enrolamento, validação (genuínos + impostores) e teste (genuínos + impostores diferentes)",
            "Histogramas das similaridades genuíno vs impostor",
            "Escolher o limiar na validação (ex. FAR ≤ 1% ou EER) e fixá-lo antes de abrir o teste",
            "No teste: Rank-1, FAR, FRR, taxa de rejeição correta de \"Unknown\" e curva de sensibilidade ao limiar",
            "Ablação: número de imagens de enrolamento (1, 3, 5, 10) e diversidade",
            "Ablação: tamanho da face (distância), pose, iluminação e oclusão",
        ],
        done=[
            "Limiar final em `configs/default.yaml`",
            "Tabelas e figuras na documentação",
        ],
    ),
    # 19 ----------------------------------------------------------------------
    dict(
        title="Interface gráfica (PySide6)",
        owner="B", due="11-23", milestone="M4", labels=["stage-4", "sistema"], deps=[14, 16],
        doc="docs/modulos/09-interface.md",
        desc=(
            "Módulo 6: vídeo anotado com uma caixa por face (identidade ou \"Unknown\", com idade, género e "
            "expressão como anotações secundárias) e painel de controlo. Requisitos das secções 2.1 e 7."
        ),
        tasks=[
            "Janela PySide6 com o vídeo e um painel lateral",
            "Controlos: iniciar/parar a câmara e escolher a fonte (índice de webcam ou ficheiro)",
            "Ligar/desligar cada módulo: deteção, idade/género, expressão, identidade",
            "Galeria: enrolar (nome + captura guiada), listar, apagar pessoa e limpar a galeria, com confirmação",
            "Painel de estado: FPS, latência, número de faces e módulos ativos (dados vindos de #20)",
            "Processamento numa thread (`QThread`) para a interface não bloquear",
            "Aviso visível de que é uma demonstração académica, sem uso para vigilância ou decisões",
            "Screenshots em `docs/figs/`",
        ],
        done=[
            "Todas as funções da secção 2.1 funcionam através da interface",
        ],
    ),
    # 20 ----------------------------------------------------------------------
    dict(
        title="Monitorização de desempenho e otimização",
        owner="A", due="11-25", milestone="M4", labels=["stage-4", "sistema", "avaliacao"], deps=[19],
        doc="docs/modulos/10-desempenho.md",
        desc=(
            "Módulo 7 e Etapa 4: medir FPS, latência fim-a-fim, memória e utilização de CPU/GPU, e afinar a "
            "aplicação para o portátil (frequência dos modelos e resolução)."
        ),
        tasks=[
            "Medir o tempo por módulo: captura, deteção, alinhamento, backbone, cabeças, identidade e render",
            "Medir latência fim-a-fim, FPS, CPU%, RAM e GPU (`pynvml`, se houver)",
            "Mostrar as métricas no painel (#19) e gravá-las em CSV",
            "Experiências: resolução de deteção (320/480/640), variante SCRFD, frequência N dos atributos, lote de faces, threads do ONNX Runtime",
            "Escalabilidade com 1, 2 e 4 faces",
            "Tabela antes/depois, gráficos e configuração final recomendada",
        ],
        done=[
            "Configuração final definida",
            "Resultados e gráficos na documentação",
        ],
    ),
    # 21 ----------------------------------------------------------------------
    dict(
        title="Testes de sistema obrigatórios (secção 6.2)",
        owner="B", due="11-26", milestone="M4", labels=["stage-4", "avaliacao", "sistema"], deps=[18, 19, 20],
        doc="docs/modulos/11-testes-sistema.md",
        desc=(
            "Executar e registar todos os cenários obrigatórios da secção 6.2 com a aplicação final, usando os "
            "vídeos gravados (repetível) e testes ao vivo."
        ),
        tasks=[
            "Checklist com critérios de sucesso para cada cenário",
            "Uma pessoa e várias pessoas na cena",
            "Pessoas a entrar e a sair do campo de visão",
            "Variações de pose, distância à câmara e iluminação",
            "Duas ou mais pessoas enroladas em simultâneo",
            "Pessoa não enrolada → tem de aparecer \"Unknown\", nunca um match forçado",
            "Tabela de resultados com screenshots e métricas de #20",
            "Abrir issues de bug para cada falha encontrada",
        ],
        done=[
            "Todos os cenários testados e documentados",
            "Bugs críticos resolvidos ou documentados como limitações",
        ],
    ),
    # 22 ----------------------------------------------------------------------
    dict(
        title="README, licenças e configuração de exemplo",
        owner="A", due="11-27", milestone="M5", labels=["stage-5", "docs"], deps=[19],
        doc="README.md",
        desc=(
            "Entregável da secção 10: código organizado com README que descreva instalação, dependências, "
            "obtenção dos modelos, execução e limitações; referências e licenças dos modelos, sem redistribuir pesos."
        ),
        tasks=[
            "Instalação e dependências",
            "Obtenção dos modelos (scripts de download + links + licença de cada modelo)",
            "Obtenção e preparação dos datasets (scripts)",
            "Reprodução: pré-processamento, extração de features, treino das MLPs e avaliações",
            "Execução da aplicação e configuração de exemplo comentada",
            "Limitações conhecidas e secção de ética e uso responsável",
            "Tabela de licenças (modelos e datasets)",
            "O André testa a instalação do zero num ambiente limpo",
            "Confirmar que o repositório não contém pesos, dados pessoais nem galeria",
        ],
        done=[
            "Um terceiro consegue instalar e correr a aplicação só com o README",
        ],
    ),
    # 23 ----------------------------------------------------------------------
    dict(
        title="Relatório técnico",
        owner="AB", due="11-28", milestone="M5", labels=["stage-5", "docs"], deps=[12, 13, 17, 18, 20, 21],
        doc="relatório (PDF)",
        desc=(
            "Relatório com escolhas de arquitetura, fontes dos modelos, protocolo experimental, resultados, "
            "perfil de desempenho, limitações e considerações éticas. É construído a partir de `docs/modulos/`, "
            "que já deve estar preenchido."
        ),
        tasks=[
            "Estrutura: introdução, arquitetura, deteção, alinhamento, backbone, idade/género, expressão, identidade, aplicação, desempenho, testes de sistema, ética, conclusão, referências",
            "[Xaneca] Alinhamento, idade/género, identidade, desempenho",
            "[André] Deteção, expressão, aplicação/interface, testes de sistema, ética",
            "[Ambos] Introdução, backbone, conclusão",
            "Uniformizar figuras e tabelas (estilo, numeração, legendas)",
            "Revisão cruzada: cada um revê as secções do outro",
            "Exportar para PDF",
        ],
        done=[
            "PDF final revisto pelos dois membros",
        ],
    ),
    # 24 ----------------------------------------------------------------------
    dict(
        title="Vídeo de demonstração e submissão final",
        owner="AB", due="11-29", milestone="M5", labels=["stage-5", "docs"], deps=[21, 22, 23],
        doc="—",
        desc=(
            "Vídeo curto (ou demonstração ao vivo) com várias faces, overlays de atributos, matches de "
            "identidade e rejeição \"Unknown\". Fecho e submissão do projeto."
        ),
        tasks=[
            "Guião de 2–3 min: várias faces, overlays, enrolar/apagar, matches, \"Unknown\" e painel de FPS",
            "Gravar só com voluntários que consentiram",
            "Criar a tag de release `v1.0`",
            "Verificação final: README, licenças, nenhum dado pessoal nem pesos no repositório",
            "Submeter o relatório, o código e o vídeo",
            "Apagar a galeria e as imagens de enrolamento após a avaliação, conforme o consentimento",
        ],
        done=[
            "Projeto submetido até 29 nov",
        ],
    ),
]


def owner_names(code):
    return " + ".join(MEMBERS[c] for c in code)


def build_body(i, issue, ref):
    """ref(n) -> texto para a tarefa local n (ex. '#12' ou 'Tarefa 12')."""
    deps = issue["deps"]
    dep_txt = ", ".join(ref(d) for d in deps) if deps else "—"
    blocks = [n for n, it in enumerate(ISSUES, 1) if i in it["deps"]]
    blocks_txt = ", ".join(ref(b) for b in blocks) if blocks else "—"
    tasks = list(issue["tasks"])
    if issue["doc"].startswith("docs/"):
        tasks.append(DOC_NOTE.format(doc=issue["doc"]))
    lines = [
        f"**Responsável:** {owner_names(issue['owner'])} · **Prazo:** {YEAR}-{issue['due']} · "
        f"**Etapa:** {MILESTONES[issue['milestone']][0]}",
        "",
        "## Descrição",
        issue["desc"],
        "",
        "## Tarefas",
        *[f"- [ ] {t}" for t in tasks],
        "",
        "## Critérios de conclusão",
        *[f"- {d}" for d in issue["done"]],
        "",
        "## Dependências",
        f"- **Bloqueada por:** {dep_txt}",
        f"- **Bloqueia:** {blocks_txt}",
        "",
        f"_Tarefa {i} de 24 do plano do Assignment 3._",
    ]
    return "\n".join(lines)


def gh_api(path, method="GET", payload=None, check=True):
    cmd = ["gh", "api", "-X", method, path, "-H", "Accept: application/vnd.github+json"]
    if payload is not None:
        cmd += ["--input", "-"]
    r = subprocess.run(cmd, input=json.dumps(payload) if payload is not None else None,
                       capture_output=True, text=True)
    if r.returncode != 0:
        if check:
            sys.exit(f"Erro em gh api {method} {path}:\n{r.stderr or r.stdout}")
        return None
    return json.loads(r.stdout) if r.stdout.strip() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Xaneca/IAVC_proj_3")
    ap.add_argument("--a-user", default="Xaneca", help="login GitHub do membro A (idade/género)")
    ap.add_argument("--b-user", default=None, help="login GitHub do membro B (expressão)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        out = ["# Pré-visualização das issues – Assignment 3\n"]
        for i, it in enumerate(ISSUES, 1):
            out.append(f"\n---\n\n# Tarefa {i} · {it['title']}\n")
            out.append(f"Labels: `{'`, `'.join(it['labels'])}` · Milestone: {MILESTONES[it['milestone']][0]}\n")
            out.append(build_body(i, it, lambda n: f"Tarefa {n}") + "\n")
        with open("issues_preview.md", "w", encoding="utf-8") as f:
            f.write("\n".join(out))
        print("Gerado issues_preview.md")
        return

    repo = args.repo
    logins = {"A": args.a_user, "B": args.b_user}

    # Labels
    for name, (color, desc) in LABELS.items():
        subprocess.run(["gh", "label", "create", name, "--repo", repo, "--color", color,
                        "--description", desc, "--force"], capture_output=True)
    print("Labels prontas")

    # Milestones
    existing = {m["title"]: m["number"] for m in gh_api(f"repos/{repo}/milestones?state=all&per_page=100")}
    ms_num = {}
    for key, (title, due, desc) in MILESTONES.items():
        if title in existing:
            ms_num[key] = existing[title]
        else:
            m = gh_api(f"repos/{repo}/milestones", "POST",
                       {"title": title, "due_on": f"{due}T23:59:59Z", "description": desc})
            ms_num[key] = m["number"]
    print("Milestones prontas")

    # Issues (criadas por ordem: todas as dependências apontam para issues anteriores)
    nummap, idmap = {}, {}
    for i, it in enumerate(ISSUES, 1):
        assignees = [logins[c] for c in it["owner"] if logins.get(c)]
        payload = {
            "title": it["title"],
            "body": build_body(i, it, lambda n: f"#{nummap[n]}" if n in nummap else f"Tarefa {n}"),
            "labels": it["labels"],
            "milestone": ms_num[it["milestone"]],
            "assignees": assignees,
        }
        res = gh_api(f"repos/{repo}/issues", "POST", payload)
        nummap[i], idmap[i] = res["number"], res["id"]
        print(f"  Tarefa {i:>2} → #{res['number']}  {it['title']}")

    # Segunda passagem: todas as referências passam a números reais; dependências nativas do GitHub
    for i, it in enumerate(ISSUES, 1):
        gh_api(f"repos/{repo}/issues/{nummap[i]}", "PATCH",
               {"body": build_body(i, it, lambda n: f"#{nummap[n]}")})
        for d in it["deps"]:
            gh_api(f"repos/{repo}/issues/{nummap[i]}/dependencies/blocked_by", "POST",
                   {"issue_id": idmap[d]}, check=False)
    print("Dependências ligadas. Concluído.")


if __name__ == "__main__":
    main()
