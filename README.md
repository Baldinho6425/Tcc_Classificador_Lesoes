# Classificação de Lesões de Pele com Deep Learning

Trabalho de Conclusão de Curso — aplicação de **Redes Neurais Convolucionais (CNNs)** para classificação automática de imagens dermatológicas do dataset **HAM10000**.  
Foram treinadas e comparadas duas arquiteturas: **VGG16** e **MobileNetV2**, ambas com pesos pré-treinados no ImageNet via *transfer learning*.

---

## Contexto

Lesões de pele representam um dos tipos de câncer mais comuns no mundo. O diagnóstico precoce aumenta significativamente as chances de tratamento bem-sucedido. Este projeto explora o uso de visão computacional e aprendizado profundo para apoiar o diagnóstico dermatológico, classificando imagens em sete categorias de lesões.

---

## Classes do Dataset (HAM10000)

O dataset **HAM10000** contém 10.015 imagens dermatoscópicas de 7 tipos de lesões:

| Código | Classe | Descrição |
|--------|--------|-----------|
| `nv` | Melanocytic Nevi | Nevos melanocíticos (pintas benignas) — classe majoritária |
| `mel` | Melanoma | Melanoma maligno — tipo mais grave de câncer de pele |
| `bkl` | Benign Keratosis | Ceratose benigna (seborreica, lentigo solar) |
| `bcc` | Basal Cell Carcinoma | Carcinoma basocelular — câncer de pele mais comum |
| `akiec` | Actinic Keratoses | Ceratose actínica / carcinoma intraepitelial |
| `df` | Dermatofibroma | Dermatofibroma — lesão benigna do tecido fibroso |
| `vasc` | Vascular Lesions | Lesões vasculares (angiomas, granuloma piogênico) |

> O dataset apresenta forte desbalanceamento: `melanocytic_nevi` representa ~67% das amostras.

---

## Estrutura do Repositório

```
Tcc_Classificador_Lesoes-1/
├── Projeto/
│   ├── menu.py                          # Menu interativo para executar o treinamento
│   ├── predict.py                       # [BETA] Inferência em nova imagem com Grad-CAM
│   ├── src/
│   │   └── training/
│   │       └── train.py                 # Pipeline principal de treinamento
│   ├── scripts/
│   │   └── save/
│   │       ├── organizar_ham10000.py    # Organiza imagens por classe em subpastas
│   │       └── gerar_amostra_classes.py # Gera grade visual com exemplos por classe
│   └── results/                         # Gerado automaticamente pelo pipeline
│       ├── metrics_{modelo}.json        # Métricas por modelo (vgg16, mobilenetv2...)
│       ├── resumo_modelos.json          # Comparativo geral entre todos os modelos
│       ├── confusion_matrices/          # Matrizes de confusão normalizadas (%)
│       ├── plots/                       # Curvas de acurácia e loss por época
│       └── gradcam/                     # Visualizações Grad-CAM — predict.py [BETA]
├── data/                                # Dataset (não versionado — ver .gitignore)
│   ├── raw/                             # Imagens brutas + HAM10000_metadata.csv
│   └── processed/                       # Imagens organizadas por classe e split
├── models/                              # Modelos treinados .h5 (não versionados)
├── requirements.txt
├── setup_project.py                     # Cria estrutura de diretórios automaticamente
└── README.md
```

---

## Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/SEU_USUARIO/Tcc_Classificador_Lesoes.git
cd Tcc_Classificador_Lesoes
```

### 2. Crie e ative o ambiente virtual

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Gere a estrutura de diretórios

```bash
python setup_project.py
```

---

## Dataset — HAM10000

Baixe o dataset no Kaggle:  
[HAM10000 - Skin Lesion Analysis Toward Melanoma Detection](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

Após o download, extraia o conteúdo e organize assim:

```
data/raw/
├── Images/               # Todas as imagens (HAM10000_images_part_1 + part_2 unificados)
└── HAM10000_metadata.csv
```

Em seguida, execute o script de organização para separar as imagens por classe e conjunto:

```bash
python Projeto/scripts/save/organizar_ham10000.py
```

O script realiza uma divisão estratificada **80% treino / 20% validação** por `lesion_id` (não por imagem), garantindo que fotos da mesma lesão nunca apareçam nos dois conjuntos — o que inflaria artificialmente as métricas.

Estrutura gerada:

```
data/processed/
├── train/
│   ├── actinic_keratoses/
│   ├── melanoma/
│   └── ...
└── val/
    ├── actinic_keratoses/
    └── ...
```

---

## Treinamento

### Via menu interativo (recomendado)

```bash
python Projeto/menu.py
```

Opções disponíveis:

```
--- Treinamento ---
1 - Treinar somente MobileNetV2
2 - Treinar somente VGG16
3 - Treinar somente EfficientNetB0
4 - Treinar com fine-tuning (10 + 10 épocas)
5 - Treinar com LR reduzido (20 épocas)
6 - Rodar pacote completo (VGG16 + MobileNetV2 + EfficientNetB0)
7 - Executar treinamento padrão (VGG16 + MobileNetV2)

--- Inferência ---
8 - Classificar nova imagem (+ Grad-CAM)

0 - Sair
```

### Via script direto

```bash
python Projeto/src/training/train.py
```

### Reprodutibilidade

A seed `42` é fixada automaticamente no início do treinamento (Python, NumPy e TensorFlow), garantindo resultados idênticos entre execuções.

### Pipeline de treinamento

O treinamento é dividido em duas fases:

**Fase 1 — Baseline (backbone congelado)**
- O backbone pré-treinado (ImageNet) permanece congelado
- Apenas a "cabeça" densa é treinada
- Learning rate: `1e-4`
- Salva o melhor checkpoint como `{modelo}_best_baseline.h5`

**Fase 2 — Fine-tuning (opcional, backbone descongelado)**
- Todas as camadas são liberadas para atualização
- Learning rate reduzido: `1e-5`
- Salva o melhor checkpoint como `{modelo}_best_finetune.h5`

### Data Augmentation aplicada

| Transformação | Valor |
|---------------|-------|
| Rotação | até 20° |
| Deslocamento horizontal | 10% |
| Deslocamento vertical | 10% |
| Zoom | 10% |
| Flip horizontal | sim |
| Flip vertical | sim |

---

## Resultados

Métricas calculadas no conjunto de validação (20% do dataset), com média ponderada pelas classes:

### VGG16

| Métrica | Valor |
|---------|-------|
| Acurácia | **75,00%** |
| Precisão (weighted) | 71,22% |
| Recall (weighted) | 75,00% |
| F1-Score (weighted) | 71,66% |
| Especificidade (macro) | 91,78% |

**Especificidade por classe (VGG16):**

| Classe | Especificidade |
|--------|---------------|
| Actinic Keratoses | 99,59% |
| Basal Cell Carcinoma | 98,89% |
| Benign Keratosis | 94,78% |
| Dermatofibroma | 99,90% |
| Melanocytic Nevi | 52,96% |
| Melanoma | 96,51% |
| Vascular Lesions | 99,80% |

---

### MobileNetV2

| Métrica | Valor |
|---------|-------|
| Acurácia | **76,85%** |
| Precisão (weighted) | 76,37% |
| Recall (weighted) | 76,85% |
| F1-Score (weighted) | 75,42% |
| Especificidade (macro) | 93,80% |

**Especificidade por classe (MobileNetV2):**

| Classe | Especificidade |
|--------|---------------|
| Actinic Keratoses | 99,43% |
| Basal Cell Carcinoma | 98,79% |
| Benign Keratosis | 90,62% |
| Dermatofibroma | **100,00%** |
| Melanocytic Nevi | 72,08% |
| Melanoma | 96,23% |
| Vascular Lesions | 99,44% |

---

### Comparativo Geral

| Modelo | Acurácia | F1-Score | Especificidade |
|--------|----------|----------|----------------|
| VGG16 | 75,00% | 71,66% | 91,78% |
| **MobileNetV2** | **76,85%** | **75,42%** | **93,80%** |

**MobileNetV2 obteve os melhores resultados em todas as métricas**, sendo também mais eficiente computacionalmente que a VGG16 (menos parâmetros, menor tempo de inferência).

> A baixa especificidade na classe `melanocytic_nevi` em ambos os modelos reflete o desbalanceamento severo do dataset — a grande quantidade de amostras dessa classe leva o modelo a classificar outras lesões como névos com maior frequência.

---

## Inferência e Grad-CAM (BETA)

Para classificar uma nova imagem com um modelo já treinado:

```bash
python Projeto/predict.py \
    --image caminho/para/lesao.jpg \
    --model models/mobilenetv2_best_baseline.h5 \
    --arch mobilenetv2
```

O script exibe as probabilidades para cada classe e salva automaticamente uma figura com a imagem original e o mapa **Grad-CAM** (Gradient-weighted Class Activation Mapping), que destaca as regiões da imagem que influenciaram a decisão do modelo.

| Argumento | Obrigatório | Descrição |
|-----------|-------------|-----------|
| `--image` / `-i` | Sim | Caminho para a imagem de entrada (`.jpg` ou `.png`) |
| `--model` / `-m` | Sim | Caminho para o modelo treinado (`.h5`) |
| `--arch` / `-a` | Sim | Arquitetura: `vgg16`, `mobilenetv2` ou `efficientnetb0` |
| `--output-dir` | Não | Diretório de saída (padrão: `results/gradcam`) |

Também disponível via menu interativo, opção **8**.

> **BETA:** a inferência com Grad-CAM depende de um modelo previamente treinado (`.h5`). Execute o treinamento antes de usar esta funcionalidade.

---

## Saídas Geradas

Após o treinamento, os seguintes arquivos são criados em `Projeto/results/`:

| Arquivo | Descrição |
|---------|-----------|
| `metrics_{modelo}.json` | Métricas finais no conjunto de validação |
| `resumo_modelos.json` | Comparativo geral entre todos os modelos |
| `confusion_matrices/cm_{modelo}_percent.png` | Matriz de confusão normalizada (%) |
| `plots/accuracy_{modelo}.png` | Curva de acurácia por época |
| `plots/loss_{modelo}.png` | Curva de loss por época |
| `gradcam/gradcam_{imagem}_{modelo}.png` | Visualização Grad-CAM da inferência |
| `amostras_classes.png` | Grade visual com exemplos de cada classe |

---

## Requisitos de Hardware

| Recurso | Mínimo | Recomendado |
|---------|--------|-------------|
| Python | 3.9+ | 3.11 |
| TensorFlow | 2.17+ | 2.17+ |
| RAM | 8 GB | 16 GB |
| GPU | — | NVIDIA com CUDA (Mixed Precision habilitado automaticamente) |
| Armazenamento | 5 GB | 10 GB |

### Hardware utilizado nos experimentos

| Componente | Especificações |
|------------|---------------|
| CPU | Intel Core i7-10700F |
| Memória RAM | 32 GB DDR4 |
| GPU | NVIDIA GeForce RTX 3050 Pegasus |
| VRAM | 8 GB GDDR6 |

> Sem GPU, o script detecta automaticamente e executa na CPU. O tempo de treinamento será significativamente maior.

> **Ambiente de desenvolvimento:** este projeto foi desenvolvido e treinado no **Ubuntu LTS**. No Windows 11, foram identificados problemas na detecção da GPU pelo TensorFlow. Caso enfrente o mesmo problema no Windows, recomenda-se utilizar **Ubuntu** ou **WSL2** (Windows Subsystem for Linux 2) com os drivers NVIDIA e CUDA instalados corretamente.

---

## Dependências

```
tensorflow>=2.17.0
matplotlib
seaborn
numpy
pandas
scikit-learn
opencv-python
jupyter
```

---

## Melhorias Possíveis

- Aplicar **oversampling sintético** (SMOTE ou geração via GANs) para ampliar as classes minoritárias além do balanceamento por pesos
- Desenvolver uma **interface web** (Streamlit ou Flask) para upload de imagens e visualização do Grad-CAM em tempo real
- Avaliar o impacto de diferentes estratégias de **fine-tuning parcial** (congelar apenas as primeiras N camadas do backbone)
- Experimentar arquiteturas mais recentes como **EfficientNetV2** ou **ConvNeXt**

---

## Autores

**Eduardo Giehl** e **Eduardo Tessaro**

Trabalho de Conclusão de Curso — Classificação de Lesões de Pele com Inteligência Artificial  
2025
