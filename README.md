# 🧠 Classificação de Lesões de Pele com Deep Learning

Este projeto utiliza **Redes Neurais Convolucionais (CNNs)** com a arquitetura **VGG16** para classificar imagens de lesões de pele da base **HAM10000**.  
O objetivo é auxiliar na identificação automatizada de diferentes tipos de lesões dermatológicas.

---

## 📂 Estrutura do Projeto

```
📦 prototipo_classificacao_lesoes
├── data/
│   ├── raw/              # Dados brutos (originais)
│   └── processed/        # Dados processados (imagens redimensionadas)
├── models/               # Modelos treinados (.h5)
├── src/
│   ├── training/         # Script principal de treinamento (train.py)
│   ├── preprocessing/    # Pré-processamento de dados
│   └── evaluation/       # Avaliação e métricas
├── notebooks/            # Jupyter notebooks
├── reports/figures/      # Gráficos e resultados
├── logs/                 # Logs de execução
├── outputs/              # Resultados e predições
├── requirements.txt      # Dependências do projeto
└── setup_project.py      # Script mestre para gerar toda a estrutura
```

---

## ⚙️ Instalação e Configuração

### 1. Clone o repositório

```bash
git clone https://github.com/SEU_USUARIO/prototipo_classificacao_lesoes.git
cd prototipo_classificacao_lesoes
```

### 2. Crie o ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

---

## 🧩 Estrutura inicial do projeto

Antes de iniciar o treinamento, gere automaticamente toda a estrutura de pastas executando:

```bash
python3 setup_project.py
```

Isso criará todas as pastas necessárias para dados, modelos e relatórios.

---

## 🧠 Treinamento do Modelo

O script principal está localizado em `src/training/train.py`.

Para iniciar o treinamento:

```bash
python3 src/training/train.py
```

Durante o processo:
- O modelo base **VGG16** é carregado com pesos do ImageNet.  
- O treinamento ocorre em duas fases: **cabeça da rede** e **fine-tuning**.  
- O melhor modelo é salvo automaticamente em `models/modelo_cnn.h5`.

Gráficos de **acurácia** e **loss** serão salvos em `models/grafico_acuracia.png` e `models/grafico_loss.png`.

---

## 📊 Resultados e Métricas

Após o treinamento, os seguintes arquivos são gerados:

- `models/modelo_cnn.h5` → modelo final treinado.  
- `models/acuracia_final.txt` → resultados numéricos de treino e validação.  
- `models/grafico_acuracia.png` → gráfico da acurácia.  
- `models/grafico_loss.png` → gráfico da perda (loss).

---

## 🖥️ Requisitos de Hardware

- GPU NVIDIA compatível com CUDA (recomendado).  
- TensorFlow 2.17+  
- 8 GB de RAM (mínimo recomendado).  
- Python 3.9+  

Caso não haja GPU, o treinamento é executado automaticamente na CPU.

---

## 💡 Créditos

Desenvolvido por **Eduardo Giehl e Eduardo Tessaro**  
Projeto acadêmico de classificação de lesões de pele utilizando **Inteligência Artificial** e **Visão Computacional**.

---

## 🧩 Sugestões futuras

- Implementar **EfficientNet** ou **ResNet50**.  
- Adicionar **visualização Grad-CAM** para interpretação do modelo.  
- Criar uma **interface web** para upload e classificação de imagens.
