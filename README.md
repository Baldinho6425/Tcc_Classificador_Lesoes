# 🧠 Classificação de Lesões de Pele com Deep Learning

Este projeto aplica **Redes Neurais Convolucionais (CNNs)** baseadas na arquitetura **VGG16** para classificar imagens dermatológicas do conjunto **HAM10000**.  
O objetivo é **auxiliar o diagnóstico automatizado** de diferentes tipos de lesões de pele, contribuindo para o suporte clínico em dermatologia.

---

## 📂 Estrutura do Projeto

```
📦 prototipo_classificacao_lesoes
├── data/
│   ├── raw/              # Dados brutos (originais do HAM10000)
│   └── processed/        # Dados tratados e redimensionados
├── models/               # Modelos treinados (.h5) e métricas
├── src/
│   ├── training/         # Script principal de treinamento (train.py)
│   ├── preprocessing/    # Pré-processamento e augmentação de dados
│   └── evaluation/       # Avaliação do modelo e geração de métricas
├── notebooks/            # Notebooks Jupyter para experimentos
├── reports/figures/      # Gráficos e visualizações
├── logs/                 # Registros de execução
├── outputs/              # Predições e resultados finais
├── requirements.txt      # Dependências do projeto
└── setup_project.py      # Script mestre para criação da estrutura
```

---

## ⚙️ Instalação e Configuração

### 1️⃣ Clone o repositório

```bash
git clone https://github.com/SEU_USUARIO/prototipo_classificacao_lesoes.git
cd prototipo_classificacao_lesoes
```

### 2️⃣ Crie o ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

### 3️⃣ Instale as dependências

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Gere a estrutura do projeto

Antes de baixar o dataset, crie automaticamente todas as pastas necessárias:

```bash
python3 setup_project.py
```

Isso criará os diretórios `data/`, `models/`, `reports/`, `logs/`, entre outros.

---

### 5️⃣ Baixe o dataset **HAM10000**

O dataset pode ser obtido no **Kaggle**:  
👉 [HAM10000 - Skin Lesion Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

**Passo a passo:**
1. Crie uma conta no Kaggle (se ainda não tiver).  
2. Faça o download do dataset.  
3. Extraia o conteúdo do `.zip`, que contém:
   ```
   HAM10000_images_part_1/
   HAM10000_images_part_2/
   HAM10000_metadata.csv
   ```
4. Copie esses arquivos para o diretório:
   ```
   prototipo_classificacao_lesoes/data/raw/
   ```

**Estrutura esperada:**
```
data/raw/
├── HAM10000_images_part_1/
├── HAM10000_images_part_2/
└── HAM10000_metadata.csv
```

**Verifique a cópia:**
```bash
ls data/raw
# Deve exibir:
# HAM10000_images_part_1  HAM10000_images_part_2  HAM10000_metadata.csv
```

---

## 🧠 Treinamento do Modelo

O script principal está em `src/training/train.py`.

Para iniciar o treinamento:

```bash
python3 src/training/train.py
```

Durante o processo:
- A rede **VGG16** é carregada com pesos pré-treinados do **ImageNet**.  
- O modelo passa por **duas etapas de treinamento**:  
  1. Treino da cabeça densa (camadas superiores)  
  2. **Fine-tuning** das últimas camadas convolucionais  
- O melhor modelo é salvo automaticamente em:
  ```
  models/modelo_cnn.h5
  ```

**Saídas geradas:**
- `models/grafico_acuracia.png`
- `models/grafico_loss.png`
- `models/acuracia_final.txt`

---

## 📊 Resultados e Métricas

Após o treinamento, o projeto gera automaticamente:

| Arquivo | Descrição |
|----------|------------|
| `models/modelo_cnn.h5` | Modelo treinado final |
| `models/acuracia_final.txt` | Resultados numéricos (acurácia, loss, validação) |
| `models/grafico_acuracia.png` | Evolução da acurácia |
| `models/grafico_loss.png` | Evolução da perda (loss) |

---

## ⚡ Requisitos de Hardware

| Recurso | Recomendado |
|----------|--------------|
| GPU | NVIDIA compatível com CUDA |
| TensorFlow | ≥ 2.17 |
| RAM | ≥ 8 GB |
| Python | ≥ 3.9 |

> 💡 Caso não haja GPU disponível, o script detecta automaticamente e utiliza a CPU para o treinamento.

---

## 👨‍💻 Créditos

Desenvolvido por:
- **Eduardo Giehl**  
- **Eduardo Tessaro**  

Projeto acadêmico de **Classificação de Lesões de Pele utilizando Inteligência Artificial**  
Universitário • 2025

---

## 🚀 Melhorias Futuras

- Implementar arquiteturas **EfficientNet** e **ResNet50**  
- Adicionar explicabilidade via **Grad-CAM**  
- Desenvolver uma **interface web interativa** para upload e classificação de imagens  
