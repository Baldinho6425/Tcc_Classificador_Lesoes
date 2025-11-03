# 🧠 Classificação de Lesões de Pele com Deep Learning

Este projeto utiliza **Redes Neurais Convolucionais (CNNs)** para classificar imagens de lesões de pele, com base na base de dados **HAM10000**.  
O objetivo é auxiliar o diagnóstico precoce de doenças dermatológicas, explorando técnicas modernas de **transfer learning** com o modelo **VGG16**.

---

## 📂 Estrutura do Projeto

```bash
prototipo_classificacao_lesoes/
├── data/
│   ├── raw/               # Dados brutos originais
│   └── processed/         # Dados pré-processados (redimensionados, limpos, balanceados)
│
├── models/                # Modelos treinados e checkpoints (.h5, txt, etc.)
│
├── notebooks/             # Notebooks de experimentação e análise
│
├── reports/
│   └── figures/           # Gráficos e visualizações de resultados
│
├── src/
│   ├── preprocessing/     # Scripts de pré-processamento das imagens
│   ├── training/          # Código principal de treinamento (train.py)
│   └── evaluation/        # Scripts de avaliação e métricas
│
├── logs/                  # Logs de execução e histórico de treinamento
├── outputs/               # Predições, relatórios e resultados finais
├── requirements.txt       # Dependências do projeto
└── README.md              # Este arquivo
