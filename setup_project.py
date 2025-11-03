# setup_project.py
import os

def criar_estrutura_projeto():
    """
    Cria toda a estrutura de diretórios e arquivos padrão
    para o projeto de classificação de lesões.
    """
    estrutura = [
        "data/raw",
        "data/processed",
        "models",
        "src/training",
        "src/preprocessing",
        "src/evaluation",
        "notebooks",
        "reports/figures",
        "logs",
        "outputs",
    ]

    for pasta in estrutura:
        os.makedirs(pasta, exist_ok=True)
        print(f"📁 Pasta criada: {pasta}")

    # =============================
    # Criação de arquivos base
    # =============================
    arquivos = {
        "README.md": "# 🧠 Projeto de Classificação de Lesões de Pele\n\n"
                     "Este projeto utiliza redes neurais convolucionais (CNNs) para classificar imagens de lesões de pele.\n"
                     "A base de dados utilizada é a **HAM10000**, e o modelo é baseado na arquitetura **VGG16**.\n\n"
                     "## Estrutura do Projeto\n"
                     "- `data/raw`: dados brutos originais\n"
                     "- `data/processed`: imagens processadas\n"
                     "- `models`: modelos treinados e checkpoints\n"
                     "- `src/`: scripts de código-fonte (treino, avaliação e pré-processamento)\n"
                     "- `reports/`: gráficos e resultados visuais\n"
                     "- `logs/`: registros de execução\n"
                     "- `outputs/`: resultados finais e predições\n",
        
        "requirements.txt": "tensorflow>=2.17.0\n"
                            "matplotlib\n"
                            "numpy\n"
                            "pandas\n"
                            "scikit-learn\n"
                            "opencv-python\n"
                            "jupyter\n",

        "src/training/train.py": "# Treinamento principal do modelo (adicione aqui seu código de treino)\n",
        "src/preprocessing/preprocess.py": "# Script de pré-processamento das imagens\n",
        "src/evaluation/evaluate.py": "# Script para avaliação do modelo\n",
    }

    for caminho, conteudo in arquivos.items():
        pasta = os.path.dirname(caminho)
        if pasta:
            os.makedirs(pasta, exist_ok=True)
        with open(caminho, "w", encoding="utf-8") as f:
            f.write(conteudo)
        print(f"📝 Arquivo criado: {caminho}")

    print("\n✅ Estrutura completa do projeto criada com sucesso!")

if __name__ == "__main__":
    criar_estrutura_projeto()
