import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Caminhos (relativos ao diretório Projeto/scripts/save/)
CSV_PATH = '../../data/raw/HAM10000_metadata.csv'
IMG_DIR  = '../../data/raw/Images'
DEST_DIR = '../../data/processed'

SEED = 42

LABEL_MAP = {
    'nv':    'melanocytic_nevi',
    'mel':   'melanoma',
    'bkl':   'benign_keratosis',
    'bcc':   'basal_cell_carcinoma',
    'akiec': 'actinic_keratoses',
    'df':    'dermatofibroma',
    'vasc':  'vascular_lesions',
}

SPLITS = ['train', 'val']


def criar_pastas():
    for split in SPLITS:
        for label in LABEL_MAP.values():
            os.makedirs(os.path.join(DEST_DIR, split, label), exist_ok=True)


def carregar_metadata():
    df = pd.read_csv(CSV_PATH)
    df['label'] = df['dx'].map(LABEL_MAP)
    return df


def split_por_lesion_id(df):
    """
    Divide por lesion_id (não por imagem) para evitar data leakage:
    todas as fotos de uma mesma lesão ficam no mesmo conjunto.
    Proporção: 80% treino / 20% validação, estratificado por classe.
    """
    lesoes_unicas = df.drop_duplicates(subset='lesion_id')[['lesion_id', 'label']]

    treino, val = train_test_split(
        lesoes_unicas,
        test_size=0.20,
        stratify=lesoes_unicas['label'],
        random_state=SEED
    )

    mapa_split = {}
    for lesion_id in treino['lesion_id']:
        mapa_split[lesion_id] = 'train'
    for lesion_id in val['lesion_id']:
        mapa_split[lesion_id] = 'val'

    return mapa_split, treino, val


def copiar_imagens(df, mapa_split):
    copiadas = 0
    nao_encontradas = 0

    for _, row in df.iterrows():
        image_id  = row['image_id']
        lesion_id = row['lesion_id']
        label     = row['label']
        split     = mapa_split.get(lesion_id)

        src = os.path.join(IMG_DIR, image_id + '.jpg')
        dst = os.path.join(DEST_DIR, split, label, image_id + '.jpg')

        if os.path.exists(src):
            shutil.copy2(src, dst)
            copiadas += 1
        else:
            print(f"Imagem não encontrada: {src}")
            nao_encontradas += 1

    return copiadas, nao_encontradas


def imprimir_resumo(df, treino, val, mapa_split):
    print("\n📊 Distribuição de lesões únicas por conjunto:")
    print(f"   Treino : {len(treino):>5} lesões (80%)")
    print(f"   Val    : {len(val):>5} lesões (20%)")

    print("\n📊 Distribuição de imagens por conjunto e classe:")
    df['split'] = df['lesion_id'].map(mapa_split)
    tabela = df.groupby(['split', 'label']).size().unstack(fill_value=0)
    print(tabela.to_string())


if __name__ == "__main__":
    print("📁 Criando estrutura de diretórios...")
    criar_pastas()

    print("📄 Carregando metadados...")
    df = carregar_metadata()

    print("✂️  Dividindo dataset (80/20) por lesion_id...")
    mapa_split, treino, val = split_por_lesion_id(df)

    print("📋 Copiando imagens para os conjuntos...")
    copiadas, nao_encontradas = copiar_imagens(df, mapa_split)

    imprimir_resumo(df, treino, val, mapa_split)

    print(f"\n✅ Concluído: {copiadas} imagens copiadas, {nao_encontradas} não encontradas.")
