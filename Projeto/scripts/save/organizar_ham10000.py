import os
import pandas as pd
import shutil

# Caminhos
csv_path = '../data/raw/HAM10000_metadata.csv'
img_dir = '../data/raw/Images'
dest_dir = '../data/processed'

# Mapeamento dx -> nomes legíveis
label_map = {
    'nv': 'melanocytic_nevi',
    'mel': 'melanoma',
    'bkl': 'benign_keratosis',
    'bcc': 'basal_cell_carcinoma',
    'akiec': 'actinic_keratoses',
    'df': 'dermatofibroma',
    'vasc': 'vascular_lesions'
}

# Criar pastas de destino
for label in label_map.values():
    os.makedirs(os.path.join(dest_dir, label), exist_ok=True)

# Ler o CSV
df = pd.read_csv(csv_path)

# Mover as imagens para as pastas corretas
for index, row in df.iterrows():
    image_id = row['image_id']
    dx = row['dx']
    label_folder = label_map.get(dx)

    src_path_jpg = os.path.join(img_dir, image_id + ".jpg")
    dst_path = os.path.join(dest_dir, label_folder, image_id + ".jpg")

    if os.path.exists(src_path_jpg):
        shutil.copy(src_path_jpg, dst_path)
    else:
        print(f"Imagem não encontrada: {src_path_jpg}")
