import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Usa o conjunto de treino para amostrar as classes
BASE_PATH   = '../../data/processed/train'
OUTPUT_PATH = '../../results/amostras_classes.png'

classes = sorted(os.listdir(BASE_PATH))

fig, axs = plt.subplots(1, len(classes), figsize=(18, 4))
fig.suptitle('Exemplos de Imagens por Classe (HAM10000)', fontsize=16)

for idx, classe in enumerate(classes):
    classe_path = os.path.join(BASE_PATH, classe)
    imagens = [f for f in os.listdir(classe_path) if f.endswith('.jpg')]

    if not imagens:
        axs[idx].axis('off')
        continue

    img = mpimg.imread(os.path.join(classe_path, imagens[0]))
    axs[idx].imshow(img)
    axs[idx].axis('off')
    axs[idx].set_title(classe.replace('_', ' ').capitalize(), fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.9])
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=150)
plt.close()
print(f"Imagem salva em {OUTPUT_PATH}")
