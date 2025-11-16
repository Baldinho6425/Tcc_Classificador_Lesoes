import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Caminho onde estão as imagens organizadas por classe
base_path = '../data/processed'

classes = sorted(os.listdir(base_path))

# Criar figura com 7 subplots (uma para cada classe)
fig, axs = plt.subplots(1, len(classes), figsize=(18, 4))
fig.suptitle('Exemplos de Imagens por Classe (HAM10000)', fontsize=16)

for idx, classe in enumerate(classes):
    classe_path = os.path.join(base_path, classe)
    imagens = [f for f in os.listdir(classe_path) if f.endswith(".jpg")]
    
    if not imagens:
        continue

    img_path = os.path.join(classe_path, imagens[0])  # Pega a primeira imagem da classe
    img = mpimg.imread(img_path)

    axs[idx].imshow(img)
    axs[idx].axis('off')
    axs[idx].set_title(classe.replace("_", " ").capitalize(), fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig('models/amostras_classes.png')
plt.close()
print("Imagem salva em models/amostras_classes.png")
