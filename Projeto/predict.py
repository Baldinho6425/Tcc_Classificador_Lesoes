"""
Inferência em imagens de lesões de pele com visualização Grad-CAM.

Uso:
    python Projeto/predict.py \
        --image caminho/lesao.jpg \
        --model models/mobilenetv2_best_baseline.h5 \
        --arch mobilenetv2
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess

IMG_SIZE = (224, 224)

CLASSES = [
    "actinic_keratoses",
    "basal_cell_carcinoma",
    "benign_keratosis",
    "dermatofibroma",
    "melanocytic_nevi",
    "melanoma",
    "vascular_lesions",
]

CLASSES_PT = {
    "actinic_keratoses":    "Ceratose Actínica",
    "basal_cell_carcinoma": "Carcinoma Basocelular",
    "benign_keratosis":     "Ceratose Benigna",
    "dermatofibroma":       "Dermatofibroma",
    "melanocytic_nevi":     "Nevo Melanocítico",
    "melanoma":             "Melanoma",
    "vascular_lesions":     "Lesão Vascular",
}

# Último layer convolucional de cada backbone (alvo do Grad-CAM)
LAST_CONV_LAYER = {
    "vgg16":          "block5_conv3",
    "mobilenetv2":    "Conv_1",
    "efficientnetb0": "top_conv",
}

PREPROCESS_FN = {
    "vgg16":          vgg_preprocess,
    "mobilenetv2":    mobilenet_preprocess,
    "efficientnetb0": eff_preprocess,
}


# ============================================================
# PRÉ-PROCESSAMENTO
# ============================================================

def load_image(image_path: str, preprocess_fn) -> np.ndarray:
    img = load_img(image_path, target_size=IMG_SIZE)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return preprocess_fn(arr.copy())


# ============================================================
# GRAD-CAM
# ============================================================

def find_conv_output(model, layer_name: str):
    """Busca o tensor de saída de um layer pelo nome (suporta modelo flat e aninhado)."""
    try:
        return model.get_layer(layer_name).output
    except ValueError:
        pass
    for layer in model.layers:
        if hasattr(layer, "layers"):
            try:
                return layer.get_layer(layer_name).output
            except ValueError:
                continue
    raise ValueError(
        f"Layer '{layer_name}' não encontrado. "
        "Verifique se --arch corresponde ao modelo carregado."
    )


def compute_gradcam(model, img_array: np.ndarray, class_idx: int, arch: str) -> np.ndarray:
    """Calcula o heatmap Grad-CAM normalizado [0, 1] para a classe predita."""
    conv_output = find_conv_output(model, LAST_CONV_LAYER[arch])
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[conv_output, model.output]
    )

    img_tensor = tf.cast(img_array, tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise RuntimeError(
            "Gradientes não computados. Tente reconstruir o modelo antes de carregar."
        )

    pooled_grads = tf.reduce_mean(tf.cast(grads, tf.float32), axis=(0, 1, 2))
    conv_f32 = tf.cast(conv_outputs[0], tf.float32)
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_f32), axis=-1).numpy()

    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap


def overlay_heatmap(image_path: str, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Sobrepõe o heatmap Grad-CAM na imagem original (retorna uint8 RGB)."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)

    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    return cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)


def save_result(image_path, heatmap, predicted_class, confidence, arch, output_dir) -> str:
    """Salva figura com imagem original + Grad-CAM lado a lado."""
    original = plt.imread(image_path)
    overlay = overlay_heatmap(image_path, heatmap)
    class_pt = CLASSES_PT[predicted_class]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(original)
    axes[0].set_title("Imagem Original", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(
        f"Grad-CAM — {class_pt}\nConfiança: {confidence:.1%}", fontsize=12
    )
    axes[1].axis("off")

    plt.suptitle(f"Classificação de Lesão de Pele ({arch.upper()})", fontsize=13, y=1.02)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"gradcam_{basename}_{arch}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def classify(image_path: str, model_path: str, arch: str, output_dir: str = "results/gradcam"):
    arch = arch.lower()

    print(f"\n📂 Imagem      : {image_path}")
    print(f"🤖 Modelo      : {model_path}")
    print(f"🏗️  Arquitetura : {arch.upper()}")

    model = load_model(model_path)
    img_array = load_image(image_path, PREPROCESS_FN[arch])

    preds = model.predict(img_array, verbose=0)[0]
    class_idx = int(np.argmax(preds))
    predicted_class = CLASSES[class_idx]
    confidence = float(preds[class_idx])

    print(f"\n🏷️  Classe predita : {CLASSES_PT[predicted_class]} ({predicted_class})")
    print(f"📊 Confiança       : {confidence:.2%}")
    print("\n📋 Probabilidades por classe:")
    for i, (cls, prob) in enumerate(zip(CLASSES, preds)):
        bar = "█" * int(prob * 20)
        marker = " ◄" if i == class_idx else ""
        print(f"   {CLASSES_PT[cls]:<28} {prob:5.1%}  {bar}{marker}")

    try:
        heatmap = compute_gradcam(model, img_array, class_idx, arch)
        out_path = save_result(image_path, heatmap, predicted_class, confidence, arch, output_dir)
        print(f"\n🔬 Grad-CAM salvo em: {out_path}")
    except Exception as exc:
        print(f"\n⚠️  Grad-CAM não gerado: {exc}")

    return predicted_class, confidence


# ============================================================
# ARGUMENTOS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Classifica uma lesão de pele e gera mapa Grad-CAM."
    )
    parser.add_argument("--image", "-i", required=True,
                        help="Caminho para a imagem de entrada (.jpg ou .png)")
    parser.add_argument("--model", "-m", required=True,
                        help="Caminho para o modelo treinado (.h5)")
    parser.add_argument("--arch", "-a", required=True,
                        choices=["vgg16", "mobilenetv2", "efficientnetb0"],
                        help="Arquitetura do modelo (deve corresponder ao arquivo .h5)")
    parser.add_argument("--output-dir", default="results/gradcam",
                        help="Diretório de saída para o Grad-CAM (padrão: results/gradcam)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    classify(
        image_path=args.image,
        model_path=args.model,
        arch=args.arch,
        output_dir=args.output_dir,
    )
