# src/training/train.py
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG16, MobileNetV2, EfficientNetB0
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1) CONFIGURAÇÃO DE HARDWARE (GPU + Mixed Precision)
# ============================================================

def configurar_hardware():
    """
    Configura GPU e Mixed Precision (se disponível).
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ {len(gpus)} GPU(s) detectada(s): {[gpu.name for gpu in gpus]}")

            # Habilita Mixed Precision apenas se GPU suportar
            try:
                from tensorflow.keras import mixed_precision
                mixed_precision.set_global_policy("mixed_float16")
                print("⚙️  Treinamento configurado para Mixed Precision (float16).")
            except Exception as e:
                print("⚠️  Não foi possível habilitar Mixed Precision:", e)

        except RuntimeError as e:
            print("Erro ao configurar GPU:", e)
    else:
        print("⚠️  Nenhuma GPU detectada. O treino será executado na CPU.")


# ============================================================
# 2) GERADORES DE DADOS COM DATA AUGMENTATION
# ============================================================

def get_preprocess_function(model_name: str):
    """
    Retorna a função de pré-processamento adequada para cada arquitetura.
    """
    model_name = model_name.lower()
    if model_name == "vgg16":
        return vgg_preprocess
    elif model_name == "mobilenetv2":
        return mobilenet_preprocess
    elif model_name == "efficientnetb0":
        return eff_preprocess
    else:
        raise ValueError(f"Modelo não suportado para preprocessamento: {model_name}")


def create_data_generators(
    model_name: str,
    data_dir: str = "data/processed",
    img_size=(224, 224),
    batch_size: int = 32,
    val_split: float = 0.2
):
    """
    Cria os geradores de dados para treino e validação, com data augmentation.
    Usa validation_split para separar 80%/20% (treino/validação).
    """

    preprocess_fn = get_preprocess_function(model_name)

    # Data augmentation mais rica, como discutido com o orientador
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        validation_split=val_split,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest"
    )

    # Para validação não precisamos de augmentation, só do preprocess
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        validation_split=val_split
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False  # importante para alinhar classes com as predições
    )

    return train_generator, val_generator


# ============================================================
# 3) CONSTRUÇÃO DO MODELO (VGG16, MobileNetV2, EfficientNetB0)
# ============================================================

def build_model(model_name: str, input_shape=(224, 224, 3), num_classes: int = 7,
                train_base: bool = False, hidden_units: int = 256):
    """
    Constrói e retorna (model, base_model) para a arquitetura desejada.
    """
    model_name = model_name.lower()

    if model_name == "vgg16":
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    elif model_name == "mobilenetv2":
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    elif model_name == "efficientnetb0":
        base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Arquitetura não suportada: {model_name}")

    base_model.trainable = train_base  # False para treino baseline; True para fine-tuning

    x = base_model.output
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)

    if hidden_units is not None and hidden_units > 0:
        x = Dense(hidden_units, activation="relu", name="dense_hidden")(x)

    # Camada de saída com Softmax (probabilidades por classe)
    predictions = Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)

    model = Model(inputs=base_model.input, outputs=predictions, name=f"{model_name}_classifier")

    return model, base_model


# ============================================================
# 4) TREINAMENTO (BASELINE + OPCIONAL FINE-TUNING)
# ============================================================

def train_model(
    model,
    base_model,
    train_data,
    val_data,
    model_name: str,
    epochs_base: int = 10,
    epochs_finetune: int = 0,
    lr_base: float = 1e-4,
    lr_finetune: float = 1e-5,
    output_dir_models: str = "models"
):
    """
    Treina o modelo em duas fases:
    - Fase 1: Treino da "cabeça" com o backbone congelado
    - Fase 2 (opcional): Fine-tuning com o backbone descongelado
    Retorna (history_base, history_finetune).
    """

    os.makedirs(output_dir_models, exist_ok=True)

    # Métricas incluídas já na compilação
    metrics = [
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]

    # --------------------------------------------------------
    # Fase 1 - Treino da cabeça
    # --------------------------------------------------------
    base_model.trainable = False
    optimizer = Adam(learning_rate=lr_base)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=metrics
    )

    checkpoint_path = os.path.join(output_dir_models, f"{model_name}_best_baseline.h5")
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    print(f"\n🚀 Iniciando treinamento BASELINE ({model_name})...")
    history_base = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs_base,
        callbacks=[checkpoint]
    )

    history_finetune = None

    # --------------------------------------------------------
    # Fase 2 - Fine-tuning (apenas se epochs_finetune > 0)
    # --------------------------------------------------------
    if epochs_finetune > 0:
        print(f"\n🔓 Iniciando FINE-TUNING ({model_name})...")
        base_model.trainable = True  # descongela backbone inteiro

        # Normalmente, congelar algumas camadas iniciais pode ser interessante,
        # mas a estratégia recomendada foi simplificar a lógica.
        optimizer_ft = Adam(learning_rate=lr_finetune)

        model.compile(
            optimizer=optimizer_ft,
            loss="categorical_crossentropy",
            metrics=metrics
        )

        checkpoint_path_ft = os.path.join(output_dir_models, f"{model_name}_best_finetune.h5")
        checkpoint_ft = ModelCheckpoint(
            checkpoint_path_ft,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        )

        history_finetune = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs_finetune,
            callbacks=[checkpoint_ft]
        )

    return history_base, history_finetune


# ============================================================
# 5) CÁLCULO DAS MÉTRICAS E MATRIZ DE CONFUSÃO
# ============================================================

def compute_specificity_multiclass(cm: np.ndarray):
    """
    Calcula a especificidade macro a partir da matriz de confusão.
    """
    num_classes = cm.shape[0]
    specificities = []

    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fn + fp)

        denom = (tn + fp)
        spec = tn / denom if denom > 0 else 0.0
        specificities.append(spec)

    macro_spec = np.mean(specificities)
    return macro_spec, specificities


def evaluate_and_save_metrics(
    model,
    val_data,
    model_name: str,
    output_dir_results: str = "results"
):
    """
    Faz predições no conjunto de validação, calcula métricas globais,
    gera e salva a matriz de confusão e retorna um dicionário de métricas.
    """

    os.makedirs(output_dir_results, exist_ok=True)
    os.makedirs(os.path.join(output_dir_results, "confusion_matrices"), exist_ok=True)

    # Número de amostras
    n_samples = val_data.samples
    steps = int(np.ceil(n_samples / val_data.batch_size))

    # Predições
    y_prob = model.predict(val_data, steps=steps)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = val_data.classes

    # Classes (nomes)
    class_indices = val_data.class_indices
    idx_to_class = {v: k for k, v in class_indices.items()}
    labels = [idx_to_class[i] for i in range(len(idx_to_class))]

    # Métricas
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    macro_spec, spec_per_class = compute_specificity_multiclass(cm)

    # Salva matriz de confusão como figura
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Matriz de Confusão - {model_name}")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    plt.ylabel("Verdadeiro")
    plt.xlabel("Predito")
    plt.tight_layout()
    cm_path = os.path.join(output_dir_results, "confusion_matrices", f"cm_{model_name}.png")
    plt.savefig(cm_path)
    plt.close()

    # Salva métricas em JSON (pode virar tabela depois)
    metrics_dict = {
        "model": model_name,
        "accuracy": float(acc),
        "precision_weighted": float(prec),
        "recall_weighted": float(rec),
        "f1_weighted": float(f1),
        "specificity_macro": float(macro_spec),
        "classes": labels,
        "specificity_per_class": [float(v) for v in spec_per_class]
    }

    metrics_path = os.path.join(output_dir_results, f"metrics_{model_name}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=4, ensure_ascii=False)

    print(f"\n📊 Métricas salvas em: {metrics_path}")
    print(f"🧩 Matriz de confusão salva em: {cm_path}")

    return metrics_dict


# ============================================================
# 6) PLOT DOS GRÁFICOS DE TREINAMENTO
# ============================================================

def plot_training_history(
    history_base,
    history_finetune,
    model_name: str,
    output_dir_results: str = "results"
):
    """
    Gera gráficos de acurácia e loss combinando treino e fine-tuning.
    Marca a transição entre as fases (se houver fine-tuning).
    """

    os.makedirs(output_dir_results, exist_ok=True)
    os.makedirs(os.path.join(output_dir_results, "plots"), exist_ok=True)

    # Combina históricos
    acc = history_base.history["accuracy"]
    val_acc = history_base.history["val_accuracy"]
    loss = history_base.history["loss"]
    val_loss = history_base.history["val_loss"]

    ft_start_epoch = len(acc)  # início do fine-tuning na numeração global

    if history_finetune is not None:
        acc += history_finetune.history["accuracy"]
        val_acc += history_finetune.history["val_accuracy"]
        loss += history_finetune.history["loss"]
        val_loss += history_finetune.history["val_loss"]

    epochs = range(1, len(acc) + 1)

    # Gráfico de acurácia
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, acc, label="Acurácia treino")
    plt.plot(epochs, val_acc, label="Acurácia validação")

    if history_finetune is not None:
        plt.axvline(x=ft_start_epoch, color="gray", linestyle="--", label="Início fine-tuning")

    plt.title(f"Acurácia durante o treinamento - {model_name}")
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.grid(True)

    acc_path = os.path.join(output_dir_results, "plots", f"accuracy_{model_name}.png")
    plt.savefig(acc_path)
    plt.close()

    # Gráfico de loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss, label="Loss treino")
    plt.plot(epochs, val_loss, label="Loss validação")

    if history_finetune is not None:
        plt.axvline(x=ft_start_epoch, color="gray", linestyle="--", label="Início fine-tuning")

    plt.title(f"Loss durante o treinamento - {model_name}")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    loss_path = os.path.join(output_dir_results, "plots", f"loss_{model_name}.png")
    plt.savefig(loss_path)
    plt.close()

    print(f"📈 Gráficos salvos em:\n  - {acc_path}\n  - {loss_path}")


# ============================================================
# 7) PIPELINE PRINCIPAL
# ============================================================

def main():
    configurar_hardware()

    data_dir = "data/processed"
    img_size = (224, 224)
    batch_size = 32  # Se der erro de memória, reduzir este valor.
    val_split = 0.2  # 80% treino, 20% validação

    # Modelos que serão testados (sem fine-tuning inicialmente)
    modelos = ["vgg16", "mobilenetv2", "efficientnetb0"]

    epochs_base = 10
    epochs_finetune = 0  # 👉 recomendação: primeiro 0; depois ativar só para o melhor modelo.

    resultados_gerais = []

    for model_name in modelos:
        print("\n" + "=" * 60)
        print(f"🏗️  Treinando modelo: {model_name.upper()}")
        print("=" * 60)

        train_gen, val_gen = create_data_generators(
            model_name=model_name,
            data_dir=data_dir,
            img_size=img_size,
            batch_size=batch_size,
            val_split=val_split
        )

        model, base_model = build_model(
            model_name=model_name,
            input_shape=img_size + (3,),
            num_classes=train_gen.num_classes,
            train_base=False,
            hidden_units=256
        )

        history_base, history_finetune = train_model(
            model=model,
            base_model=base_model,
            train_data=train_gen,
            val_data=val_gen,
            model_name=model_name,
            epochs_base=epochs_base,
            epochs_finetune=epochs_finetune
        )

        # Avaliação + métricas + matriz de confusão
        metrics_dict = evaluate_and_save_metrics(
            model=model,
            val_data=val_gen,
            model_name=model_name,
            output_dir_results="results"
        )

        resultados_gerais.append(metrics_dict)

        # Gráficos de loss e accuracy
        plot_training_history(
            history_base=history_base,
            history_finetune=history_finetune,
            model_name=model_name,
            output_dir_results="results"
        )

    # Salva um resumo com todas as arquiteturas testadas
    resumo_path = os.path.join("results", "resumo_modelos.json")
    with open(resumo_path, "w") as f:
        json.dump(resultados_gerais, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Treinamento concluído para todos os modelos.")
    print(f"📁 Resumo geral salvo em: {resumo_path}")


if __name__ == "__main__":
    main()
