# src/training/train.py
import os
import json
import random
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG16, MobileNetV2, EfficientNetB0
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.utils.class_weight import compute_class_weight

SEED = 42


# ============================================================
# 1) SEED DE REPRODUTIBILIDADE
# ============================================================

def configurar_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    print(f"🔒 Seed fixada: {SEED}")


# ============================================================
# 2) CONFIGURAÇÃO DE HARDWARE (GPU + Mixed Precision)
# ============================================================

def configurar_hardware():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ {len(gpus)} GPU(s) detectada(s): {[gpu.name for gpu in gpus]}")

            try:
                from tensorflow.keras import mixed_precision
                mixed_precision.set_global_policy("mixed_float16")
                print("⚙️  Mixed Precision (float16) habilitado.")
            except Exception as e:
                print("⚠️  Não foi possível habilitar Mixed Precision:", e)

        except RuntimeError as e:
            print("Erro ao configurar GPU:", e)
    else:
        print("⚠️  Nenhuma GPU detectada. O treino será executado na CPU.")


# ============================================================
# 3) GERADORES DE DADOS COM DATA AUGMENTATION
# ============================================================

def get_preprocess_function(model_name: str):
    model_name = model_name.lower()
    if model_name == "vgg16":
        return vgg_preprocess
    elif model_name == "mobilenetv2":
        return mobilenet_preprocess
    elif model_name == "efficientnetb0":
        return eff_preprocess
    else:
        raise ValueError(f"Modelo não suportado: {model_name}")


def create_data_generators(
    model_name: str,
    data_dir: str = "data/processed",
    img_size=(224, 224),
    batch_size: int = 32,
):
    """
    Espera a estrutura:
        data_dir/train/<classe>/
        data_dir/val/<classe>/

    Gerada pelo script organizar_ham10000.py (divisão 80/20 por lesion_id).
    """
    preprocess_fn = get_preprocess_function(model_name)

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest"
    )

    eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=SEED
    )

    val_generator = eval_datagen.flow_from_directory(
        os.path.join(data_dir, "val"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    return train_generator, val_generator


# ============================================================
# 4) PESOS POR CLASSE (compensa desbalanceamento do HAM10000)
# ============================================================

def calcular_class_weights(train_generator) -> dict:
    classes = train_generator.classes
    class_labels = np.unique(classes)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=class_labels,
        y=classes
    )
    class_weight_dict = dict(zip(class_labels.tolist(), weights.tolist()))

    idx_to_class = {v: k for k, v in train_generator.class_indices.items()}
    print("\n⚖️  Pesos por classe (compensação de desbalanceamento):")
    for idx, weight in class_weight_dict.items():
        print(f"   {idx_to_class[idx]}: {weight:.4f}")

    return class_weight_dict


# ============================================================
# 5) CONSTRUÇÃO DO MODELO
# ============================================================

def build_model(
    model_name: str,
    input_shape=(224, 224, 3),
    num_classes: int = 7,
    train_base: bool = False,
    hidden_units: int = 256,
    dropout_rate: float = 0.5
):
    model_name = model_name.lower()

    if model_name == "vgg16":
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    elif model_name == "mobilenetv2":
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    elif model_name == "efficientnetb0":
        base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Arquitetura não suportada: {model_name}")

    base_model.trainable = train_base

    x = base_model.output
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)

    if hidden_units and hidden_units > 0:
        x = Dense(hidden_units, activation="relu", name="dense_hidden")(x)
        x = Dropout(dropout_rate, name="dropout")(x)

    predictions = Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)

    model = Model(inputs=base_model.input, outputs=predictions, name=f"{model_name}_classifier")

    return model, base_model


# ============================================================
# 6) TREINAMENTO (BASELINE + OPCIONAL FINE-TUNING)
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
    output_dir_models: str = "models",
    class_weight: dict = None
):
    os.makedirs(output_dir_models, exist_ok=True)

    metrics = [
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]

    # --------------------------------------------------------
    # Fase 1 — Treino da cabeça (backbone congelado)
    # --------------------------------------------------------
    base_model.trainable = False
    model.compile(
        optimizer=Adam(learning_rate=lr_base),
        loss="categorical_crossentropy",
        metrics=metrics
    )

    checkpoint_path = os.path.join(output_dir_models, f"{model_name}_best_baseline.h5")

    callbacks_baseline = [
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
    ]

    print(f"\n🚀 Iniciando treinamento BASELINE ({model_name})...")
    history_base = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs_base,
        callbacks=callbacks_baseline,
        class_weight=class_weight
    )

    history_finetune = None

    # --------------------------------------------------------
    # Fase 2 — Fine-tuning (backbone descongelado)
    # --------------------------------------------------------
    if epochs_finetune > 0:
        print(f"\n🔓 Iniciando FINE-TUNING ({model_name})...")
        base_model.trainable = True

        model.compile(
            optimizer=Adam(learning_rate=lr_finetune),
            loss="categorical_crossentropy",
            metrics=metrics
        )

        checkpoint_path_ft = os.path.join(output_dir_models, f"{model_name}_best_finetune.h5")

        callbacks_finetune = [
            ModelCheckpoint(
                checkpoint_path_ft,
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1
            ),
            EarlyStopping(
                monitor="val_accuracy",
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
        ]

        history_finetune = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs_finetune,
            callbacks=callbacks_finetune,
            class_weight=class_weight
        )

    return history_base, history_finetune


# ============================================================
# 7) MÉTRICAS E MATRIZ DE CONFUSÃO
# ============================================================

def compute_specificity_multiclass(cm: np.ndarray):
    specificities = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fn + fp)
        denom = tn + fp
        specificities.append(tn / denom if denom > 0 else 0.0)
    return np.mean(specificities), specificities


def evaluate_and_save_metrics(
    model,
    val_data,
    model_name: str,
    output_dir_results: str = "results"
):
    os.makedirs(output_dir_results, exist_ok=True)
    os.makedirs(os.path.join(output_dir_results, "confusion_matrices"), exist_ok=True)

    y_prob = model.predict(val_data)
    y_pred = np.argmax(y_prob, axis=1)[:val_data.samples]
    y_true = val_data.classes

    idx_to_class = {v: k for k, v in val_data.class_indices.items()}
    labels = [idx_to_class[i] for i in range(len(idx_to_class))]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report_dict = classification_report(
        y_true, y_pred, target_names=labels, output_dict=True, zero_division=0
    )
    print("\n📋 Relatório por classe:")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))

    cm_raw = confusion_matrix(y_true, y_pred)
    macro_spec, spec_per_class = compute_specificity_multiclass(cm_raw)

    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

    plt.figure(figsize=(8, 6))
    im = plt.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.title(f"Matriz de Confusão Normalizada (%) - {model_name}")
    plt.colorbar(im)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            value = cm_norm[i, j] * 100.0
            text_color = "white" if cm_norm[i, j] > 0.5 else "black"
            plt.text(j, i, f"{value:.1f}%", ha="center", va="center",
                     color=text_color, fontsize=8)
    plt.ylabel("Classe verdadeira")
    plt.xlabel("Classe predita")
    plt.tight_layout()

    cm_path = os.path.join(
        output_dir_results, "confusion_matrices", f"cm_{model_name}_percent.png"
    )
    plt.savefig(cm_path, dpi=300)
    plt.close()

    metrics_dict = {
        "model": model_name,
        "accuracy": float(acc),
        "precision_weighted": float(prec),
        "recall_weighted": float(rec),
        "f1_weighted": float(f1),
        "specificity_macro": float(macro_spec),
        "classes": labels,
        "specificity_per_class": [float(v) for v in spec_per_class],
        "classification_report": report_dict,
    }

    metrics_path = os.path.join(output_dir_results, f"metrics_{model_name}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=4, ensure_ascii=False)

    print(f"\n📊 Acurácia: {acc:.4f} | F1: {f1:.4f} | Especificidade: {macro_spec:.4f}")
    print(f"💾 Métricas salvas em: {metrics_path}")

    return metrics_dict


# ============================================================
# 8) GRÁFICOS DE TREINAMENTO
# ============================================================

def plot_training_history(
    history_base,
    history_finetune,
    model_name: str,
    output_dir_results: str = "results"
):
    os.makedirs(os.path.join(output_dir_results, "plots"), exist_ok=True)

    acc = history_base.history["accuracy"]
    val_acc = history_base.history["val_accuracy"]
    loss = history_base.history["loss"]
    val_loss = history_base.history["val_loss"]
    ft_start_epoch = len(acc)

    if history_finetune is not None:
        acc += history_finetune.history["accuracy"]
        val_acc += history_finetune.history["val_accuracy"]
        loss += history_finetune.history["loss"]
        val_loss += history_finetune.history["val_loss"]

    epochs = range(1, len(acc) + 1)

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

    print(f"📈 Gráficos salvos em: {acc_path} | {loss_path}")


# ============================================================
# 9) ARGUMENTOS DE LINHA DE COMANDO
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Classificador de lesões de pele — HAM10000"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["vgg16", "mobilenetv2", "efficientnetb0"],
        help="Treina somente o modelo especificado (padrão: vgg16 + mobilenetv2)"
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Ativa fine-tuning após baseline (10 + 10 épocas)"
    )
    parser.add_argument(
        "--reduced-lr",
        action="store_true",
        help="Usa LR reduzido (5e-5) e mais épocas (20) no baseline"
    )
    parser.add_argument(
        "--full-test-suite",
        action="store_true",
        help="Treina todos os modelos, incluindo EfficientNetB0"
    )
    return parser.parse_args()


# ============================================================
# 10) PIPELINE PRINCIPAL
# ============================================================

def main():
    args = parse_args()
    configurar_seed()
    configurar_hardware()

    data_dir = "data/processed"
    img_size = (224, 224)
    batch_size = 32

    if args.full_test_suite:
        modelos = ["vgg16", "mobilenetv2", "efficientnetb0"]
    elif args.model:
        modelos = [args.model]
    else:
        modelos = ["vgg16", "mobilenetv2"]

    if args.reduced_lr:
        epochs_base = 20
        lr_base = 5e-5
    else:
        epochs_base = 10
        lr_base = 1e-4

    epochs_finetune = 10 if args.finetune else 0

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
        )

        class_weight_dict = calcular_class_weights(train_gen)

        model, base_model = build_model(
            model_name=model_name,
            input_shape=img_size + (3,),
            num_classes=train_gen.num_classes,
            train_base=False,
            hidden_units=256,
            dropout_rate=0.5
        )

        history_base, history_finetune = train_model(
            model=model,
            base_model=base_model,
            train_data=train_gen,
            val_data=val_gen,
            model_name=model_name,
            epochs_base=epochs_base,
            epochs_finetune=epochs_finetune,
            lr_base=lr_base,
            output_dir_models="models",
            class_weight=class_weight_dict
        )

        metrics_dict = evaluate_and_save_metrics(
            model=model,
            val_data=val_gen,
            model_name=model_name,
            output_dir_results="results"
        )

        resultados_gerais.append(metrics_dict)

        plot_training_history(
            history_base=history_base,
            history_finetune=history_finetune,
            model_name=model_name,
            output_dir_results="results"
        )

    resumo_path = os.path.join("results", "resumo_modelos.json")
    with open(resumo_path, "w", encoding="utf-8") as f:
        json.dump(resultados_gerais, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Treinamento concluído para todos os modelos.")
    print(f"📁 Resumo geral salvo em: {resumo_path}")


if __name__ == "__main__":
    main()
