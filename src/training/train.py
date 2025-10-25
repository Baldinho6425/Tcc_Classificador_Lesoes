# src/training/train.py
import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

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
                print("⚠ Não foi possível habilitar Mixed Precision:", e)

        except RuntimeError as e:
            print("Erro ao configurar GPU:", e)
    else:
        print("⚠ Nenhuma GPU detectada. O treino será executado na CPU.")

def plotar_historico(history, history_finetune):
    """
    Gera gráficos de acurácia e loss combinando treino e fine-tuning.
    """
    # Combina os históricos de treino e fine-tuning
    acc = history.history['accuracy'] + history_finetune.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_finetune.history['val_accuracy']
    loss = history.history['loss'] + history_finetune.history['loss']
    val_loss = history.history['val_loss'] + history_finetune.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # Gráfico de acurácia
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, acc, 'b', label='Acurácia treino')
    plt.plot(epochs, val_acc, 'r', label='Acurácia validação')
    plt.title('Acurácia durante o treinamento')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True)
    os.makedirs("models", exist_ok=True)
    plt.savefig("models/grafico_acuracia.png")
    plt.show()

    # Gráfico de loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss, 'b', label='Loss treino')
    plt.plot(epochs, val_loss, 'r', label='Loss validação')
    plt.title('Loss durante o treinamento')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("models/grafico_loss.png")
    plt.show()

def treinar_modelo():
    # =============================
    # Configuração de hardware
    # =============================
    configurar_hardware()

    # =============================
    # Data augmentation
    # =============================
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.15,
        horizontal_flip=True,
        vertical_flip=True,
    )

    train_data = datagen.flow_from_directory(
        'data/processed',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        'data/processed',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    # =============================
    # Modelo base (VGG16)
    # =============================
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Treina apenas a cabeça no início

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(train_data.num_classes, activation='softmax', dtype='float32')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # =============================
    # Compilação do modelo
    # =============================
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Callback para salvar o melhor modelo
    checkpoint = ModelCheckpoint("models/modelo_cnn.h5", monitor="val_accuracy",
                                 save_best_only=True, mode="max")

    # =============================
    # Treinamento - Fase 1
    # =============================
    print("\n🚀 Iniciando treinamento (Fase 1 - Cabeça da rede)...")
    history = model.fit(train_data, validation_data=val_data, epochs=10, callbacks=[checkpoint])

    # =============================
    # Fine-tuning - Fase 2
    # =============================
    print("\n🔓 Descongelando as últimas camadas da VGG16 para fine-tuning...")
    base_model.trainable = True
    for layer in base_model.layers[:-4]:  # congela todas, menos as últimas 4 camadas
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("\n🚀 Iniciando Fine-Tuning (Fase 2)...")
    history_finetune = model.fit(train_data, validation_data=val_data, epochs=5, callbacks=[checkpoint])

    # =============================
    # Resultados finais
    # =============================
    final_acc = history_finetune.history["accuracy"][-1]
    final_val_acc = history_finetune.history["val_accuracy"][-1]

    os.makedirs("models", exist_ok=True)
    with open("models/acuracia_final.txt", "w") as f:
        f.write(f"Acurácia (treinamento): {final_acc:.4f}\n")
        f.write(f"Acurácia (validação): {final_val_acc:.4f}\n")

    print(f"\n✅ Modelo treinado com acurácia de treino: {final_acc:.4f}")
    print(f"✅ Acurácia de validação: {final_val_acc:.4f}")

    # =============================
    # Geração dos gráficos
    # =============================
    plotar_historico(history, history_finetune)

if __name__ == "__main__":
    treinar_modelo()
