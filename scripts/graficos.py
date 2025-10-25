import matplotlib.pyplot as plt
import json

def plot_history(history):
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Acurácia
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, acc, 'bo-', label='Treinamento')
    plt.plot(epochs, val_acc, 'ro-', label='Validação')
    plt.title('Evolução da Acurácia')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True)
    plt.savefig('models/accuracy.png')
    plt.close()

    # Perda
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss, 'bo-', label='Treinamento')
    plt.plot(epochs, val_loss, 'ro-', label='Validação')
    plt.title('Evolução da Perda (Loss)')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('models/loss.png')
    plt.close()

if __name__ == "__main__":
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # Repetir treinamento para obter histórico
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.15
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

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(train_data.num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, validation_data=val_data, epochs=10)

    plot_history(history.history)
