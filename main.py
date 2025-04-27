import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import numpy as np
import time

def preprocess_image(image, label):
    image = tf.image.resize_with_pad(image, 224, 224)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def build_cnn_model(num_layers=4, dropout_rate=0.0):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    for i in range(num_layers - 1):
        model.add(layers.Conv2D(64 * (2**i), (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(10, activation='softmax'))
    return model

def main():
    data_dir = 'dataset'
    batch_size = 32
    img_size = (224, 224)
    seed = 123

    validation_splits = [0.2, 0.3]
    num_epochs_list = [15, 20]
    num_layers_list = [4, 3, 2]

    all_histories = {}

    for split, epochs in zip(validation_splits, num_epochs_list):
        train_ds = image_dataset_from_directory(
            data_dir,
            validation_split=split,
            subset="training",
            seed=seed,
            shuffle=True,
            image_size=img_size,
            batch_size=batch_size
        )

        val_ds = image_dataset_from_directory(
            data_dir,
            validation_split=split,
            subset="validation",
            seed=seed,
            shuffle=True,
            image_size=img_size,
            batch_size=batch_size
        )

        train_ds = train_ds.map(preprocess_image).prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = val_ds.map(preprocess_image).prefetch(buffer_size=tf.data.AUTOTUNE)

        for num_layers in num_layers_list:
            print(f"\nTraining model with {num_layers} layers and {100*(1-split)}/{100*split} split for {epochs} epochs...")
            model = build_cnn_model(num_layers=num_layers)
            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                          metrics=['accuracy'])

            start_time = time.time()
            history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=0)
            end_time = time.time()
            print(f"Training took {end_time - start_time:.2f} seconds")

            all_histories[(num_layers, f'{int(100*(1-split))}-{int(100*split)}')] = history.history

    print("\nTraining FINAL model: 2 layers, dropout=0.5, 70/30 split, 20 epochs...")

    final_split = 0.3
    final_epochs = 20
    final_dropout = 0.5
    final_num_layers = 2

    final_train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=final_split,
        subset="training",
        seed=seed,
        shuffle=True,
        image_size=img_size,
        batch_size=batch_size
    )

    final_val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=final_split,
        subset="validation",
        seed=seed,
        shuffle=True,
        image_size=img_size,
        batch_size=batch_size
    )

    final_train_ds = final_train_ds.map(preprocess_image).prefetch(buffer_size=tf.data.AUTOTUNE)
    final_val_ds = final_val_ds.map(preprocess_image).prefetch(buffer_size=tf.data.AUTOTUNE)

    final_model = build_cnn_model(num_layers=final_num_layers, dropout_rate=final_dropout)
    final_model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                        metrics=['accuracy'])

    final_history = final_model.fit(final_train_ds, validation_data=final_val_ds, epochs=final_epochs, verbose=0)

    plt.figure(figsize=(14, 10))
    for (num_layers, split_label), history in all_histories.items():
        if num_layers == 2 and split_label == '70-30':
            plt.plot(history['accuracy'], label=f'{num_layers}-layer Train ({split_label})', color='red', linewidth=3)
            plt.plot(history['val_accuracy'], label=f'{num_layers}-layer Val ({split_label})', color='red', linestyle='--', linewidth=3)
        else:
            plt.plot(history['accuracy'], label=f'{num_layers}-layer Train ({split_label})')
            plt.plot(history['val_accuracy'], label=f'{num_layers}-layer Val ({split_label})', linestyle='--')

    plt.title('Training and Validation Accuracy for Different Models and Splits')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.annotate('Final Model',
                 xy=(final_epochs-1, all_histories[(2, '70-30')]['val_accuracy'][-1]),
                 xytext=(final_epochs-5, 0.6),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 fontsize=12, color='red')
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    ax[0].plot(final_history.history['accuracy'], label='Train Accuracy')
    ax[0].plot(final_history.history['val_accuracy'], label='Val Accuracy')
    ax[0].set_title('Final Model Accuracy (2 layers, dropout)')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(final_history.history['loss'], label='Train Loss')
    ax[1].plot(final_history.history['val_loss'], label='Val Loss')
    ax[1].set_title('Final Model Loss (2 layers, dropout)')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()
    ax[1].grid(True)

    plt.show()

if __name__ == "__main__":
    main()
