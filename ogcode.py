'''
The Data Divas:
    Anya Shrestha
    Elena Rodriguez
    Haley Trinh
    Lendy Varela

COSC 4337 - Data Science II
'''

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import time

def preprocess_image(image, label):
    image = tf.image.resize_with_pad(image,224,224)
    image = tf.cast(image, tf.float32) / 255.0
    return(image, label)

def main():
    data_dir = 'dataset'
    # Load dataset
    batch_size = 32
    img_size = (224, 224) 

    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.3,
        subset="training",
        seed=123,
        shuffle=True,
        image_size=img_size,
        batch_size=batch_size
    )
    
    val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.3,
        subset="validation",
        seed=123,
        shuffle=True,
        image_size=img_size,
        batch_size=batch_size
    )

    # Preprocessing
    train_ds = train_ds.map(preprocess_image)
    val_ds = val_ds.map(preprocess_image)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    #CNN
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')  # 10 dog breeds
    ])
    '''model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')  # 10 dog breeds
    ])'''

    #original model
    '''model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')  # 10 dog breeds
    ])'''
    #model w/ 4 layers
    '''model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')  # 10 dog breeds
    ])'''

    #model w/ dropout
    '''model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')  # 10 dog breeds
    ])'''

    # Compile 
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])

    history = model.fit(train_ds, validation_data=val_ds, epochs=20)
    
    # Evaluate
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    plt.show()

    test_loss, test_acc = model.evaluate(val_ds)
    print('\nValidation accuracy:', test_acc)


if __name__ == "__main__":
    main()



'''
final model = 2 layers, dropout, 70/30
other trials:
original model = 4 layers , 80/20 split, 10 epochs
4 layers, 80/20 split, drop out
4 layers, 70/30 split
4 layers, 70/30 split, drop out
4 layers, 70/30 split, drop out, 20 epochs

3 layers, 80/20 split
3 layers, 80/20 split, drop out
-> we wanted to see if less layers were better. 3 layers produced same results as 4. we saw change when we switched to 2 layers.

2 layers, 80/20 split
2 layers, 80/20 split, drop out
2 layers, 70/30 split
2 layers, 70/30 split, drop out
2 layers, 70/30 split, drop out, 20 epochs

epochs = [10,20] -> for history =...
dropout_rate = [0,0.5] -> 0 = no dropout , 0.5 = yes dropout


create 3 diff models:
- 4 layers
- 3 layers
- 2 layers

2 diff splits:
- 80/20   validation_split=0.2
- 70/30   validation_split=0.3

ONLY need to implement 20 epochs for the 70/30 split
try plotting on one plot?


oh yeah, for the code above - for the perceptron assignment we did, i think it would be kind of similar. i did nested for loops like this:
for sample in sample_sizes:
        for rate in learning_rates:
            for epoch in num_epochs:

then saved the accuracies in a list like:
final = {}
final[(sample, epoch, rate)] = history.history['accuracy']

then plotted like:
for label, val in final.items():
    sample, epoch, rate = label
    if rate == 0.001:
        ax1.plot(range(1, epoch+1), val, label=f'Sample Size = {sample}, Epochs={epoch}')
        ax1.set_title('Learning Rate = 0.001')
        ax1.set(xlabel='Epoch', ylabel='Accuracy')
    elif rate == 0.01:
        ax2.plot(range(1, epoch+1), val, label=f'Sample Size = {sample}, Epochs={epoch}')
        ax2.set_title('Learning Rate = 0.01')
        ax2.set(xlabel='Epoch', ylabel='Accuracy')
    else:
        ax3.plot(range(1, epoch+1), val, label=f'Sample Size = {sample}, Epochs={epoch}')
        ax3.set_title('Learning Rate = 0.1')
        ax3.set(xlabel='Epoch', ylabel='Accuracy')
ax1.legend()
ax2.legend()
ax3.legend()
plt.show()


'''