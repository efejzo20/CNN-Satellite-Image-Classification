import os
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import mlflow
from tensorflow.keras.layers import Dense, Input, RandomRotation, RandomFlip, RandomZoom, RandomHeight, RandomWidth, Rescaling, GlobalAveragePooling2D
from tensorflow.keras import Sequential

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub
from tensorflow.keras.applications import ResNet50V2


def train_data_gen():
    # Data preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        fill_mode='nearest'
    )
    train_data = train_datagen.flow_from_directory('./output/train',
                                                   target_size=(224, 224),
                                                   batch_size=32,
                                                   class_mode='categorical',
                                                   seed=42)
    return train_data


def val_data_gen():
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    val_data = val_datagen.flow_from_directory('./output/val',
                                               target_size=(224, 224),
                                               batch_size=32,
                                               class_mode='categorical',
                                               seed=42)
    return val_data


def test_data_gen():
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_data = test_datagen.flow_from_directory('./output/test',
                                                 target_size=(224, 224),
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 seed=42)
    return test_data


def train():
    augmentation_layer = Sequential([
        RandomFlip("horizontal", seed=42),
        RandomRotation(0.2, seed=42),
        RandomZoom(0.2, seed=42),
        RandomHeight(0.2, seed=42),
        RandomWidth(0.2, seed=42),
        Rescaling(1 / 255.)
    ], name="augmentation_layer")

    resnet_train_data = train_data_gen()
    resnet_val_data = val_data_gen()

    # Load the ResNet50V2 model with pre-trained weights
    resnet_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    # Freeze the pre-trained layers
    resnet_model.trainable = False
    unfreeze_layers = 15
    for layer in resnet_model.layers[:-unfreeze_layers]:
        layer.trainable = True

    # Add augmentation_layer too
    # Remove Flatten and add GlobalAveragePooling2D
    model = tf.keras.Sequential([
        augmentation_layer,
        resnet_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(30, activation='softmax')
    ])

    mlflow.autolog()

    # Compile the model
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    # Train the model
    model.fit(
        resnet_train_data,
        epochs=20,
        steps_per_epoch=len(resnet_train_data),
        validation_data=resnet_val_data,
        validation_steps=len(resnet_val_data)
    )

    # Save the model
    model.save("./models/ResNet50V2/1")


if __name__ == "__main__":
    train()


