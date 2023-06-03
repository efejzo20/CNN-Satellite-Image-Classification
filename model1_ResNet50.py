import os
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import mlflow

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub
from tensorflow.keras.applications import ResNet50V2


def train_data_gen():
    # Data preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    train_data = train_datagen.flow_from_directory('./output/train',
                                                target_size=(224,224),
                                                batch_size=32, 
                                                class_mode='categorical', 
                                                seed=42)
    return train_data



def val_data_gen():

    val_datagen = ImageDataGenerator(rescale=1./255)
    val_data = val_datagen.flow_from_directory('./output/val', 
                                                target_size=(224,224), 
                                                batch_size=32, 
                                                class_mode='categorical', 
                                                seed=42)
    return val_data


def test_data_gen():

    test_datagen =  ImageDataGenerator(rescale=1./255)
    test_data = test_datagen.flow_from_directory('./output/test',
                                                target_size=(224,224),
                                                batch_size=32,
                                                class_mode='categorical',
                                                seed=42)
    return test_data


def train():
    resnet_train_data = train_data_gen()
    resnet_val_data = val_data_gen()

    # Load the ResNet50V2 model with pre-trained weights
    resnet_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    # Freeze the pre-trained layers
    resnet_model.trainable = False

    # Add custom classification layers on top
    model = tf.keras.Sequential([
        resnet_model,
        tf.keras.layers.Flatten(),
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


    