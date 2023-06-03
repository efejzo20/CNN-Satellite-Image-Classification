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
                                                target_size=(600,600),
                                                batch_size=32, 
                                                class_mode='categorical', 
                                                seed=42)
    return train_data



def val_data_gen():

    val_datagen = ImageDataGenerator(rescale=1./255)
    val_data = val_datagen.flow_from_directory('./output/val', 
                                                target_size=(600,600), 
                                                batch_size=32, 
                                                class_mode='categorical', 
                                                seed=42)
    return val_data


def test_data_gen():

    test_datagen =  ImageDataGenerator(rescale=1./255)
    test_data = test_datagen.flow_from_directory('./output/test',
                                                target_size=(600,600),
                                                batch_size=32,
                                                class_mode='categorical',
                                                seed=42)
    return test_data





def train():

    train_data = train_data_gen()
    val_data = val_data_gen()

    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=8, 
                           kernel_size=3,
                           strides=1,
                           activation='relu',
                           input_shape=(600,600,3)), # input shape acts as input layer 
    tf.keras.layers.MaxPool2D(pool_size=2),
    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(), 
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(30, activation='softmax')
    ])  


    mlflow.autolog()

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])

    model.fit(train_data,
                                epochs=15, 
                                steps_per_epoch=len(train_data),
                                validation_data=val_data, 
                                validation_steps=len(val_data)) 
    model.save("./models/mymodel1/1")


if __name__ == "__main__":
    train()





