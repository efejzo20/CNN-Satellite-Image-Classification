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

mobilenet_url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'

def train():

    mobilenet_train_data = train_data_gen()
    mobilenet_val_data = val_data_gen()
    # install the pretrained Imagenet model and save it as a Keras Layer (Sequential API)
    feature_vector_layer = hub.KerasLayer(mobilenet_url,
                                        trainable=False,
                                        input_shape=(224,224,3))

    mobilenet_model = tf.keras.Sequential([
        feature_vector_layer,
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(30, activation='softmax')
    ])


    mlflow.autolog()


    # compile the TL model
    mobilenet_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=['accuracy'])
    
    mobilenet_model.fit(mobilenet_train_data,
                                      epochs=20, 
                                      steps_per_epoch=len(mobilenet_train_data),
                                      validation_data=mobilenet_val_data,
                                      validation_steps=len(mobilenet_val_data))
    
    mobilenet_model.save("./models/MobileNet/1")


if __name__ == "__main__":
    train()


    