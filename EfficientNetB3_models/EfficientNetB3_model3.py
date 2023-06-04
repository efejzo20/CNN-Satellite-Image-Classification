import os
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mlflow
import mlflow.tensorflow

efficientnet_url = 'https://tfhub.dev/google/efficientnet/b3/feature-vector/1'


def create_data_generators():
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    train_data = datagen.flow_from_directory(
        './output/train',
        target_size=(300, 300),
        batch_size=32,
        class_mode='categorical',
        seed=42
    )

    val_data = datagen.flow_from_directory(
        './output/val',
        target_size=(300, 300),
        batch_size=32,
        class_mode='categorical',
        seed=42
    )

    test_data = datagen.flow_from_directory(
        './output/test',
        target_size=(300, 300),
        batch_size=32,
        class_mode='categorical',
        seed=42
    )

    return train_data, val_data, test_data


def create_model():
    feature_vector_layer = hub.KerasLayer(
        efficientnet_url,
        trainable=False,
        input_shape=(300, 300, 3)
    )

    for layer in feature_vector_layer.layers[:-10]:  # Unfreeze last 10 layers
        layer.trainable = True

    #add layers to the model, add batch normalization after each dense
    model = tf.keras.Sequential([
        feature_vector_layer,
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(30, activation='softmax')
    ])

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Adjust the learning rate
        metrics=['accuracy']
    )

    return model


def train():
    train_data, val_data, _ = create_data_generators()
    model = create_model()

    mlflow.autolog()

    # Add callbacks for early stopping and learning rate scheduling
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=3)
    ]

    model.fit(
        train_data,
        epochs=30,  # Increase the number of epochs if needed
        steps_per_epoch=len(train_data),
        validation_data=val_data,
        validation_steps=len(val_data),
        callbacks=callbacks
    )

    model.save("./models/EfficientNetB3/3")


if __name__ == "__main__":
    train()
