import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def build_model(num_classes, img_height, img_width):
    '''Setup the neural network model. Source: https://www.tensorflow.org/tutorials/images/classification

    From the tutorial:
    The model consists of three convolution blocks with a max pool layer
    in each of them. There's a fully connected layer with 128 units on top of it
    that is activated by a relu activation function.

    Arguments:
    num_classes -- the number of output classes
    img_height -- height of the input images
    img_width -- width of the input images

    Return:
    A Keras Sequential model object
    '''

    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    return model


def compile_model(model, optimizer='adam'):
    '''Compile model using chosen optimizer.

    Arguments:
    model -- a Keras model instance

    Return:
    Compiled version of the provided model
    '''

    model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    return model


def fit_model(model, train_set, val_set, epochs):
    '''Fit the model on some data.

    Arguments:
    model -- a compiled Keras model
    train_set -- training set
    val_set -- validation set
    epochs -- number of epochs on which to train

    Return:
    model -- the model with adjusted parameters
    history -- history of the training run
    '''

    history = model.fit(
        train_set,
        validation_data=val_set,
        epochs=epochs
    )

    return model, history
