import numpy as np
import matplotlib.pyplot as plt
from math import floor
import tensorflow as tf


def load_images(images_path, val_prop=0.2, batch_size=32, img_height=100, img_width=100):
    '''Load training and validation datasets using Keras' convience method.

    Required arguments:
    images_path -- a path to the top-level image directory

    Keyword arguments (optional):
    val_prop -- what proportion of the images will be used for validation?
    batch_size -- how many images per batch?
    img_height -- height dimensions (by default assumes square images)
    img_width -- width dimensions (by default assumes square images)

    Return:
    A tuple containing the training and validation sets
    '''

    # Use same seed to ensure no overlap between training and validation sets
    seed = np.random.randint(99999)
    train_set = tf.keras.preprocessing.image_dataset_from_directory(
        images_path,
        validation_split=val_prop,
        subset="training",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    val_set = tf.keras.preprocessing.image_dataset_from_directory(
        images_path,
        validation_split=val_prop,
        subset="validation",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    # Return tuple of train and validation sets
    return train_set, val_set


def preview_batch(batch, class_names, n_rows=4):
    '''Sanity check to ensure images were loaded correctly.

    Reguired arguments:
    batch -- the batch automatically built by Keras from calling load_images()
    class_names -- list of class names

    Keyword arguments (optional):
    n_rows -- when plotting array of images, number of rows to display
    '''

    for images, labels in batch:
        batch_size = images.shape[0]
        n_cols = int(floor(batch_size / n_rows))
        # Initialize plot
        fig, ax = plt.subplots(n_rows, n_cols)
        for i in range(n_rows):
            for j in range(n_cols):
                ax[i, j].imshow(images[i * n_rows + j].numpy().astype("uint8"))
                ax[i, j].set_title(class_names[labels[i * n_rows + j]])
                ax[i, j].axis("off")
        plt.show()


def configure_data_prefetching(train_set, val_set):
    '''Set up buffered prefetching to cache images after first epoch.

    Arguments:
    train_set -- training set produced by calling load_images()
    val_set -- validation set produced by calling load_images()

    Return:
    A tuple of the configured versions of both input sets
    '''

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_set = train_set.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_set = val_set.cache().prefetch(buffer_size=AUTOTUNE)


def plot_performance(history, n_epochs):
    '''Plot accuracy and loss for the training and validation sets.

    Arguments:
    history -- a Keras History object produced by a call to fit()
    n_epochs -- how many epochs to plot?
    '''

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(n_epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
