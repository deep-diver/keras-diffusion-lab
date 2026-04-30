"""Fashion-MNIST CNN classifier for FID feature extraction and accuracy evaluation.

Provides a simple CNN that trains to ~92% accuracy on Fashion-MNIST.
The penultimate Dense(128) layer serves as the feature extraction layer
for FID computation — domain-appropriate for 28x28 grayscale images
where InceptionV3 (299x299 RGB) is impractical.
"""

import numpy as np
import keras
from keras import layers


def build_classifier(image_size=28, channels=1, num_classes=10):
    """Build a CNN classifier for Fashion-MNIST.

    Architecture: Conv(32) -> Conv(64) -> Flatten -> Dense(128) -> Dense(10)
    The Dense(128) layer is the feature extraction layer for FID.

    Args:
        image_size: Input image size (default 28 for Fashion-MNIST).
        channels: Input channels (default 1 for grayscale).
        num_classes: Number of output classes.

    Returns:
        keras.Model with (batch, H, W, C) input in [-1, 1] range.
    """
    inputs = keras.Input(shape=(image_size, image_size, channels))

    # Normalize from [-1, 1] to [0, 1] for conv layers
    x = (inputs + 1.0) / 2.0

    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu', name='feature_layer')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name='fashion_mnist_classifier')
    return model


def build_feature_extractor(classifier):
    """Return a model that outputs 128-dim penultimate features.

    Args:
        classifier: A model built by build_classifier().

    Returns:
        keras.Model with same input, output shape (batch, 128).
    """
    feature_layer = classifier.get_layer('feature_layer')
    return keras.Model(
        classifier.inputs,
        feature_layer.output,
        name='feature_extractor'
    )


def train_classifier(images, labels, epochs=10, batch_size=128,
                     validation_split=0.1, save_path=None):
    """Train a Fashion-MNIST classifier.

    Args:
        images: (N, H, W, C) array in [-1, 1] range.
        labels: (N,) int array of class labels.
        epochs: Training epochs.
        batch_size: Batch size.
        validation_split: Fraction of data for validation.
        save_path: If set, save classifier weights to this path after training.

    Returns:
        Tuple of (classifier, feature_extractor, val_accuracy).
    """
    classifier = build_classifier(
        image_size=images.shape[1],
        channels=images.shape[-1],
        num_classes=int(labels.max()) + 1
    )
    classifier.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = classifier.fit(
        images, labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    )

    val_accuracy = history.history['val_accuracy'][-1]

    if save_path:
        classifier.save_weights(save_path)

    feature_extractor = build_feature_extractor(classifier)
    return classifier, feature_extractor, val_accuracy
