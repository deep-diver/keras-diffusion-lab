"""Dataset loading and preprocessing.

Supports CIFAR-10, Fashion-MNIST, and MNIST with [-1, 1] normalization.
All datasets are loaded via keras.datasets and normalized consistently.
"""

import numpy as np


def load_dataset(name: str = "cifar10", subset_size: int = None,
                 seed: int = 42, return_labels: bool = False):
    """Load an image dataset, normalized to [-1, 1].

    Args:
        name: Dataset name. One of "cifar10", "fashion_mnist", "mnist".
        subset_size: If set, randomly sample this many images.
        seed: Random seed for subset selection.
        return_labels: If True, return (images, labels) tuple.

    Returns:
        If return_labels is False: numpy array (N, H, W, C) in float32, range [-1, 1].
        If return_labels is True: tuple (images, labels) where labels is (N,) int array.
    """
    import keras

    loaders = {
        "cifar10": keras.datasets.cifar10.load_data,
        "fashion_mnist": keras.datasets.fashion_mnist.load_data,
        "mnist": keras.datasets.mnist.load_data,
    }

    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(loaders.keys())}")

    (x_train, y_train), _ = loaders[name]()
    y_train = y_train.squeeze().astype("int32")  # Flatten from (N, 1) to (N,)

    # Normalize to [-1, 1]
    x_train = x_train.astype("float32") / 127.5 - 1.0

    # Add channel dimension for grayscale
    if x_train.ndim == 3:
        x_train = x_train[..., np.newaxis]

    # Subset
    if subset_size is not None and subset_size < len(x_train):
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(x_train), subset_size, replace=False)
        x_train = x_train[indices]
        y_train = y_train[indices]

    if return_labels:
        return x_train, y_train
    return x_train


def denormalize(images: np.ndarray) -> np.ndarray:
    """Convert images from [-1, 1] to [0, 255] uint8."""
    return np.clip(images * 127.5 + 127.5, 0, 255).astype(np.uint8)


def make_dataset(images: np.ndarray, batch_size: int, shuffle: bool = True,
                 seed: int = 0) -> callable:
    """Create an infinite batched data iterator.

    Args:
        images: numpy array of shape (N, H, W, C).
        batch_size: Batch size.
        shuffle: Whether to shuffle each epoch.
        seed: Random seed.

    Returns:
        Callable that returns the next batch when called.
    """
    rng = np.random.RandomState(seed)
    n = len(images)
    idx = 0

    def next_batch():
        nonlocal idx, rng
        if idx == 0 and shuffle:
            rng.shuffle(images)
        if idx + batch_size > n:
            idx = 0
            if shuffle:
                rng.shuffle(images)
        batch = images[idx:idx + batch_size]
        idx += batch_size
        return batch

    return next_batch


def make_dataset_with_labels(images: np.ndarray, labels: np.ndarray,
                             batch_size: int, shuffle: bool = True,
                             seed: int = 0) -> callable:
    """Create an infinite batched data iterator that yields (images, labels).

    Args:
        images: numpy array of shape (N, H, W, C).
        labels: numpy array of shape (N,) with integer class labels.
        batch_size: Batch size.
        shuffle: Whether to shuffle each epoch.
        seed: Random seed.

    Returns:
        Callable that returns (image_batch, label_batch) when called.
    """
    rng = np.random.RandomState(seed)
    n = len(images)
    idx = 0

    def next_batch():
        nonlocal idx, rng
        if idx == 0 and shuffle:
            perm = rng.permutation(n)
            shuffled_images = images[perm]
            shuffled_labels = labels[perm]
        else:
            shuffled_images = images
            shuffled_labels = labels
        if idx + batch_size > n:
            idx = 0
            if shuffle:
                perm = rng.permutation(n)
                shuffled_images = images[perm]
                shuffled_labels = labels[perm]
        batch_images = shuffled_images[idx:idx + batch_size]
        batch_labels = shuffled_labels[idx:idx + batch_size]
        idx += batch_size
        return batch_images, batch_labels

    return next_batch


def get_dataset_info(name: str) -> dict:
    """Return dataset metadata."""
    info = {
        "cifar10": {"image_size": 32, "channels": 3, "train_size": 50000, "num_classes": 10},
        "fashion_mnist": {"image_size": 28, "channels": 1, "train_size": 60000, "num_classes": 10},
        "mnist": {"image_size": 28, "channels": 1, "train_size": 60000, "num_classes": 10},
    }
    if name not in info:
        raise ValueError(f"Unknown dataset: {name}")
    return info[name]
