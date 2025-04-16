
import tensorflow as tf
import numpy as np 
import json 
from numba import njit, int32, int64, float32, float64

fire = 1
output_length = 10

@njit
def magnitude(x: np.ndarray) -> int | float:
    return np.sqrt(np.sum(x**2))

@njit
def max_normalize(data: np.ndarray):
    return data / np.max(data)


@njit
def gaussian_normalize(data: np.ndarray):
    return (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)


def hot_encode(x: np.ndarray, output_length: (int | float)) -> np.ndarray:
    """
    Converts an array of integer indices into one-hot encoded vectors.

    Parameters:
        x (np.array): Array of integer indices.
        output_length (int): Length of the one-hot encoded vectors.

    Returns:
        np.array: A 2D array where each row is a one-hot encoded vector
        corresponding to the input indices.
    """
    tmp = []
    for index in x:
        x_vec = np.zeros(output_length)
        x_vec[int(index)] = fire
        tmp.append(x_vec.reshape((output_length)))
    return np.array(tmp)


def prepare_data(
    dataset: "tf.keras.datasets" = "mnist", normalize_scheme: callable = max_normalize
) -> tuple[np.ndarray]:
    """
    Prepares and preprocesses a dataset for training and testing.

    Parameters:
        dataset (str): Name of the dataset to load from tf.keras.datasets (default: 'mnist').
        normalize_scheme (function): Function to normalize the dataset
        (default: max_normalize).

    Returns:
        tuple: Preprocessed training and testing data:
            - x_train (np.array): Flattened and normalized training input data.
            - y_train (np.array): One-hot encoded training labels.
            - x_test (np.array): Flattened and normalized testing input data.
            - y_test (np.array): One-hot encoded testing labels.
    """
    # Dynamically get the dataset
    try:
        dataset_module = getattr(tf.keras.datasets, dataset)
    except AttributeError:
        raise ValueError(f"Dataset '{dataset}' not found in tf.keras.datasets")
    (x_train, y_train), (x_test, y_test) = dataset_module.load_data()
    x_train, y_train = np.array(x_train, dtype=float), np.array(y_train, dtype=float)

    # Take n number of 28*28 matrices and convert them to 784 vectors
    (r, m, n), (rt, mt, nt) = x_train.shape, x_test.shape
    dim_x, dim_xt = (r, m * n), (rt, mt * nt)
    x_train, x_test = x_train.reshape(dim_x), x_test.reshape(dim_xt)

    y_train, y_test = (
        hot_encode(y_train, output_length),
        hot_encode(y_test, output_length),
    )

    # normalize datasets
    # x_train = (x_train - np.mean(x_train, axis=0)) / (np.std(x_train, axis=0) + 1e-8)
    # x_test = (x_test - np.mean(x_test, axis=0)) / (np.std(x_test, axis=0) + 1e-8)
    x_train, x_test = map(normalize_scheme, [x_train, x_test])

    return x_train, y_train, x_test, y_test


def read_weights(file, weight_label: str = "last_params") ->list[np.ndarray[:,:]]:
    with open(file) as f:
        data = json.load(f)

    return data[weight_label]
