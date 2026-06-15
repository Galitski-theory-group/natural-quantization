import json

import numpy as np

fire = 1
output_length = 10


def magnitude(x: np.ndarray) -> int | float:
    return np.sqrt(np.sum(x**2))


def max_normalize(data: np.ndarray):
    return data / np.max(data)


def gaussian_normalize(data: np.ndarray):
    return (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)


def one_hot(digit: int, length: int = 10) -> np.ndarray:
    """
    Return a one‑hot vector of given length with a 1 at position `digit`.

    Parameters
    ----------
    digit : int
        An integer between 0 and length‑1 inclusive.
    length : int
        Size of the output vector (default 10).

    Returns
    -------
    np.ndarray
        One‑hot encoded vector of shape (length,).
    """
    if not (0 <= digit < length):
        raise ValueError(f"digit must be in [0, {length-1}]")
    vec = np.zeros(length, dtype=int)
    vec[digit] = 1
    return vec


# def prepare_data(
#     dataset: "tf.keras.datasets" = "mnist", normalize_scheme: callable = max_normalize
# ) -> tuple[np.ndarray]:
#     """
#     Prepares and preprocesses a dataset for training and testing.

#     Parameters:
#         dataset (str): Name of the dataset to load from tf.keras.datasets (default: 'mnist').
#         normalize_scheme (function): Function to normalize the dataset
#         (default: max_normalize).

#     Returns:
#         tuple: Preprocessed training and testing data:
#             - x_train (np.array): Flattened and normalized training input data.
#             - y_train (np.array): One-hot encoded training labels.
#             - x_test (np.array): Flattened and normalized testing input data.
#             - y_test (np.array): One-hot encoded testing labels.
#     """
#     # Dynamically get the dataset
#     try:
#         dataset_module = getattr(tf.keras.datasets, dataset)
#     except AttributeError:
#         raise ValueError(f"Dataset '{dataset}' not found in tf.keras.datasets")
#     (x_train, y_train), (x_test, y_test) = dataset_module.load_data()
#     x_train, y_train = np.array(x_train, dtype=float), np.array(y_train, dtype=float)

#     # Take n number of 28*28 matrices and convert them to 784 vectors
#     (r, m, n), (rt, mt, nt) = x_train.shape, x_test.shape
#     dim_x, dim_xt = (r, m * n), (rt, mt * nt)
#     x_train, x_test = x_train.reshape(dim_x), x_test.reshape(dim_xt)

#     y_train, y_test = (
#         one_hot(y_train, output_length),
#         one_hot(y_test, output_length),
#     )

#     # normalize datasets
#     # x_train = (x_train - np.mean(x_train, axis=0)) / (np.std(x_train, axis=0) + 1e-8)
#     # x_test = (x_test - np.mean(x_test, axis=0)) / (np.std(x_test, axis=0) + 1e-8)
#     x_train, x_test = map(normalize_scheme, [x_train, x_test])

#     return x_train, y_train, x_test, y_test


def prepare_data_from_file(
    file_path: str = "/Users/dlakhdar/physics/copy_repos/natural-quantization/data/mnist_data/mnist_data_reduced.json",
    n_samples: int = None,
    indices: list[int] = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Load MNIST test data from JSON and optionally return only a subset.

    Args:
        file_path: path to the JSON file containing {"test_set":
        [[pixels, label], ...]}.
        n_samples: number of random examples to draw
        (ignored if `indices` provided).
        indices: explicit list of indices to select.

    Returns:
        Either the full (x_test, y_test) or the sampled (x_sample, y_sample).
    """
    # 1) load and build full test set
    with open(file_path, "r") as f:
        image_data = json.load(f)

    test_set = image_data["test_set"]
    x_test = [np.array(entry[0]) for entry in test_set]
    y_test = [one_hot(entry[1]) for entry in test_set]

    # 2) decide which indices to use
    if indices is not None:
        pick = indices
    elif n_samples is not None:
        pick = np.random.choice(len(x_test), size=n_samples, replace=False).tolist()
    else:
        return x_test, y_test

    # 3) slice out the subset
    x_sample = [x_test[i] for i in pick]
    y_sample = [y_test[i] for i in pick]
    return x_sample, y_sample


def read_weights(file, weight_label: str = "last_params") -> list[np.ndarray[:, :]]:
    """
    Load a list of weight matrices from a JSON file.

    Parameters
    ----------
    file : str or pathlib.Path
        Path to the JSON file that contains saved weight parameters.
    weight_label : str, optional
        The key in the JSON object under which the weight matrices are stored.
        Defaults to "last_params".

    Returns
    -------
    List[np.ndarray]
        A list of 2D NumPy arrays corresponding to the weight matrices
        loaded from the JSON. Each entry in the returned list is one matrix.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    KeyError
        If `weight_label` is not found in the JSON data.
    """

    with open(file, "r") as f:
        data = json.load(f)

    return data[weight_label]
