import itertools

import numpy as np
import scipy.stats as stats

files = ["/data/experiment_data/image_6929_results.txt"]
As = [0.0, 0.1, 0.2, 0.5, 1.0, 1.5]

# index 0 -> 7, index, index 1 ->2, index 2 ->1 , index 3 -> 0 , index 4 -> 4 , index 7 -> 9
# index 11 -> 6, index 15 -> 5, index 18 -> 8, last_index -> 3
correct_vector = [7, 2, 1, 0, 4, 9, 6, 5, 8, 3]


def binomial_error(p: (int | float), N: (int | float)) -> float:
    """
    Calculate error of binomial experiment.

    Parameters:
    -p : probability frequency_success/total_trials
    -N : number of trials

    Returns:
    -errors: for the set-up

    """
    return (p * (1 - p) / N) ** (1 / 2)


def calculate_validation_rate(
    predicted_y: list[np.ndarray], y: list[np.ndarray]
) -> float | int:
    """
    Calculate the validation rate (accuracy) for predicted and actual labels.

    Parameters:
    - predicted_y: array-like, predicted probabilities or logits (e.g., from
    a neural network).
    - y: array-like, one-hot encoded true labels.

    Returns:
    - float, the accuracy rate as the proportion of correctly predicted
      samples.
    """
    predicted_indices = np.array(list(map(np.argmax, predicted_y)))
    true_indices = np.array(list(map(np.argmax, y)))
    accuracy = np.mean(predicted_indices == true_indices)
    return accuracy


def mode_of_nested_list(list_of_lists: list[list[list[float]]]) -> list[int]:
    """
    For each sub‐list `l` in `list_of_lists`:
      1. Compute argmax of each inner list `l'`.
      2. Take the statistical mode of those argmaxes.
    Returns a list of one mode‐index per `l`.
    """
    modes = []
    for l_ in list_of_lists:
        mode_idx = int(stats.mode(l_)[0])
        modes.append(mode_idx)
    return modes


def mode_of_argmaxes(list_of_lists: list[list[list[float]]]) -> list[int]:
    """
    For each sub‐list `l` in `list_of_lists`:
      1. Compute argmax of each inner list `l'`.
      2. Take the statistical mode of those argmaxes.
    Returns a list of one mode‐index per `l`.
    """
    modes = []
    for l_ in list_of_lists:
        # 1) argmax of each l'
        argmaxes = [int(np.argmax(l_prime)) for l_prime in l_]
        # 2)most common
        mode_idx = int(stats.mode(argmaxes)[0])
        modes.append(mode_idx)
    return modes


def error_of_argmaxes(list_of_lists: list[list[list[float]]]) -> list[int]:
    """
    For each sub‐list `l` in `list_of_lists`:
      1. Compute argmax of each inner list `l'`.
      2. Take the statistical mode of those argmaxes.
    Returns a list of one mode‐index per `l`.
    """
    modes = []
    for l_ in list_of_lists:
        # 1) argmax of each l'
        argmaxes = [int(np.argmax(l_prime)) for l_prime in l_]
        # 2)most common
        mode_idx = int(stats.mode(argmaxes)[0])
        modes.append(mode_idx)
    return modes


def concat_positionwise(seven_lists):
    """
    Given a list of length-N, where each element is itself a list of 7 lists,
    returns a new list of 7 lists where position i is the concatenation of
    all the i-th lists from each element of seven_lists.
    """
    # zip(*seven_lists) will yield 7 tuples; each tuple is (x[0], y[0], z[0], …), then (x[1], y[1], z[1], …), etc.
    result = []
    for group in zip(*seven_lists):
        # `group` is a tuple of lists: (first_list_from_all, second_list_from_all, …)
        # chain.from_iterable flattens it into a single iterator; then list(…) makes it back into a list.
        concatenated = list(itertools.chain.from_iterable(group))
        result.append(concatenated)
    return result


def run_single_image_classification_analysis(As=As, files=files, true_index=2):
    """
    Compute per‐sample classification accuracy and binomial error from saved
    prediction outputs.

    Parameters
    ----------
    As : list or array-like
        (Currently unused) Placeholder for arrays of class scores or
        probabilities returned by the classifier for each sample.
    files : list of str
        Paths to text files. Each file should contain a Python literal that
        evaluates to as list of triples `(score, predicted_class_index,
        list_of_ndarrays)`.
    true_index : int, optional
        The ground-truth class index to compare against (default is 2).

    Returns
    -------
    accuracies : list of float
        For each file, the fraction of examples whose predicted index equals
        `true_index`.
    errors : list of float
        The binomial error estimate for each accuracy, computed via
          `binomial_error`.

    Notes
    -----
    Each input file is read and `eval`-ed in a restricted namespace where
    only `array = np.array` is available.  The data are flattened and
    concatenated before computing accuracy and error.
    """

    accuracies = []
    errors = []
    data = []

    for file in files:
        # 1) Read the file
        with open(file, "r") as f:
            text = f.read()

        # 2) Evaluate it, giving `array` in the namespace, it becomes np.array
        tmp = eval(
            text,
            {"__builtins__": None},  # disable built-ins for safety
            {"array": np.array},  # map the name 'array' to numpy's array
        )

        data.append(tmp)
    # 3) Now `data` is a list of [float, int, list_of_ndarrays]
    #    with all types preserved.

    # 1. Read the raw text
    # concatentae lists
    data = sum(data, [])

    new_data = []
    for first, second, arr_list in data:
        concatenated = np.concatenate(arr_list).flatten()
        new_data.append([first, second, concatenated])

    for d in new_data:
        predicted_indices = d[2]
        N = len(predicted_indices)
        true_indices = np.zeros(len(predicted_indices))
        true_indices = true_indices.fill(true_index)
        accuracy = np.mean(predicted_indices == true_indices)
        error = binomial_error(accuracy, N)
        accuracies.append(accuracy)
        errors.append(error)

    return accuracies, errors
