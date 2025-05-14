import json 
import numpy as np 
import scipy.stats as stats 
import matplotlib.pyplot as plt 


# index 0 -> 7, index, index 1 ->2, index 2 ->1 , index 3 -> 0 , index 4 -> 4 , index 7 -> 9
# index 11 -> 6, index 15 -> 5, index 18 -> 8, last_index -> 3 
correct_vector = [7,2,1,0,4,9,6,5,8,3]

def calculate_validation_rate(predicted_y: list[np.ndarray],
                              y: list[np.ndarray]) -> float | int:
    """
    Calculate the validation rate (accuracy) for predicted and actual labels.

    Parameters:
    - predicted_y: array-like, predicted probabilities or logits (e.g., from a neural network).
    - y: array-like, one-hot encoded true labels.

    Returns:
    - float, the accuracy rate as the proportion of correctly predicted samples.
    """
    predicted_indices = np.array(list(map(np.argmax, predicted_y)))
    true_indices = np.array(list(map(np.argmax, y)))
    accuracy = np.mean(predicted_indices == true_indices)
    return accuracy

def mode_of_argmaxes(list_of_lists: list[list[list[float]]]) -> list[int]:
    """
    For each sub‐list `l` in `list_of_lists`:
      1. Compute argmax of each inner list `l'`.
      2. Take the statistical mode of those argmaxes.
    Returns a list of one mode‐index per `l`.
    """
    modes = []
    for l in list_of_lists:
        # 1) argmax of each l'
        argmaxes = [int(np.argmax(l_prime)) for l_prime in l]
        # 2) most common
        mode_idx = int(stats.mode(argmaxes)[0])
        modes.append(mode_idx)
    return modes
