import datetime
import json
import os

import numpy as np
from qiskit_ibm_runtime.fake_provider import FakeKyiv

from natural_quantization.activations import htanh
from natural_quantization.preprocess import read_weights
from natural_quantization.quantum_neuralnet import QuantumNeuralNetwork

WORKING_DIR = "/Users/dlakhdar/physics/copy_repos/natural-quantization"
As = [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 5.0]
WIDTHS = [16]
INDICES = [6929]
N_SAMPLES = 96
N_INSTANCES_PER_BLOCK = 6


def run_experiment(
    inputs: list[np.ndarray[:]],
    data_directory: str = f"{WORKING_DIR}/data",
    a_values: list[float] = [0.0, 0.1, 0.2, 0.5, 1.0],
    layer_widths: list[int] = [16],
    input_n: int = 784,
    output_n: int = 10,
    simulation_mode: bool = True,
    save_bitstring_to: bool = False,
    save_results_to: str = f"{WORKING_DIR}/data/experiment_data/tmp.json",
    n_samples: int = 10,
    n_instances_per_block: int = 1,
    n_blocks: int = None,
    optimization_level: int = 0,
) -> list[tuple[int, int, list[list]]]:
    """

    Return: list[tuple[int,int,list[list]]]

    This is a list containing setup data. Outer list is collection of setups. The tuple contains
    setup a, setup width, and the data from that setup. The data list is for collection of images.
    Each image list has n_samples.

    """

    setup = []
    for a in a_values:
        for width in layer_widths:
            fname = (
                f"quantum_nn_rotation_angles/mnist_a{a}_lr-2_shots20_width{width}.json"
            )
            path = os.path.join(data_directory, fname)
            setup.append((path, a, width))

    setup_results = []

    for file_path, a, width in setup:
        print(f"a: {a} , l: {width}")
        # initialize network
        qnn = QuantumNeuralNetwork(
            layer_sizes=[input_n, width, width, width, output_n],
            activation_f=htanh,
        )
        # load and assign weights
        weights = read_weights(file=file_path)
        qnn.weights = [np.array(w) for w in weights]

        # ensure IBM connection
        qnn.establish_communication_with_ibm(
            simulation_mode=simulation_mode,
            simulator=FakeKyiv(),
            operational=True,
            optimization_level=optimization_level,
        )

        # run predictions
        results = qnn.sample_block_predict(
            inputs,
            quantumness=a,
            n_samples=n_samples,
            n_samples_per_block=n_instances_per_block,
            save_results_to=save_results_to,
            save_bitstring_to=save_bitstring_to,
        )
        setup_results.append([a, width, results])

    return setup_results


def main():
    with open(f"{WORKING_DIR}/data/mnist_data/mnist_data_reduced.json") as f:
        data = json.load(f)

    test_set = data["test_set"]
    # print(f"data:\n {test_set[0][0]}")
    x_test = [np.array(d[0]) for d in test_set]
    # indices = [0, 1, 2, 3, 4, 7, 11, 15, 18,
    #            30, 0, 1]

    indices = INDICES
    # index 0 -> 7, index, index 1 ->2, index 2 ->1 , index 3 -> 0 ,
    # index 4 -> 4 , index 7 -> 9
    # index 11 -> 6, index 15 -> 5, index 18 -> 8, index 30 -> 3
    x_test_sample, _ = [x_test[index] for index in indices]

    date = datetime.datetime.now()
    date = str(date).replace(" ", "_")

    results = run_experiment(
        x_test_sample,
        a_values=As,
        layer_widths=WIDTHS,
        simulation_mode=False,
        n_samples=N_SAMPLES,
        n_instances_per_block=N_INSTANCES_PER_BLOCK,
        save_bitstring_to=f"{WORKING_DIR}/data/experiment_data/\
        {date}_qubits_tmp.json",
        save_results_to=f"{WORKING_DIR}/data/experiment_data/\
        {date}_tmp_results.json",
    )

    with open(
        f"{WORKING_DIR}/data/\
              experiment_data/{date}_results.txt",
        "w",
    ) as f:
        print(results, file=f)


if __name__ == "__main__":
    main()
