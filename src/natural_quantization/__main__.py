import datetime
import json
import os
import numpy as np
from qiskit_ibm_runtime.fake_provider import FakeKyiv
from natural_quantization.activations import htanh
from natural_quantization.preprocess import read_weights, one_hot
from natural_quantization.quantum_neuralnet import QuantumNeuralNetwork

WORKING_DIR = "/Users/dlakhdar/physics/copy_repos/natural-quantization"
As = [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
WIDTHS = [16]
# index 0 -> 7, index, index 1 ->2, index 2 ->1 , index 3 -> 0 ,
# index 4 -> 4 , index 7 -> 9
# index 11 -> 6, index 15 -> 5, index 18 -> 8, index 30 -> 3
# indices = [0, 1, 2, 3, 4, 7, 11, 15, 18,
#            30, 0, 1]
# single image test
# INDICES = [6929]
# full sota
#sota = [1901,2130,2597,3422,6576]
sota = [1901]
INDICES = sota
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
    noise_level: int = 0,
    two_qubit_noise: bool = False
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
            noise=noise_level
        )

        # run predictions
        results = qnn.sample_block_predict(
            inputs,
            quantumness=a,
            n_samples=n_samples,
            n_samples_per_block=n_instances_per_block,
            save_results_to=save_results_to,
            save_bitstring_to=save_bitstring_to,
            noise_level=noise_level,
            two_qubit_noise=two_qubit_noise
        )
        setup_results.append([a, width, results])

    return setup_results


def main(working_dir=WORKING_DIR,
         save_to_dir: str = None,
         simulation_mode=True,
         As=As,
         widths=WIDTHS,
         indices=INDICES,
         n_samples=N_SAMPLES,
         n_instances_per_block=N_INSTANCES_PER_BLOCK,
         random_selection_n: int = None,
         noise_levels: int = [0]) -> None:
    with open(f"{working_dir}/data/mnist_data/mnist_data_reduced.json") as f:
        data = json.load(f)

    test_set = data["test_set"]
    # print(f"data:\n {test_set[0][0]}")
    x_test = [np.array(d[0]) for d in test_set]
                 
    if random_selection_n:
        indices = np.random.choice(list(range(0, len(x_test))),
                                   size=random_selection_n).tolist()
        date = datetime.datetime.now()
        date = str(date).replace(" ", "_")
        with open(f"{WORKING_DIR}/data/experiment_data/statistics/{date}_indices.txt", "w") as f:
            print(indices, file=f)

    x_test_sample = x_test if indices == "all" else [x_test[index] for index in indices]

    for i, sample in enumerate(x_test_sample):
        print("sample index:", indices[i])
        for noise_level in noise_levels:
            print(f"Running experiment for noise level: {noise_level}")
            date = datetime.datetime.now()
            date = str(date).replace(" ", "_")
            result = run_experiment(
                [sample],
                a_values=As,
                layer_widths=widths,
                simulation_mode=simulation_mode,
                n_samples=n_samples,
                n_instances_per_block=n_instances_per_block,
                save_bitstring_to=False,
                save_results_to=False,
                noise_level=noise_level,
            )

            if indices == "all":
                with open(
                    f"{working_dir}/data/experiment_data/statistics/run5/all_index_noise_{noise_level}_results.txt",
                    "w",
                ) as f:
                    print(result, file=f, end="\n")
                    f.flush()
                    os.fsync(f.fileno())
            else: 
                with open(
                    f"{working_dir}/data/experiment_data/noise_test/run4/{date}_index_{indices[i]}_noise_{noise_level}_results.txt",
                    "w",
                ) as f:
                    print(result, file=f)
                    f.flush()
                    os.fsync(f.fileno())
            
            
if __name__ == "__main__":
    main(simulation_mode=True,
         indices="all",
         As=[0.0,.1,.2,.5,1.0,1.5,2.0],
         n_samples=10,
         noise_levels=[0.0],
         n_instances_per_block=2)
