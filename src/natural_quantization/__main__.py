
import json 
import numpy as np 
import os 
import datetime
from qiskit_ibm_runtime.fake_provider import FakeKyiv
from natural_quantization.quantum_neuralnet import QuantumNeuralNetwork
from natural_quantization.preprocess import one_hot,read_weights
from natural_quantization.activations import htanh

WORKING_DIR="/scratch/zt1/project/galitski-prj/user/dlakhdar/repos/natural-quantization"

def run_experiment(
    inputs: list[np.ndarray[:]],
    data_directory: str = f"{WORKING_DIR}/data/zaratan",
    a_values: list[float] = [0.0, 0.1, 0.2, 0.5, 1.0],
    layer_widths: list[int] = [16],
    input_n: int = 784,
    output_n: int = 10,
    simulation_mode: bool = True,
    save_bitstring_to: bool = False,
    verbose: bool = False,
    save_results_to: str = f"{WORKING_DIR}/data/zaratan/tmp.json",
    n_samples: int = 10,
    n_instances_per_block: int = 1,
    n_blocks:int = None,
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
                f"mnist_a{a}_lr-2_shots20_width{width}.json"
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
            simulation_mode=simulation_mode, simulator=FakeKyiv(), operational=True
        )

        # run predictions
        results = qnn.predict(
            inputs,
            quantumness=a,
            n_samples=n_samples,
            n_instances_per_block=n_instances_per_block,
            verbose=verbose,
            save_results_to=save_results_to,
            save_bitstring_to=save_bitstring_to,
        )
        setup_results.append([a, width, results])

    return setup_results

if __name__ == "__main__":

    with open(f"{WORKING_DIR}/data/zaratan/mnist_data_reduced.json") as f :
        data = json.load(f)

    test_set =  data["test_set"]
    print(f"data:\n {test_set[0][0]}")
    x_test, y_test = [np.array(d[0]) for d in test_set], [one_hot(d[1]) for d in test_set]
    indices = [0,1,2,3,4,7,11,15,18,30,]
    # index 0 -> 7, index, index 1 ->2, index 2 ->1 , index 3 -> 0 , index 4 -> 4 , index 7 -> 9
    # index 11 -> 6, index 15 -> 5, index 18 -> 8
    x_test_sample = [x_test[index] for index in indices]
    y_test_sample = [y_test[index] for index in indices]

    date = datetime.datetime.now()
    date = str(date).replace(" ","_")

    results = run_experiment(x_test_sample,
                             a_values=[.5],
                            layer_widths=[16],
                            verbose=True, 
                            simulation_mode=True, 
                            n_samples=5,
                            n_instances_per_block=2,
                            save_bitstring_to=f"{WORKING_DIR}/data/zaratan/{date}_qubits_tmp.json",
                            save_results_to=f"{WORKING_DIR}/data/zaratan/{date}_tmp_results.json")


    with open(f"{WORKING_DIR}/data/zaratan/{date}_results.txt", "w") as f1, open(f"{WORKING_DIR}/data/zaratan/{date}_results.json","w") as f2:
        print(results, file=f1); json.dumps(results,f2)