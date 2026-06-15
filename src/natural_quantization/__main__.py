import datetime
import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import typer
from qiskit_ibm_runtime.fake_provider import FakeKyiv, FakeTorino

from natural_quantization.activations import htanh
from natural_quantization.preprocess import read_weights
from natural_quantization.quantum_neuralnet import QuantumNeuralNetwork

# TODO: eliminate defaults in run_experiment
# TODO: add topology option
# TODO : add help strings
# TODO: better save data structure

WORKING_DIR = "/Users/dlakhdar/physics/copy_repos/natural-quantization"
EXPERIMENT_DIR = "data/experiment_data"
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
# sota = [1901,2130,2597,3422,6576]
sota = [1901]
INDICES = sota
N_SAMPLES = 96
N_INSTANCES_PER_BLOCK = 6

method_map = {
    "fake_kyiv": FakeKyiv(),
    "fake_torino": FakeTorino()
    # …etc…
}

app = typer.Typer()


def run_experiment(
    inputs: list[np.ndarray[:]],
    machine: str = "least_busy",
    data_directory: str = f"{WORKING_DIR}/data",
    a_values: list[float] = [0.0, 0.1, 0.2, 0.5, 1.0],
    layer_widths: list[int] = [16],
    input_n: int = 784,
    output_n: int = 10,
    simulation_mode: bool = True,
    simulator: "FakeBackend" = FakeTorino(),
    save_bitstring_to: bool = False,
    save_results_to: str = f"{WORKING_DIR}/data/experiment_data/tmp.json",
    n_samples: int = 10,
    n_instances_per_block: int = 1,
    n_blocks: int = None,
    optimization_level: int = 0,
    noise_level: int = 0,
    two_qubit_noise: bool = False,
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
        # TODO: will break unless layer widths is modularized
        nqbits = layer_widths[0] * n_instances_per_block if two_qubit_noise else None
        print("number qubits:", nqbits)
        qnn.establish_communication_with_ibm(
            machine=machine,
            simulation_mode=simulation_mode,
            simulator=simulator,
            operational=True,
            optimization_level=optimization_level,
            noise=noise_level,
            two_qubit_noise=two_qubit_noise,
            nqbits=nqbits,
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
            two_qubit_noise=two_qubit_noise,
        )
        setup_results.append([a, width, results])

    return setup_results


@app.command()
def main(
    result_dir: str = typer.Argument(..., help="Directory to save experiment results"),
    working_dir: str = typer.Option(
        WORKING_DIR, help="Base working directory containing data"
    ),
    experiment_dir: str = typer.Option(
        EXPERIMENT_DIR, help="Directory of experiment results"
    ),
    machine: str = typer.Option("least_busy", help="IBM machine to use"),
    simulation_mode: bool = typer.Option(
        True, help="Whether to run in simulation mode"
    ),
    simulator: str = typer.Option("fake_torino", help="simulator to use"),
    optimization_level: int = typer.Option(0, help="Optimization level for transpiler"),
    As: List[float] = typer.Option(
        [0.0, 0.1, 0.2, 0.5, 1.0], "--a-values", help="List of a values"
    ),
    indices: List[int] = typer.Option(
        [], "--indices", help="Indices of test samples, or empty for 'all'"
    ),
    random_selection_n: Optional[int] = typer.Option(
        None, help="Number of random samples to select"
    ),
    input_n: int = typer.Option(784, help="Input dimension size"),
    output_n: int = typer.Option(10, help="Output dimension size"),
    widths: List[int] = typer.Option([16], "--widths", help="Layer widths List"),
    n_samples: int = typer.Option(10, help="Number of samples per prediction"),
    n_instances_per_block: int = typer.Option(1, help="Instances per block"),
    noise_levels: List[int] = typer.Option(
        [0], "--noise-levels", help="List of noise levels"
    ),
    two_qubit_noise: bool = typer.Option(
        False, "--two-qubit-noise", is_flag=True, help="Enable two-qubit noise"
    ),
) -> None:
    mnist_file = Path(f"{working_dir}/data/mnist_data/mnist_data_reduced.json")
    result_directory = Path(os.path.join(working_dir, experiment_dir, result_dir))
    simulator = method_map.get(simulator)

    if not mnist_file.exists():
        typer.echo(f"Error: MNIST data file not found at {mnist_file}")
        raise typer.Exit(code=1)

    with open(mnist_file) as f:
        data = json.load(f)

    if not result_directory.exists():
        typer.echo(f"Creating {result_directory}...")
        os.mkdir(result_directory)

    test_set = data["test_set"]
    # print(f"data:\n {test_set[0][0]}")
    x_test = [np.array(d[0]) for d in test_set]

    # TODO: this logic is flawed, the indices will be overwritten by random selection
    if random_selection_n:
        indices = np.random.choice(
            list(range(0, len(x_test))), size=random_selection_n
        ).tolist()
        date = datetime.datetime.now()
        date = str(date).replace(" ", "_")
        idx_file = os.path.join(result_directory, f"{date}_indices.txt")

        with open(idx_file, "w") as f:
            print(indices, file=f)
        typer.echo(f"Randomly selected indices saved to {idx_file}")

    x_test_sample = x_test if indices == "all" else [x_test[index] for index in indices]

    for i, sample in enumerate(x_test_sample):
        print("sample index:", indices[i] if indices != "all" else i)
        for noise_level in noise_levels:
            print(f"Running experiment for noise level: {noise_level}")
            if two_qubit_noise:
                print(f"Running with two-qubit noise!")
            date = datetime.datetime.now()
            date = str(date).replace(" ", "_")
            result = run_experiment(
                [sample],
                machine=machine,
                a_values=As,
                layer_widths=widths,
                input_n=input_n,
                output_n=output_n,
                simulation_mode=simulation_mode,
                simulator=simulator,
                n_samples=n_samples,
                n_instances_per_block=n_instances_per_block,
                save_bitstring_to=False,
                save_results_to=False,
                noise_level=noise_level,
                two_qubit_noise=two_qubit_noise,
                optimization_level=optimization_level,
            )

            if indices == "all":
                all_result_file = os.path.join(
                    result_directory, f"all_indices_noise_{noise_level}_results.txt"
                )
                with open(all_result_file, "a") as f:
                    print(result, file=f, end="\n")
                    f.flush()
                    os.fsync(f.fileno())
            else:
                result_file = os.path.join(
                    result_directory,
                    f"{date}_index_{indices[i]}_noise_{noise_level}_results.txt",
                )
                with open(result_file, "w") as f:
                    print(result, file=f)
                    f.flush()
                    os.fsync(f.fileno())
                typer.echo(f"Result of index {indices[i]} saved!")


if __name__ == "__main__":
    # main(
    #     machine="ibm_sherbrooke",
    #     simulation_mode=False,
    #     indices=[6929],
    #     As=[0.0],
    #     n_samples=96,
    #     noise_levels=[100],
    #     two_qubit_noise=False,
    #     n_instances_per_block=6)
    app()
