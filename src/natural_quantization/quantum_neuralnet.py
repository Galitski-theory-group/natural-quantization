import datetime
import json
import os

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeEncoder
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeKyiv

from natural_quantization.preprocess import read_weights

# TODO: enforce one classical data type
# TODO: make sure we select qubits with furthest connectivity
# TODO : pytorch mnist dataset


class QuantumNeuralNetwork:

    """
    A hybrid classical–quantum feedforward neural network built on Qiskit Runtime.

    This class implements a multi-layer network where the first and last layers
    are classical linear transforms and intermediate hidden layers are executed
    as parameterized single-qubit quantum circuits. Each hidden layer’s pre‑activation
    vector is converted into rotation angles, the qubits are measured, and the
    resulting bitstring is used as input to the next layer. The final layer applies
    a classical softmax to produce output probabilities.

    Attributes
    ----------
    layer_sizes : list of int
        Sizes of each layer, including input, hidden, and output layers.
    layer_n : int
        Total number of layers.
    hidden_layer_n : int
        Number of hidden (quantum) layers.
    b : list of np.ndarray
        Bias vectors for each layer (starting from the first hidden layer).
    W : list of np.ndarray
        Weight matrices for each layer transition.
    activation_f : callable
        A function mapping (pre‑activation, quantumness) → activation, e.g. `htanh`.
    service : QiskitRuntimeService or None
        Initialized IBM runtime service after calling `establish_communication_with_ibm`.
    backend : Qiskit backend or None
        Selected device or simulator for quantum circuit execution.
    comms : pass manager or None
        Communication pass manager for circuit compilation.
    """

    def __init__(
        self, layer_sizes: list, activation_f: callable = lambda x: x
    ) -> "QuantumNeuralNetwork":
        self.layer_sizes = layer_sizes
        self.layer_n = len(layer_sizes)
        self.hidden_layer_n = len(layer_sizes) - 2
        self.b = [
            np.random.randn(layer_sizes[i]) for i in range(1, self.hidden_layer_n + 2)
        ]
        self.W = [
            np.random.randn(layer_sizes[i], layer_sizes[i - 1])
            * np.sqrt(1 / layer_sizes[i - 1])
            for i in range(1, self.hidden_layer_n + 2)
        ]
        self.bias = self.b
        self.weights = self.W
        self.activation_f = activation_f
        self.service = None
        self.backend = None
        self.comms = None

    def establish_communication_with_ibm(
        self,
        simulation_mode=True,
        simulator=FakeKyiv,
        operational=True,
        optimization_level=3,
    ) -> None:
        """
        Initialize Qiskit runtime service, select a backend, and configure the communications
        pass manager.

        This method logs into the IBM Qiskit runtime service, chooses the least-busy backend
        supporting sessions and measurements (either a simulator or a real device), and
        generates a preset pass manager with the specified optimization level for compiling
        circuits.

        Parameters
        ----------
        simulation_mode : bool, optional
            Whether to prefer a simulator backend (True) or a real device (False). Default is
            True.
        operational : bool, optional
            Whether to restrict selection to operational (online) backends (True) or include
            offline ones (False). Default is True.
        optimization_level : int, optional
            The optimization level (0–3) for the communications pass manager.
            Higher values apply
              more aggressive optimizations. Default is 3.

        Raises
        ------
        ConnectionError
            If initializing QiskitRuntimeService or selecting the backend fails.
        RuntimeError
            If generating the communications pass manager fails.
        """

        try:
            # Initialize the runtime service (assumes you're logged in)
            self.service = QiskitRuntimeService()
        except Exception as e:
            raise ConnectionError("Failed to initialize QiskitRuntimeService.") from e

        try:
            if simulation_mode is True:
                fake_simulator = simulator
                self.backend = fake_simulator
            # Choose a backend with session and measurement support
            else:
                self.backend = self.service.least_busy(
                    simulator=simulation_mode, operational=operational
                )
        except Exception as e:
            raise ConnectionError("Failed to select a backend.") from e

        try:
            # Generate the communication pass manager
            self.comms = generate_preset_pass_manager(
                backend=self.backend, optimization_level=optimization_level
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to generate the communications pass manager."
            ) from e

    def feedforward(
        self: "QuantumNeuralNetwork",
        input: np.ndarray[:],
        quantumness=0.5,
        shots: int = 1,
        record_job_ids: bool = False,
    ) -> np.ndarray[:]:
        """
        Perform a hybrid classical–quantum feedforward pass through the network.

        This method processes the input vector through a classical linear layer,
        then for each hidden layer builds and runs a parameterized quantum circuit
        whose rotation angles are determined by the difference between the pre‑activation
        zs and the activations computed by `self.activation_f`. The measurement outcomes
        of each quantum circuit become the inputs to the next layer. Finally, it
        applies a softmax transformation on the last linear readout layer to produce
        output probabilities.

        Parameters
        ----------
        self : QuantumNeuralNetwork
            The network instance, which must have attributes `weights`, `activation_f`,
            `comms`, and `backend` configured.
        input : np.ndarray, shape (n_input,)
            The input feature vector for the network.
        quantumness : float, optional
            A value between 0 and 1 controlling the interpolation between purely
            classical (0) and fully quantum (1) hidden‑layer activations. Default is 0.5.

        Returns
        -------
        np.ndarray, shape (n_output,)
            The output probability vector from the final softmax layer.
        """

        sampler = Sampler(mode=self.backend)
        n_blocks = len(input) if isinstance(input, list) else 1
        print("number of blocks:", n_blocks)
        zs = input
        for i in range(0, len(self.weights) - 1):
            # print(f"layer {i}")
            if n_blocks == 1:
                θ = (
                    np.pi
                    / 2
                    * (1 - self.activation_f(self.weights[i] @ zs, quantumness))
                )
            else:
                θ = np.hstack(
                    [
                        np.pi
                        / 2
                        * (1 - self.activation_f(self.weights[i] @ zs[j], quantumness))
                        for j in range(n_blocks)
                    ]
                )

            print("angles:", θ)
            # print("thetas:",θ)
            n_qbits = self.weights[i].shape[0]
            print(f"total qubits: {n_qbits*n_blocks}")
            qr = QuantumRegister(n_qbits * n_blocks, name="qr")
            cr = ClassicalRegister(n_qbits * n_blocks, name="cr")
            qc = QuantumCircuit(qr, cr)
            for k, ϕ in enumerate(θ):
                qc.ry(theta=ϕ, qubit=k)
                qc.measure(qubit=k, cbit=k)
            isa_circuit = self.comms.run(qc)
            qc.draw("mpl", style="iqp")
            job = sampler.run([isa_circuit], shots=shots)

            result = job.result()
            # Optionally record job IDs
            if record_job_ids:
                timestamp = datetime.datetime.now().isoformat()
                path = f"data/tmp/jobs/{timestamp}_{i}.json"
                with open(path, "w") as f:
                    json.dump(result, f, cls=RuntimeEncoder)

            counts = result[0].data.cr.get_counts()
            # print("counts:", counts)
            # takes bit string e.g. '10010' and makes array [1,0,0,1,0]
            zs = np.fromiter((int(b) for b in next(iter(counts))), dtype=np.float16)
            zs = zs[::-1]
            # print("bit-string -> array:", zs)
            # set 0 to 1, 1 to -1
            zs = np.where(zs == 0, 1, -1)
            # print("activations:", zs)

        # apply softmax to final layer

        # print("final pre-activation:", zs)
        zs = (
            zs
            if n_blocks == 1
            else [zs[i : i + n_qbits] for i in range(0, len(zs), n_qbits)]
        )

        if n_blocks == 1:
            output = np.exp(self.weights[-1] @ zs) / sum(np.exp(self.weights[-1] @ zs))
        else:
            output = [
                np.exp(self.weights[-1] @ zs[i]) / sum(np.exp(self.weights[-1] @ zs[i]))
                for i in range(n_blocks)
            ]
        # print("output:",output)
        return output

    def simulated_feedforward(
        self: "QuantumNeuralNetwork", input: np.ndarray[:], quantumness=0.5
    ) -> np.ndarray[:]:
        """
        Perform a hybrid classical–quantum feedforward pass through the network.

        This method processes the input vector through a classical linear layer,
        then for each hidden layer builds and runs a parameterized quantum circuit
        whose rotation angles are determined by the difference between the pre‑activation
        zs and the activations computed by `self.activation_f`. The measurement outcomes
        of each quantum circuit become the inputs to the next layer. Finally, it
        applies a softmax transformation on the last linear readout layer to produce
        output probabilities.

        Parameters
        ----------
        self : QuantumNeuralNetwork
            The network instance, which must have attributes `weights`, `activation_f`,
            `comms`, and `backend` configured.
        input : np.ndarray, shape (n_input,)
            The input feature vector for the network.
        quantumness : float, optional
            A value between 0 and 1 controlling the interpolation between purely
            classical (0) and fully quantum (1) hidden‑layer activations. Default is 0.5.

        Returns
        -------
        np.ndarray, shape (n_output,)
            The output probability vector from the final softmax layer.
        """
        zs = input
        for i in range(0, len(self.weights) - 1):
            θ = np.pi / 2 * (1 - self.activation_f(self.weights[i] @ zs, quantumness))
            ϵ = np.random.rand(len(θ))
            zs = ((ϵ < (1 + np.cos(θ)) / 2) - 0.5) * 2

        # apply softmax to final layer
        output = np.exp(self.weights[-1] @ zs) / sum(np.exp(self.weights[-1] @ zs))
        return output

    def predict(
        self,
        Xs,
        quantumness=0.5,
        n_samples=10,
        n_instances=1,
        verbose: bool = False,
        collect_job_ids: bool = False,
        file: str = None,
    ) -> list[np.ndarray[:]]:
        """
        Generate multiple stochastic predictions for each input using the hybrid
        quantum–classical network.

        Parameters
        ----------
        Xs : Iterable[np.ndarray]
            A collection of input vectors to predict on. Each element should be a 1D numpy array
            matching the network's input dimension.
        quantumness : float, optional
            Interpolation parameter between purely classical (0.0) and fully quantum (1.0
            hidden-layer
            activations. Defaults to 0.5.
        n_samples : int, optional
            Number of stochastic forward passes to perform per input. Defaults to 10.

        Returns
        -------
        list[np.ndarray]
            A flat list of output probability vectors from all runs, with length equal to
            `len(Xs) * n_samples`. Each element is the softmax output of one forward pass.
        """

        f = open(file, "w") if verbose else None
        try:
            output = []
            n = n_instances
            chunks = Xs if n == 1 else [Xs[i : i + n] for i in range(0, len(Xs), n)]
            for chunk in chunks:
                per_instance = []
                for i in range(n_samples):
                    tmp = self.feedforward(
                        chunk, quantumness=quantumness, record_job_ids=collect_job_ids
                    )
                    argmaxes = (
                        np.argmax(tmp) if n_instances == 1 else np.argmax(tmp, axis=1)
                    )
                    print(argmaxes)
                    per_instance.append(argmaxes)
                    if f:
                        json.dump([list(chunk), list(tmp)], f, indent=2)
                output.append(per_instance)

        finally:
            if f:
                f.close()

        return output

    def simulated_predict(
        self,
        Xs,
        quantumness=0.5,
        n_samples=10,
        write_intermediate_data: bool = False,
        file: str = None,
    ) -> list[np.ndarray[:]]:
        """
        Generate multiple stochastic predictions for each input using the hybrid
        quantum–classical network.

        Parameters
        ----------
        Xs : Iterable[np.ndarray]
            A collection of input vectors to predict on. Each element should be a 1D numpy array
            matching the network's input dimension.
        quantumness : float, optional
            Interpolation parameter between purely classical (0.0) and fully quantum (1.0
            hidden-layer
            activations. Defaults to 0.5.
        n_samples : int, optional
            Number of stochastic forward passes to perform per input. Defaults to 10.

        Returns
        -------
        list[np.ndarray]
            A flat list of output probability vectors from all runs, with length equal to
            `len(Xs) * n_samples`. Each element is the softmax output of one forward pass.
        """

        output = []
        for x in Xs:
            per_instance = []
            for i in range(n_samples):
                tmp = self.simulated_feedforward(x, quantumness=quantumness)
                # print(f"shot #{i}", np.argmax(tmp))
                per_instance.append(np.argmax(tmp))
            output.append(per_instance)
        return output


def run_experiment(
    inputs: list[np.ndarray[:]],
    data_directory: str = "data",
    a_values: list[float] = [0.0, 0.1, 0.2, 0.5, 1.0],
    layer_widths: list[int] = [16],
    input_n: int = 784,
    output_n: int = 10,
    simulation_mode: bool = True,
    collect_job_ids: bool = False,
    verbose: bool = False,
    file: str = "/tmp.json",
    n_samples: int = 10,
    n_instances: int = 1,
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
            activation_f=lambda x: np.clip(x, -1, 1),
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
            n_instances=n_instances,
            verbose=verbose,
            file=file,
            collect_job_ids=collect_job_ids,
        )
        setup_results.append((a, width, results))

    return setup_results
