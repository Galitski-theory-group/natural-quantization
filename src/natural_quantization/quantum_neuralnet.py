import json
import os
from collections import defaultdict

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler import generate_preset_pass_manager

# noise‐enabled simulator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeEncoder
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeKyiv, FakeSherbrooke

# TODO: enforce one classical data type
# TODO: make sure we select qubits with furthest connectivity
# TODO : pytorch mnist dataset
# TODO : add number of blocks per instance
# TODO : more robust manipulation of noise
# TODO: add noise injection unitary option
# TODO: add two qubit noise option unitary


def greedy_unique_pairs(coupling):
    """
    coupling: list of [a,b] edges indicating undirected connections.
    Returns a list of disjoint pairs [(i,j), ...] with i < j,
    so each qubit appears at most once and no duplicate directions.
    """
    # Build undirected adjacency
    adj = defaultdict(list)
    nodes = set()
    for a, b in coupling:
        adj[a].append(b)
        adj[b].append(a)
        nodes.add(a)
        nodes.add(b)
    matched = set()
    pairs = []
    # Iterate through nodes in ascending order for determinism
    for i in sorted(nodes):
        if i in matched:
            continue
        # Try to match i with smallest-index neighbor that is not yet matched
        for j in sorted(adj[i]):
            if j not in matched:
                # record pair with smaller first
                if i < j:
                    pairs.append((i, j))
                else:
                    pairs.append((j, i))
                matched.add(i)
                matched.add(j)
                break
        # if no unmatched neighbor, i remains unmatched (not in any pair)
    return pairs


def find_qbit_map_wrt_pass_manager(simulator, nqbits):
    """ """
    coupling = simulator.configuration().coupling_map
    coupling_map = greedy_unique_pairs(coupling)[: nqbits // 2]
    qubit_map = []

    for q0, q1 in coupling_map:
        qubit_map.extend([q0, q1])

    return qubit_map


# class NeuralNetwork:
#     """
#     A class for constructing and training a fully connected neural network.

#     Attributes:
#         hidden_layer_n (int): Number of hidden layers in the network.
#         layer_n (int): Total number of layers (input + hidden + output).
#         layer_sizes (np.array): List of sizes for each layer in the network.
#         bias (list): List of bias vectors for each layer.
#         weights (list): List of weight matrices connecting the layers.
#         activation_f (callable): Activation function (default: sigmoid).
#         activation_df (callable): Derivative of the activation function.
#         cost_function (callable): Cost function for training (if provided).

#     Methods:
#         train(minibatch=True, minibatch_pool=10, iterations=100, η=1e-6) -> 'NeuralNetwork':
#             Trains the neural network using gradient descent.

#     Parameters:
#         hidden_layer (int): Number of hidden layers in the network.
#         layer_sizes (list[int | float]): List of hidden layer sizes (default: [10]).
#         activation_function (callable): Activation function for all layers (default: sigmoid).
#         activation_derivative (callable): Derivative of the activation function (default: derivative_sigmoid).
#         cost (callable): Cost function to minimize during training (optional).
#         cost_grad (callable): Cost function gradient with respect to activations solely

#     Train Method Parameters:
#         minibatch (bool): Whether to use mini-batch gradient descent (default: True).
#         minibatch_pool (int | float): Number of samples per mini-batch (default: 10).
#         iterations (int | float): Number of training iterations (default: 100).
#         η (int | float): Learning rate for gradient descent (default: 1e-6).

#     Returns:
#         NeuralNetwork: The trained neural network object.

#     Example:
#         nn = NeuralNetwork(
#                            layer_sizes=[10,64, 32,1],
#                            activation_function=sigmoid,
#                            activation_derivative=derivative_sigmoid)
#         nn.train(minibatch=True, minibatch_pool=32, iterations=1000, η=0.01)
#     """

#     def __init__(
#         self,
#         layer_sizes: (list[int] | list[float]),
#         activation_function: callable,
#         activation_derivative: callable,
#         cost_function: callable,
#         cost_grad: callable,
#     ) -> "NeuralNetwork":
#         self.layer_sizes = layer_sizes
#         self.layer_n = len(self.layer_sizes)
#         self.hidden_layer_n = len(self.layer_sizes) - 2
#         self.bias = [
#             np.random.randn(self.layer_sizes[i])
#             for i in range(1, self.hidden_layer_n + 2)
#         ]

#         self.weights = [
#             np.random.randn(self.layer_sizes[i], self.layer_sizes[i - 1])
#             * np.sqrt(1 / self.layer_sizes[i - 1])
#             for i in range(1, self.hidden_layer_n + 2)
#         ]

#         self.activation_f = activation_function
#         self.activation_df = activation_derivative
#         self.cost = cost_function
#         self.cost_grad = cost_grad

#     def __str__(self):
#         """
#         Returns a string representation of the neural network's architecture, weights, and biases.
#         """
#         display_str = "Neural Network Structure:\n"
#         display_str += f"Number of Layers: {self.layer_n}\n"
#         display_str += f"Hidden Layers: {self.hidden_layer_n}\n"
#         display_str += "Layer Sizes: " + " -> ".join(map(str, self.layer_sizes)) + "\n\n"

#         display_str += "Biases:\n"
#         for i, bias in enumerate(self.bias, start=1):
#             display_str += f"  Layer {i + 1}: Shape {bias.shape}\n"

#         display_str += "\nWeights:\n"
#         for i, weight in enumerate(self.weights, start=1):
#             display_str += f"  Layer {i}: Shape {weight.shape}\n"

#         display_str += f"\nActivation Function: {self.activation_f.__name__}\n"
#         display_str += f"Activation Derivative: {self.activation_df.__name__}\n"
#         display_str += f"Cost Function: {self.cost.__name__}\n"
#         display_str += f"Cost Gradient: {self.cost_grad.__name__}\n"

#         return display_str

#     def train(
#         self,
#         input,
#         output,
#         momentum: (int | float) = None,
#         minibatch: bool = True,
#         minibatch_pool: (int | float) = 10,
#         iterations: (int | float) = 100,
#         η: (int | float) = 1e-6,
#     ) -> None:
#         """
#         Trains the neural network using gradient descent.

#         Parameters:
#             minibatch (bool): Whether to use mini-batch gradient descent (default: True).
#             minibatch_pool (int | float): Size of the mini-batch for training (default: 10).
#             iterations (int | float): Number of training iterations (default: 100).
#             η (int | float): Learning rate for gradient descent (default: 1e-6).

#         Returns:
#             NeuralNetwork: The trained neural network object.

#         Description:
#             - Implements forward propagation for each input to compute activations.
#             - Performs backpropagation to compute gradients for weights and biases.
#             - Updates weights and biases using gradient descent.
#             - Supports mini-batch gradient descent if `minibatch` is set to True.

#         Example:
#             nn.train(minibatch=True, minibatch_pool=32, iterations=1000, η=0.01)
#         """

#         for _ in range(iterations):
#             if minibatch:
#                 indexes = np.random.choice(input.shape[0], size=minibatch_pool)
#                 X, Y = input[indexes], output[indexes]
#             else:
#                 X, Y = input, output

#             w_grads = [np.zeros(matrix.shape) for matrix in self.weights]

#             b_grads = [np.zeros(vector.shape) for vector in self.bias]

#             if momentum is not None:
#                 v = [np.zeros(w.shape) for w in self.weights]

#             # iterate for each set of x and y
#             # find zs and as (pre-act and activation)
#             for x, y in zip(X, Y):
#                 # print("x id:",id(x))

#                 # def feedforward
#                 z0 = self.weights[0] @ x + self.bias[0]
#                 zs = [z0]
#                 a0 = self.activation_f(z0)
#                 activations = [a0]
#                 for l in range(1, self.layer_n - 1, 1):
#                     # print("layers:",l,l-1)
#                     zl = self.weights[l] @ activations[l - 1] + self.bias[l]
#                     activation = self.activation_f(zl)
#                     # print("activation:", activation)
#                     zs.append(zl)
#                     activations.append(activation)

#                 z_output = zs[-1]
#                 a_output = activations[-1]
#                 output_error = self.cost_grad(a_output, y) * self.activation_df(
#                     z_output
#                 )
#                 errors = [output_error]
#                 for l in range(self.hidden_layer_n, 0, -1):
#                     error = (
#                         self.weights[l].T @ errors[-1] * self.activation_df(zs[l - 1])
#                     )
#                     errors.append(error)

#                 errors.reverse()
#                 # compute sum of error
#                 for l in range(0, self.hidden_layer_n + 1, 1):
#                     # w_grads[l] += errors[l]@activations[l].T
#                     w_grads[l] += np.outer(
#                         errors[l], activations[l - 1] if l > 0 else x
#                     )
#                     # print(w_grads)
#                     b_grads[l] += errors[l]
#                     # print(b_grads)

#             # gradient descent
#             if momentum is None:
#                 for l in range(0, self.hidden_layer_n + 1):
#                     self.weights[l] -= η / minibatch_pool * w_grads[l]
#                     self.bias[l] -= η / minibatch_pool * b_grads[l]
#             else:
#                 γ = momentum
#                 for l in range(0, self.hidden_layer_n + 1):
#                     v[l] = γ * v[l] + η / minibatch_pool * w_grads[l]
#                     self.weights[l] -= v[l]
#                     self.bias[l] -= η / minibatch_pool * b_grads[l]

#     def predict(self, input: list[np.ndarray]) -> list[np.ndarray]:
#         """
#         Predicts the output for a given input using the trained neural network.

#         Parameters:
#             input (np.array): Input data to predict, where each row corresponds to a single input instance.

#         Returns:
#             list: A list of predictions where each prediction corresponds to the output of the neural network
#                   for the corresponding input instance.

#         Description:
#             - Performs forward propagation through the network to compute the output layer activations.
#             - Returns the final layer activations as predictions.

#         Example:
#             predictions = nn.predict(x_test)
#         """

#         results = []
#         for x in input:
#             # print(x,"\n")
#             z0 = self.activation_f(self.weights[0] @ x + self.bias[0])
#             activations = [z0]
#             for l in range(1, self.layer_n - 1):
#                 zl = self.weights[l] @ activations[l - 1] + self.bias[l]
#                 a = self.activation_f(zl)
#                 activations.append(a)

#             results.append(activations[-1])

#         return results


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

    def __str__(self):
        """
        Returns a string representation of the neural network's architecture, weights, and biases.
        """
        display_str = "Quantum Neural Network Structure:\n"
        display_str += f"Number of Layers: {self.layer_n}\n"
        display_str += f"Hidden Layers: {self.hidden_layer_n}\n"
        display_str += (
            "Layer Sizes: " + " -> ".join(map(str, self.layer_sizes)) + "\n\n"
        )

        display_str += "Biases:\n"
        for i, bias in enumerate(self.bias, start=1):
            display_str += f"  Layer {i + 1}: Shape {bias.shape}\n"

        display_str += "\nWeights:\n"
        for i, weight in enumerate(self.weights, start=1):
            display_str += f"  Layer {i}: Shape {weight.shape}\n"

        display_str += f"\nActivation Function: {self.activation_f.__name__}\n"
        display_str += f"Activation Derivative: {self.activation_df.__name__}\n"
        display_str += f"Cost Function: {self.cost.__name__}\n"
        display_str += f"Cost Gradient: {self.cost_grad.__name__}\n"

        return display_str

    def establish_communication_with_ibm(
        self,
        simulation_mode=True,
        simulator=FakeSherbrooke(),
        machine: str = "least_busy",
        operational=True,
        optimization_level=0,
        noise=False,
        two_qubit_noise=False,
        nqbits=None,
    ) -> None:
        """
        Initialize Qiskit runtime service, select a backend, and configure
        the communications pass manager.

        This method logs into the IBM Qiskit runtime service, chooses the
        least-busy backend supporting sessions and measurements (either a
        simulator or a real device), and generates a preset pass manager with
        the specified optimization level for compiling circuits.

        Parameters
        ----------
        simulation_mode : bool, optional
            Whether to prefer a simulator backend (True) or a real device
            (False). Default is
            True.
        operational : bool, optional
            Whether to restrict selection to operational (online) backends
            (True) or include
            offline ones (False). Default is True.
        optimization_level : int, optional
            The optimization level (0–3) for the communications pass manager.
            Higher values apply
              more aggressive optimizations. Default is 3.

        Raises
        ------
        ConnectionError
            If initializing QiskitRuntimeService or selecting the backend
            fails.
        RuntimeError
            If generating the communications pass manager fails.
        """

        # Simulation branch
        if simulation_mode:
            print(f"Simulation mode: backend {simulator}...")
            if not noise:
                print("No artificial noise added!")
                fake = simulator
                self.backend = AerSimulator.from_backend(fake, noise_model=None)
                self.comms = generate_preset_pass_manager(
                    backend=self.backend, optimization_level=optimization_level
                )

                return  # skip IBM Runtime entirely

            elif noise and not two_qubit_noise:
                print("Single qubit noise on...")
                fake = simulator
                noise_model = NoiseModel.from_backend(fake)
                self.backend = AerSimulator(
                    noise_model=noise_model,
                    basis_gates=noise_model.basis_gates,
                    coupling_map=fake.configuration().coupling_map,
                )
                self.comms = generate_preset_pass_manager(
                    backend=self.backend, optimization_level=optimization_level
                )

                return  # skip IBM Runtime entirely
            elif noise and two_qubit_noise:
                print("Two-qubit noise on...")
                fake = simulator
                noise_model = NoiseModel.from_backend(fake)
                coupling_map = simulator.configuration().coupling_map
                qubit_map = find_qbit_map_wrt_pass_manager(fake, nqbits)
                # coupling_map = greedy_unique_pairs(coupling)[:nqbits//2]
                # qubit_map = []
                # for q0, q1 in coupling_map:
                #     qubit_map.extend([q0, q1])
                self.backend = AerSimulator(
                    noise_model=noise_model,
                    basis_gates=noise_model.basis_gates,
                    coupling_map=coupling_map,
                )

                self.comms = generate_preset_pass_manager(
                    backend=self.backend,
                    optimization_level=optimization_level,
                    initial_layout=qubit_map,
                )

                return  # skip IBM Runtime entirely

        # Only run this block if NOT in simulation mode
        try:
            self.service = QiskitRuntimeService()
        except Exception as e:
            raise ConnectionError("Failed to initialize QiskitRuntimeService.") from e

        try:
            if machine == "least_busy":
                self.backend = self.service.least_busy(
                    simulator=simulation_mode, operational=operational
                )
            elif type(machine) is str:
                all_backends = self.service.backends(
                    # simulator=False ensures we look only at real devices
                    simulator=False,
                    # You can also filter for e.g. minimum qubits,
                    # open for enrollment, etc.
                )
                available_names = [b.name for b in all_backends]
                try:
                    if machine in available_names:
                        self.backend = self.service.backend(machine)
                except Exception as e:
                    raise ConnectionError(
                        f"Machine '{machine}' not found in available backends: {available_names}"
                    ) from e
        except Exception as e:
            raise ConnectionError("Failed to select a backend.") from e

        try:
            if not two_qubit_noise:
                self.comms = generate_preset_pass_manager(
                    backend=self.backend, optimization_level=optimization_level
                )
            else:
                print("two-qubit noise on...")

                qubit_map = find_qbit_map_wrt_pass_manager(self.backend, nqbits)
                # coupling = self.backend.configuration().coupling_map
                # pairs = greedy_unique_pairs(coupling)[:nqbits//2]
                # qubit_map = []
                # for q0, q1 in pairs:
                #     qubit_map.extend([q0, q1])
                # initial_layout= [c[pqi] for c in coupling]
                print("num_qubits in layout...", len(qubit_map))
                # TODO: better error handling
                assert len(qubit_map) == nqbits
                self.comms = generate_preset_pass_manager(
                    backend=self.backend,
                    optimization_level=optimization_level,
                    initial_layout=qubit_map,
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
        save_bitstring_to: bool = False,
        noise_level: int = 0,
        noise_injection_unitary: "Operator" = None,
        two_qubit_noise: bool = False,
    ) -> np.ndarray[:]:
        """
        Perform a hybrid classical–quantum feedforward pass through the network.

        This method processes the input vector through a classical linear layer,
        then for each hidden layer builds and runs a parameterized quantum circuit
        whose rotation angles are determined by the difference between the
        pre‑activation zs and the activations computed by `self.activation_f`.
        The measurement outcomes of each quantum circuit become the inputs to
        the next layer. Finally, it applies a softmax transformation on the last linear readout
        layer to produce output probabilities.

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

        shots : int, optional
            Number of feedforward shots taken which are then measured. Default is 1.
        save_bitstring_to: bool = False,
            If provided, the bitstring results from the quantum circuit will be saved
            to this file path in JSON format.
        noise_level : int, optional
            The number of noise injections to apply per qubit in the quantum circuit.
            Default is 0 (no noise).
        noise_injection_unitary : Operator, optional
            A custom unitary operator to apply for noise injection. If provided,
            it will be applied instead of the default noise injection.
        two_qubit_noise : bool, optional
            If True, applies two-qubit noise instead of single-qubit noise.

        Returns
        -------
        np.ndarray, shape (n_output,)
            The output probability vector from the final softmax layer.
        """
        # γ = np.pi
        sampler = Sampler(mode=self.backend)
        n_blocks = len(input) if isinstance(input, list) else 1
        # print("number of blocks:", n_blocks)
        zs = input if n_blocks > 1 else input[0]
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

            # print("thetas:",θ)
            n_qbits = self.weights[i].shape[0]
            # print(f"total qubits: {n_qbits*n_blocks}")
            qr = QuantumRegister(n_qbits * n_blocks, name="qr")
            cr = ClassicalRegister(n_qbits * n_blocks, name="cr")
            qc = QuantumCircuit(qr, cr)
            for k, ϕ in enumerate(θ):
                qc.ry(theta=ϕ, qubit=k)
            if noise_level > 0:
                # print(f"noise-level = {noise_level}")
                if two_qubit_noise is False:
                    for k, _ in enumerate(θ):
                        for _ in range(0, noise_level):
                            # qc.ry(theta=γ, qubit=k)
                            # qc.ry(theta=γ, qubit=k).inverse()
                            qc.sx(qubit=k)
                            qc.sx(qubit=k).inverse()
                else:
                    for k in range(0, n_qbits - 1, 2):
                        for _ in range(0, noise_level):
                            qc.cz(k, k + 1)
                            qc.cz(k, k + 1).inverse()
            # measure all qubits
            for k, _ in enumerate(θ):
                qc.measure(qubit=k, cbit=k)

            isa_circuit = self.comms.run(qc)
            job = (
                sampler.run([isa_circuit], shots=shots)
                if quantumness > 0
                else sampler.run([isa_circuit], shots=None)
            )
            result = job.result()
            # Optionally record job IDs
            if save_bitstring_to:
                path = save_bitstring_to
                with open(path, "w") as f:
                    json.dump(result, f, cls=RuntimeEncoder)
                    f.flush()
                    os.fsync(f.fileno())

            counts = result[0].data.cr.get_counts()
            # print("counts:", counts)
            # takes bit string e.g. '10010' and makes array [1,0,0,1,0]
            zs = np.fromiter((int(b) for b in next(iter(counts))), dtype=np.float16)
            zs = zs[::-1]
            # print("bit-string -> array:", zs)
            # set 0 to 1, 1 to -1
            zs = np.where(zs == 0, 1, -1)
            zs = (
                zs
                if n_blocks == 1
                else [zs[i : i + n_qbits] for i in range(0, len(zs), n_qbits)]
            )  # print("activations:", zs)

        # apply softmax to final layer
        # print("final pre-activation:", zs)

        if n_blocks == 1:
            output = np.exp(self.weights[-1] @ zs) / sum(np.exp(self.weights[-1] @ zs))
        else:
            output = [
                np.exp(self.weights[-1] @ zs[i]) / sum(np.exp(self.weights[-1] @ zs[i]))
                for i in range(n_blocks)
            ]
        # print("output:",output)
        return output

    def feedforward_w_results(
        self: "QuantumNeuralNetwork",
        input: np.ndarray[:],
        quantumness=0.5,
        shots: int = 1,
        save_bitstring_to: bool = False,
        noise_level: int = 0,
        noise_injection_unitary: "Operator" = None,
    ) -> np.ndarray[:]:
        """
        Perform a hybrid classical–quantum feedforward pass through the network.

        This method processes the input vector through a classical linear layer,
        then for each hidden layer builds and runs a parameterized quantum circuit
        whose rotation angles are determined by the difference between the
        pre‑activation zs and the activations computed by `self.activation_f`.
        The measurement outcomes of each quantum circuit become the inputs to
        the next layer. Finally, it applies a softmax transformation on the last linear readout
        layer to produce output probabilities.

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
        activations = [zs]
        Zs = [zs]
        for i in range(0, len(self.weights) - 1):
            self.weights[i] @ zs
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

            # print("angles:", θ)
            # print("thetas:",θ)
            n_qbits = self.weights[i].shape[0]
            print(f"total qubits: {n_qbits*n_blocks}")
            qr = QuantumRegister(n_qbits * n_blocks, name="qr")
            cr = ClassicalRegister(n_qbits * n_blocks, name="cr")
            qc = QuantumCircuit(qr, cr)
            for k, ϕ in enumerate(θ):
                qc.ry(theta=ϕ, qubit=k)
                for _ in range(0, noise_level):
                    qc.noise_injection_unitary(qubit=k)
                    qc.noise_injection_unitary(qubit=k).inverse()
                qc.measure(qubit=k, cbit=k)
            isa_circuit = self.comms.run(qc)
            job = sampler.run([isa_circuit], shots=shots)
            result = job.result()
            # Optionally record job IDs
            if save_bitstring_to:
                path = save_bitstring_to
                with open(path, "w") as f:
                    json.dump(result, f, cls=RuntimeEncoder)
                    f.flush()
                    os.fsync(f.fileno())

            counts = result[0].data.cr.get_counts()
            # print("counts:", counts)
            # takes bit string e.g. '10010' and makes array [1,0,0,1,0]
            zs = np.fromiter((int(b) for b in next(iter(counts))), dtype=np.float16)
            zs = zs[::-1]
            # print("bit-string -> array:", zs)
            # set 0 to 1, 1 to -1
            zs = np.where(zs == 0, 1, -1)
            zs = (
                zs
                if n_blocks == 1
                else [zs[i : i + n_qbits] for i in range(0, len(zs), n_qbits)]
            )  # print("activations:", zs)

            activations.append(zs)

        # apply softmax to final layer
        # print("final pre-activation:", zs)
        if n_blocks == 1:
            output = np.exp(self.weights[-1] @ zs) / sum(np.exp(self.weights[-1] @ zs))
        else:
            output = [
                np.exp(self.weights[-1] @ zs[i]) / sum(np.exp(self.weights[-1] @ zs[i]))
                for i in range(n_blocks)
            ]
        # print("output:",output)
        return output, Zs, activations

    def image_block_predict(
        self,
        Xs,
        quantumness=0.5,
        n_samples=10,
        n_instances_per_block=1,
        save_bitstring_to: bool = False,
        save_results_to: str = None,
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
        assert len(Xs) % n_instances_per_block == 0
        output = []
        n = n_instances_per_block
        chunks = Xs if n == 1 else [Xs[i : i + n] for i in range(0, len(Xs), n)]
        print(f"n_chunks:{len(chunks)}")
        for chunk in chunks:
            print(f"n_chunk:{len(chunk)}")
            per_instance = []
            for i in range(n_samples):
                tmp = self.feedforward(
                    chunk, quantumness=quantumness, save_bitstring_to=save_bitstring_to
                )
                argmaxes = (
                    np.argmax(tmp)
                    if n_instances_per_block == 1
                    else np.argmax(tmp, axis=1)
                )
                print(argmaxes)
                per_instance.append(argmaxes)
                if save_results_to:
                    with open(save_results_to, "a") as f:
                        json.dump(
                            [[list(c) for c in chunk], [list(t) for t in tmp]],
                            f,
                            indent=2,
                        )
                        f.write("\n")
                        f.flush()
                        os.fsync(f.fileno())
                output.append(per_instance)

        return output

    def sample_block_predict(
        self,
        Xs,
        quantumness=0.5,
        n_samples=10,
        n_samples_per_block=1,
        save_bitstring_to: bool = False,
        save_results_to: str = None,
        noise_level: int = None,
        noise_injection_unitary: "Operator" = None,
        two_qubit_noise: bool = False,
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
        assert n_samples % n_samples_per_block == 0
        n_blocks = n_samples // n_samples_per_block
        output = []
        for x in Xs:
            per_instance = []
            for _ in range(n_blocks):
                copies_of_instance = [np.copy(x) for _ in range(n_samples_per_block)]
                tmp = self.feedforward(
                    copies_of_instance,
                    quantumness=quantumness,
                    save_bitstring_to=save_bitstring_to,
                    noise_level=noise_level,
                    noise_injection_unitary=id,
                    two_qubit_noise=two_qubit_noise,
                )
                argmaxes = (
                    np.argmax(tmp) if n_blocks == n_samples else np.argmax(tmp, axis=1)
                )

                print("prediction:", argmaxes)
                per_instance.append(argmaxes)
                if save_results_to:
                    with open(save_results_to, "a") as f:
                        json.dump([x.tolist(), [list(t) for t in tmp]], f, indent=2)
                        f.write("\n")
                        f.flush()
                        os.fsync(f.fileno())
            output.append(per_instance)
        return output

    # def sample_block_predict(
    #     self,
    #     Xs,
    #     quantumness=0.5,
    #     n_samples=10,
    #     n_samples_per_block=1,
    #     save_bitstring_to: bool = False,
    #     save_results_to: str = None,
    #     noise_level: int = None,
    # ) -> list[np.ndarray[:]]:
    #     """
    #     Generate multiple stochastic predictions for each input using the hybrid
    #     quantum–classical network.

    #     Parameters
    #     ----------
    #     Xs : Iterable[np.ndarray]
    #         A collection of input vectors to predict on. Each element should be a 1D numpy array
    #         matching the network's input dimension.
    #     quantumness : float, optional
    #         Interpolation parameter between purely classical (0.0) and fully quantum (1.0
    #         hidden-layer
    #         activations. Defaults to 0.5.
    #     n_samples : int, optional
    #         Number of stochastic forward passes to perform per input. Defaults to 10.

    #     Returns
    #     -------
    #     list[np.ndarray]
    #         A flat list of output probability vectors from all runs, with length equal to
    #         `len(Xs) * n_samples`. Each element is the softmax output of one forward pass.
    #     """
    #     assert n_samples % n_samples_per_block == 0
    #     n_blocks = n_samples // n_samples_per_block
    #     output = []
    #     for x in Xs:
    #         per_instance = []
    #         for _ in range(n_blocks):
    #             copies_of_instance = [np.copy(x) for _ in range(n_samples_per_block)]
    #             tmp = self.feedforward(
    #                 copies_of_instance,
    #                 quantumness=quantumness,
    #                 save_bitstring_to=save_bitstring_to,
    #                 noise_level=noise_level,
    #                 noise_injection_unitary=id
    #             )
    #             argmaxes = np.argmax(tmp) if n_samples == 1 else np.argmax(tmp, axis=1)
    #             per_instance.append(argmaxes)
    #             if save_results_to:
    #                 with open(save_results_to, "a") as f:
    #                     json.dump([x.tolist(), [list(t) for t in tmp]], f, indent=2)
    #                     f.write("\n")
    #                     f.flush()
    #                     os.fsync(f.fileno())
    #         output.append(per_instance)
    #     return output

    def train(
        self,
        input,
        output,
        quantumness: (int | float) = 0.5,
        shots: (int | float) = 1,
        noise_level: (int | float) = 0,
        momentum: (int | float) = None,
        minibatch: bool = True,
        minibatch_pool: (int | float) = 10,
        iterations: (int | float) = 100,
        η: (int | float) = 1e-6,
        update_rules: callable = None,
    ) -> None:
        """
        Trains the neural network using gradient descent.

        Parameters:
            minibatch (bool): Whether to use mini-batch gradient descent (default: True).
            minibatch_pool (int | float): Size of the mini-batch for training (default: 10).
            iterations (int | float): Number of training iterations (default: 100).
            η (int | float): Learning rate for gradient descent (default: 1e-6).

        Returns:
            NeuralNetwork: The trained neural network object.

        Description:
            - Implements forward propagation for each input to compute activations.
            - Performs backpropagation to compute gradients for weights and biases.
            - Updates weights and biases using gradient descent.
            - Supports mini-batch gradient descent if `minibatch` is set to True.

        Example:
            nn.train(minibatch=True, minibatch_pool=32, iterations=1000, η=0.01)
        """

        for _ in range(iterations):
            if minibatch:
                indexes = np.random.choice(input.shape[0], size=minibatch_pool)
                X, Y = input[indexes], output[indexes]
            else:
                X, Y = input, output

            w_grads = [np.zeros(matrix.shape) for matrix in self.weights]

            b_grads = [np.zeros(vector.shape) for vector in self.bias]

            if momentum is not None:
                v = [np.zeros(w.shape) for w in self.weights]

            # iterate for each set of x and y
            # find zs and as (pre-act and activation)
            for x, y in zip(X, Y):
                # print("x id:",id(x))

                # def "quantum" feedforward
                output, zs, activations = self.feedforward_w_results(
                    x, quantumness=quantumness, shots=shots, noise_level=noise_level
                )
                z_output = zs[-1]
                a_output = activations[-1]

                output_error = self.cost_grad(a_output, y) * self.activation_df(
                    z_output
                )
                errors = [output_error]
                for l in range(self.hidden_layer_n, 0, -1):
                    error = (
                        self.weights[l].T @ errors[-1] * self.activation_df(zs[l - 1])
                    )
                    errors.append(error)

                errors.reverse()
                # compute sum of error
                for l in range(0, self.hidden_layer_n + 1, 1):
                    # w_grads[l] += errors[l]@activations[l].T
                    w_grads[l] += np.outer(
                        errors[l], activations[l - 1] if l > 0 else x
                    )
                    # print(w_grads)
                    b_grads[l] += errors[l]
                    # print(b_grads)

            # gradient descent
            if momentum is None:
                for l in range(0, self.hidden_layer_n + 1):
                    self.weights[l] -= η / minibatch_pool * w_grads[l]
                    self.bias[l] -= η / minibatch_pool * b_grads[l]
            else:
                γ = momentum
                for l in range(0, self.hidden_layer_n + 1):
                    v[l] = γ * v[l] + η / minibatch_pool * w_grads[l]
                    self.weights[l] -= v[l]
                    self.bias[l] -= η / minibatch_pool * b_grads[l]
