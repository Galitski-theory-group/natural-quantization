import datetime
import json
import os
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeEncoder
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeKyiv


# TODO: enforce one classical data type
# TODO: make sure we select qubits with furthest connectivity
# TODO : pytorch mnist dataset
# TODO : add number of blocks per instance

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
        simulator=FakeKyiv(),
        operational=True,
        optimization_level=1,
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
        if simulation_mode:
            self.backend = simulator
            self.comms = generate_preset_pass_manager(
                backend=self.backend, optimization_level=optimization_level
            )
            return  # skip IBM Runtime entirely

        # Only run this block if NOT in simulation mode
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            self.service = QiskitRuntimeService()
        except Exception as e:
            raise ConnectionError("Failed to initialize QiskitRuntimeService.") from e

        try:
            self.backend = self.service.least_busy(
                simulator=simulation_mode, operational=operational
            )
        except Exception as e:
            raise ConnectionError("Failed to select a backend.") from e

        try:
            self.comms = generate_preset_pass_manager(
                backend=self.backend, optimization_level=optimization_level
            )
        except Exception as e:
            raise RuntimeError("Failed to generate the communications pass manager.") from e

    def feedforward(
        self: "QuantumNeuralNetwork",
        input: np.ndarray[:],
        quantumness=0.5,
        shots: int = 1,
        save_bitstring_to: bool = False,
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

            # print("angles:", θ)
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

    def predict(
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
        assert len(Xs)%n_instances_per_block == 0
        try:
            output = []
            n = n_instances_per_block
            chunks = Xs if n == 1 else [Xs[i : i + n] for i in range(0, len(Xs), n)]
            for chunk in chunks:
                per_instance = []
                for i in range(n_samples):
                    tmp = self.feedforward(
                        chunk, quantumness=quantumness, save_bitstring_to=save_bitstring_to
                    )
                    argmaxes = (
                        np.argmax(tmp) if n_instances_per_block == 1 else np.argmax(tmp, axis=1)
                    )
                    print(argmaxes)
                    per_instance.append(argmaxes)
                    if f:
                        with open(save_results_to, "a") as f:
                            json.dump([list(chunk), list(tmp)], f, indent=2)
                            f.write("\n")
                            f.flush()
                            os.fsync(f.fileno())
                output.append(per_instance)

        finally:
            if f:
                f.close()

        return output
