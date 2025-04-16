import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler

# TODO: enforce one classical data type
# TODO: make sure we select qubits with furthest connectivity


def htanh(x: np.ndarray[:], a: (int | float)) -> np.ndarray[:]:
    """
    Compute a hard tanh activation on the input x.

    This function applies a piecewise transformation to x:
      - When |x| ≤ a, it returns x/a.
      - When |x| > a, it returns 1 if x is positive, and -1 otherwise.
      - For a = 0, the function directly returns the sign of x.

    Parameters:
      x : array-like or scalar
          The input value(s) to be transformed.
      a : float
          The threshold parameter (with |a| ≤ 1) that defines the linear region.

    Returns:
      Transformed value(s) following the hard tanh definition applied element-wise.

    Raises:
      AssertionError: If the absolute value of a is greater than 1.
    """

    assert abs(a) <= 1

    if a == 0:
        return np.where(x > 0, 1, -1)
    else:
        return np.where(abs(x) <= a, x / a, np.where(x > 0, 1, -1))


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
        self, layer_sizes: list, activation_f: callable = htanh
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

        self.activation_f = activation_f
        self.service = None
        self.backend = None
        self.comms = None

    def establish_communication_with_ibm(
        self, simulation_mode=True, operational=True, optimization_level=3
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
            The optimization level (0–3) for the communications pass manager. Higher values apply
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
            # Choose a backend with session and measurement support
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

        # take classical steps
        zs = self.weights[0] @ input
        for i in range(1, len(self.weights) - 1):
            θ = (
                np.pi / 2 * (1 - self.activation_f(self.weights[i] @ zs, quantumness))
                if i > 1
                else np.pi / 2 * (1 - self.activation_f(zs, quantumness))
            )
            n_qbits = self.weights[i].shape[0]
            qr = QuantumRegister(n_qbits, name="qr")
            cr = ClassicalRegister(n_qbits, name="cr")
            qc = QuantumCircuit(qr, cr)
            for i, Θ in enumerate(θ):
                qc.ry(theta=Θ, qubit=i)
                qc.measure(qubit=i, cbit=i)
            isa_circuit = self.comms.run(qc)
            sampler = Sampler(mode=self.backend)
            job = sampler.run([isa_circuit], shots=1)
            result = job.result()
            counts = result[0].data.cr.get_counts()
            zs = np.fromiter((int(b) for b in next(iter(counts))), dtype=np.float16)

        # apply softmax to final layer
        output = np.exp(self.weights[-1] @ zs) / sum(np.exp(self.weights[-1] @ zs))
        return output

    def predict(self, Xs, quantumness=0.5, n_samples=10) -> list[np.ndarray[:]]:
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
            for i in range(n_samples):
                tmp = self.feedforward(x, quantumness)
                output.append(tmp)
                print("shot #{i}", np.argmax(tmp))
        return output
