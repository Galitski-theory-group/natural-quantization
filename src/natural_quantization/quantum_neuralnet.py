import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler

# TODO: enforce one classical data type
# TODO: make sure we select qubits with furthest connectivity


class QuantumNeuralNetwork:

    """ """

    def __init__(
        self, layer_sizes: list, activation_f: callable
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
        self: "QuantumNeuralNetwork", input: np.ndarray[:]
    ) -> np.ndarray[:]:
        zs = input
        for i in range(len(self.weights) - 1):
            θ = np.pi / 2 * (1 - self.activation_f(self.weights[i] @ zs))
            n_qbits = self.weights[i].shape[0]
            qr = QuantumRegister(n_qbits, name="qr")
            cr = ClassicalRegister(n_qbits, name="cr")
            qc = QuantumCircuit(qr, cr)
            for i, Θ in enumerate(θ):
                qc.ry(theta=Θ, qubit=i)
                qc.measure(qubit=i, cbit=i)
            isa_circuit = self.pm.run(qc)
            sampler = Sampler(mode=self.backend)
            job = sampler.run([isa_circuit], shots=1)
            result = job.result()
            counts = result[0].data.cr.get_counts()
            zs = np.fromiter((int(b) for b in next(iter(counts))), dtype=np.float16)

        output = np.exp(self.weights[-1] @ zs) / sum(np.exp(self.weights[-1] @ zs))
        return output

    # def predict(self,
    #             Xs,
    #             n_samples)->list[np.ndarray[:]]:

    #     output = []
    #     for x in Xs:
    #         for _ in num_samples:
    #             tmp = self.feedforward(x)

    #         output.append(tmp)

    #     return tmp
