import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService


def test_simulator():
    # Create a new circuit with two qubits
    qc = QuantumCircuit(2)

    # Add a Hadamard gate to qubit 0
    qc.h(0)

    # Perform a controlled-X gate on qubit 1, controlled by qubit 0
    qc.cx(0, 1)

    service = QiskitRuntimeService(channel="local")

    backend = service.least_busy(simulator=True, operational=True)

    # Convert to an ISA circuit and layout-mapped observables.
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(qc)

    isa_circuit.draw("mpl", idle_wires=False)
