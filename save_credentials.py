# script_name.py
import sys

from qiskit_ibm_runtime import QiskitRuntimeService

if len(sys.argv) > 1:
    token = sys.argv[1]
    QiskitRuntimeService.save_account(
        token=token, channel="ibm_quantum", overwrite=True
    )
else:
    print("Please provide token")
