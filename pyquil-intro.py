from pyquil import Program, get_qc
from pyquil.gates import H, CNOT
from pyquil.api import local_forest_runtime

# construct a Bell State program
p = Program(H(0), CNOT(0, 1))

with local_forest_runtime():
    # run the program on a QVM
    qc = get_qc('9q-square-qvm')
    result = qc.run_and_measure(p, trials=10)
    
print(result[0])
print(result[1])