import numba
import numpy as np
from sympy import lambdify
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter

@numba.njit
def kron(a, b):
    return np.kron(a, b)

@numba.njit
def qatgOnestateFidelity(faultyQuantumState, faultfreeQuantumState):
    fid = np.abs(faultfreeQuantumState.conj().dot(faultyQuantumState)) ** 2
    fid = float(np.real(fid))
    return fid

class U2GateSetsTranspiler():

    def __init__(self, basisGateSetString):
        q = QuantumCircuit(1)
        qiskitParameterTheta = Parameter('theta')
        qiskitParameterPhi = Parameter('phi')
        qiskitParameterLambda = Parameter('lam')
        q.u(qiskitParameterTheta, qiskitParameterPhi, qiskitParameterLambda, 0)
        try:
            effectiveUGateCircuit = transpile(q, basis_gates = basisGateSetString, optimization_level = 3)
        except Exception as e:
            raise e
        self.ops = []
        params = []
        for instruction in effectiveUGateCircuit.data:
            self.ops.append(instruction.operation)
            params.append(
                [param if isinstance(param, float) else param.sympify() for param in instruction.operation.params]
            )
        self.make_params = lambdify([
            qiskitParameterTheta.sympify(), qiskitParameterPhi.sympify(), qiskitParameterLambda.sympify()
        ], params)

    def transpile(self, UParameters):
        theta, phi, lam = UParameters
        ops = []
        for op, params in zip(self.ops, self.make_params(theta, phi, lam)):
            op = op.copy()
            op.params = params
            ops.append(op)
        return ops
