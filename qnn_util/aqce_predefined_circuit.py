import random
from typing import List

import numpy as np
from qulacs.gate import DenseMatrix

from aqce_util.AQCEDictType import MatrixCDict
from qnn_util.LearningCircuitClass import MyLearningCircuit


def add_two_qubit_parametric_gate(
    circuit: MyLearningCircuit, upper_index: int, lower_index, params: list
) -> MyLearningCircuit:
    """
    Add a two-qubit parametric gate to the quantum circuit.

    This function adds a sequence of parametric gates that implement a two-qubit interaction:
    1. Single-qubit rotations on both qubits
    2. Two-qubit entangling operations (RXX, RYY, RZZ)
    3. Final single-qubit rotations on both qubits

    Args:
        circuit: The quantum circuit to add gates to
        upper_index: Index of the first qubit
        lower_index: Index of the second qubit
        params: List of 15 parameters for the gates

    Returns:
        The updated quantum circuit
    """
    # Apply the first set of single-qubit rotations
    circuit.add_parametric_RZ_gate(index=upper_index, parameter=params[0])
    circuit.add_parametric_RY_gate(index=upper_index, parameter=params[1])
    circuit.add_parametric_RZ_gate(index=upper_index, parameter=params[2])
    circuit.add_parametric_RZ_gate(index=lower_index, parameter=params[3])
    circuit.add_parametric_RY_gate(index=lower_index, parameter=params[4])
    circuit.add_parametric_RZ_gate(index=lower_index, parameter=params[5])

    # RXX, TODO: I believe X=1, Y=2, Z=3, I=0. Need to verify.
    circuit._add_multi_qubit_parametric_R_gate_inner(
        target=[upper_index, lower_index],
        pauli_id=[1, 1],
        initial_angle=params[6],
        share_with=None,
        share_with_coef=None,
    )

    # circuit.add_parametric_multi_Pauli_rotation_gate(target=[upper_index, lower_index], pauli_id=[1, 1],
    #                                                  initial_angle=params[6])
    # RYY
    circuit._add_multi_qubit_parametric_R_gate_inner(
        target=[upper_index, lower_index],
        pauli_id=[2, 2],
        initial_angle=params[7],
        share_with=None,
        share_with_coef=None,
    )
    # circuit.add_parametric_multi_Pauli_rotation_gate(target=[upper_index, lower_index], pauli_id=[2, 2],
    #                                                  initial_angle=params[7])

    # RZZ
    circuit._add_multi_qubit_parametric_R_gate_inner(
        target=[upper_index, lower_index],
        pauli_id=[3, 3],
        initial_angle=params[8],
        share_with=None,
        share_with_coef=None,
    )
    # circuit.add_parametric_multi_Pauli_rotation_gate(target=[upper_index, lower_index], pauli_id=[3, 3],
    #                                                  initial_angle=params[8])
    # Apply the third set of single-qubit rotations
    circuit.add_parametric_RZ_gate(index=upper_index, parameter=params[9])
    circuit.add_parametric_RY_gate(index=upper_index, parameter=params[10])
    circuit.add_parametric_RZ_gate(index=upper_index, parameter=params[11])
    circuit.add_parametric_RZ_gate(index=lower_index, parameter=params[12])
    circuit.add_parametric_RY_gate(index=lower_index, parameter=params[13])
    circuit.add_parametric_RZ_gate(index=lower_index, parameter=params[14])

    return circuit


def add_RX_gate_all_qubits(
    circuit: MyLearningCircuit, n_qubits: int
) -> MyLearningCircuit:
    """
    Add RX gates to all qubits in the circuit.

    This function adds a sequence of RZ-RX-RZ gates to each qubit in the circuit,
    which is equivalent to adding an RX rotation with additional phase freedom.

    Args:
        circuit: The quantum circuit to add gates to
        n_qubits: Number of qubits in the circuit

    Returns:
        The updated quantum circuit
    """
    for i in range(n_qubits):
        circuit.add_parametric_RZ_gate(index=i, parameter=0.0)
        circuit.add_parametric_RX_gate(index=i, parameter=0.0)
        circuit.add_parametric_RZ_gate(index=i, parameter=0.0)
    return circuit


def aqce_circuit_random_parameter(
    n_qubits: int, pre_trained_gate_list: list[MatrixCDict]
) -> MyLearningCircuit:
    """
    Create a quantum circuit with random parameters based on the structure of a pre-trained circuit.

    This function:
    1. Gets the quantum circuit of an explicit model converted from an implicit model.
    2. Reconstructs the quantum circuit U (note: not U^dagger). Adds RX gates at the beginning to evaluate |k⟩.
    3. For random circuits: When rearranging, inserts parametric circuits instead of the original quantum circuit.
       Initial values are set randomly.
    4. Returns the circuit.

    Args:
        n_qubits: Number of qubits in the circuit
        pre_trained_gate_list: List of pre-trained gates with their indices

    Returns:
        A quantum circuit with random parameters
    """
    circuit = MyLearningCircuit(n_qubit=n_qubits)
    circuit = add_RX_gate_all_qubits(circuit=circuit, n_qubits=n_qubits)
    rng = np.random.default_rng()

    for data in pre_trained_gate_list:
        i = data["i"]
        j = data["j"]
        params = list(rng.random(15) * 2 * np.pi)  # 0~2piの乱数を15個生成
        circuit = add_two_qubit_parametric_gate(
            circuit=circuit, upper_index=i, lower_index=j, params=params
        )

    return circuit


def aqce_circuit_pretrained(
    n_qubits: int, pre_trained_gate_list: list[MatrixCDict]
) -> MyLearningCircuit:
    """
    Create a quantum circuit using pre-trained gates followed by parametric gates.

    This function:
    1. Gets the quantum circuit of an explicit model converted from an implicit model.
    2. Reconstructs the quantum circuit (note: not U^dagger). Adds RX gates at the beginning to evaluate |k⟩.
    3. Rearranges and inserts parametric circuits next to the pre-trained gates. Initial values are set to 0.
    4. Returns the circuit.

    Args:
        n_qubits: Number of qubits in the circuit
        pre_trained_gate_list: List of pre-trained gates with their indices

    Returns:
        A quantum circuit with pre-trained gates and parametric gates
    """
    circuit = MyLearningCircuit(n_qubit=n_qubits)
    circuit = add_RX_gate_all_qubits(circuit=circuit, n_qubits=n_qubits)

    for data in pre_trained_gate_list:
        i = data["i"]
        j = data["j"]
        U = data["U"]
        # AQCEで最適化したゲートを追加
        C_quantum_gate = DenseMatrix([i, j], U)
        circuit.add_gate(C_quantum_gate)
        # その隣に初期値を0に設定したパラメータ付き量子回路を追加
        params = [0.0] * 15
        circuit = add_two_qubit_parametric_gate(
            circuit=circuit, upper_index=i, lower_index=j, params=params
        )

    return circuit


def aqce_circuit_native(
    n_qubits: int, pre_trained_gate_list: list[MatrixCDict]
) -> MyLearningCircuit:
    """
    Create a quantum circuit using only pre-trained gates without additional parametric gates.

    This function creates a circuit by directly adding the pre-trained gates without
    any additional RX gates or parametric gates.

    Args:
        n_qubits: Number of qubits in the circuit
        pre_trained_gate_list: List of pre-trained gates with their indices

    Returns:
        A quantum circuit with only pre-trained gates
    """
    circuit = MyLearningCircuit(n_qubit=n_qubits)

    for data in pre_trained_gate_list:
        i = data["i"]
        j = data["j"]
        U = data["U"]
        # AQCEで最適化したゲートを追加
        C_quantum_gate = DenseMatrix([i, j], U)
        circuit.add_gate(C_quantum_gate)

    return circuit


def generate_pairs(n_qubit: int, n: int) -> list[tuple[int, int]]:
    """
    Generate a list of random qubit index pairs.

    This function generates n random pairs of qubit indices (i,j) where i < j.
    It ensures that consecutive pairs are different.

    Args:
        n_qubit: Number of qubits in the circuit
        n: Number of pairs to generate

    Returns:
        List of tuples, each containing two qubit indices
    """
    pairs: list[tuple[int, int]] = []
    for _ in range(n):
        while True:
            A = random.randint(0, n_qubit - 1 - 1)
            B = random.randint(A + 1, n_qubit - 1)
            # 前回追加したペアと同じでなければ追加
            if not pairs or pairs[-1] != (A, B):
                pairs.append((A, B))
                break
    return pairs


def aqce_structure_random_circuit(
    n_qubits: int, pre_trained_gate_list: list[MatrixCDict]
) -> MyLearningCircuit:
    """
    Create a quantum circuit with random structure and parameters.

    Unlike aqce_circuit_random_parameter which preserves the structure of the pre-trained circuit,
    this function randomizes both the structure (qubit indices) and the parameters.

    Args:
        n_qubits: Number of qubits in the circuit
        pre_trained_gate_list: List of pre-trained gates (used only to determine the number of gates)

    Returns:
        A quantum circuit with random structure and parameters
    """
    circuit = MyLearningCircuit(n_qubit=n_qubits)
    circuit = add_RX_gate_all_qubits(circuit=circuit, n_qubits=n_qubits)
    rng = np.random.default_rng()

    pairs = generate_pairs(n_qubit=n_qubits, n=len(pre_trained_gate_list))

    for pair in pairs:
        print(pair)
        i = pair[0]
        j = pair[1]
        params = list(rng.random(15) * 2 * np.pi)  # 0~2piの乱数を15個生成
        circuit = add_two_qubit_parametric_gate(
            circuit=circuit, upper_index=i, lower_index=j, params=params
        )

    return circuit
