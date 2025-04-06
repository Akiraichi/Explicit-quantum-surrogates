from qulacs import Observable, ParametricQuantumCircuit

from vqe_dataset_loader.ansatz_name import AnsatzName
from vqe_dataset_loader.hamiltonian_define import get_defined_hamiltonian


def chain_map(n_qubits):
    mapping = []
    for index in range(0, n_qubits, 2):  # n_qubits=8, [0,2,4,6]
        mapping.append([index, index + 1])
    for index in range(1, n_qubits - 1, 2):  # n_qubits=8, [1,3,5]
        mapping.append([index, index + 1])
    return mapping


def stair_map(n_qubits):
    mapping = []
    for index in range(n_qubits - 1):
        mapping.append([index, index + 1])
    return mapping


def complete_map(n_qubits):
    mapping = []
    for i in range(n_qubits - 1):
        for k in range(i + 1, n_qubits):
            mapping.append([i, k])

    return mapping


def ladder_map(n_qubits):
    mapping = []
    for index in range(0, n_qubits, 2):  # n_qubits=8, [0,2,4,6]
        mapping.append([index, index + 1])
    for index in range(0, n_qubits - 2, 2):  # n_qubits=8, [0,2, 4]
        mapping.append([index, index + 2])
    for index in range(1, n_qubits - 2, 2):  # n_qubits=8, [1,3,5]
        mapping.append([index, index + 2])

    return mapping


def cross_map(n_qubits):
    mapping = []
    for index in range(0, n_qubits - 2, 2):  # n_qubits=8, [0,2,4]
        mapping.append([index, index + 3])
        mapping.append([index + 1, index + 2])
    return mapping


def _U2(ansatz, top_index, bottom_index):
    for _ in range(2):
        ansatz.add_parametric_RY_gate(index=top_index, angle=0)
        ansatz.add_parametric_RY_gate(index=bottom_index, angle=0)
        ansatz.add_CNOT_gate(control=top_index, target=bottom_index)


def __bb_ansatz_core(reps, n_qubits, mapping):
    ansatz = ParametricQuantumCircuit(n_qubits)
    for _ in range(reps):
        for couple in mapping:
            _U2(ansatz=ansatz, top_index=couple[0], bottom_index=couple[1])

    for index_of_qubit in range(n_qubits):
        ansatz.add_parametric_RY_gate(index=index_of_qubit, angle=0)

    return ansatz


def bb_chain_ansatz(reps, n_qubits):
    mapping = chain_map(n_qubits=n_qubits)
    return __bb_ansatz_core(reps=reps, n_qubits=n_qubits, mapping=mapping)


def bb_complete_ansatz(reps, n_qubits):
    mapping = complete_map(n_qubits=n_qubits)
    return __bb_ansatz_core(reps=reps, n_qubits=n_qubits, mapping=mapping)


def bb_ladder_ansatz(reps, n_qubits):
    mapping = ladder_map(n_qubits=n_qubits)
    return __bb_ansatz_core(reps=reps, n_qubits=n_qubits, mapping=mapping)


def bb_cross_ladder_ansatz(reps, n_qubits):
    ladder_mapping = ladder_map(n_qubits=n_qubits)
    cross_mapping = cross_map(n_qubits=n_qubits)
    return __bb_ansatz_core(
        reps=reps, n_qubits=n_qubits, mapping=ladder_mapping + cross_mapping
    )


def bb_stair_ansatz(reps, n_qubits):
    mapping = stair_map(n_qubits=n_qubits)
    return __bb_ansatz_core(reps=reps, n_qubits=n_qubits, mapping=mapping)


def __he_ansatz_core(reps, n_qubits, mapping):
    ansatz = ParametricQuantumCircuit(n_qubits)

    for _ in range(reps):
        for i in range(n_qubits):
            ansatz.add_parametric_RY_gate(index=i, angle=0)
            ansatz.add_parametric_RZ_gate(index=i, angle=0)
        for couple in mapping:
            ansatz.add_CZ_gate(couple[0], couple[1])

    for i in range(n_qubits):
        ansatz.add_parametric_RY_gate(index=i, angle=0)
        ansatz.add_parametric_RZ_gate(index=i, angle=0)

    return ansatz


def he_chain_ansatz(reps, n_qubits):
    mapping = chain_map(n_qubits=n_qubits)
    return __he_ansatz_core(reps=reps, n_qubits=n_qubits, mapping=mapping)


def he_stair_ansatz(reps, n_qubits):
    mapping = stair_map(n_qubits=n_qubits)
    return __he_ansatz_core(reps=reps, n_qubits=n_qubits, mapping=mapping)


def he_complete_ansatz(reps, n_qubits):
    mapping = complete_map(n_qubits=n_qubits)
    return __he_ansatz_core(reps=reps, n_qubits=n_qubits, mapping=mapping)


def he_ladder_ansatz(reps, n_qubits):
    mapping = ladder_map(n_qubits=n_qubits)
    return __he_ansatz_core(reps=reps, n_qubits=n_qubits, mapping=mapping)


def he_cross_ladder_ansatz(reps, n_qubits):
    ladder_mapping = ladder_map(n_qubits=n_qubits)
    cross_mapping = cross_map(n_qubits=n_qubits)
    return __he_ansatz_core(
        reps=reps, n_qubits=n_qubits, mapping=ladder_mapping + cross_mapping
    )


def split_hamiltonian(hamiltonian):
    pauli_ids_list = []
    targets_list = []
    coef_list = []
    for index_of_term in range(hamiltonian.get_term_count()):
        pauli = hamiltonian.get_term(index_of_term)
        coef = abs(pauli.get_coef())
        indexes = pauli.get_index_list()
        pauli_ids = pauli.get_pauli_id_list()

        pauli_ids_list.append(pauli_ids)
        targets_list.append(indexes)
        coef_list.append(coef)
    return targets_list, pauli_ids_list, coef_list


def hamiltonian_ansatz(depth, hamiltonian, setting):
    n_qubits = hamiltonian.get_qubit_count()
    ansatz = ParametricQuantumCircuit(n_qubits)

    targets_list, pauli_ids_list, coefs_list = split_hamiltonian(hamiltonian)

    for _ in range(depth):
        for target, pauli_index, coef in zip(targets_list, pauli_ids_list, coefs_list):
            ansatz.add_parametric_multi_Pauli_rotation_gate(target, pauli_index, 0)
        for i in range(n_qubits):
            # ラベルによっては、RXやRZを追加しても無意味な場合があるので、場合わけする
            if setting["label"] != 0:
                ansatz.add_parametric_RX_gate(index=i, angle=0)
            if setting["label"] != 1:
                ansatz.add_parametric_RZ_gate(index=i, angle=0)

    return ansatz


def get_ansatz(setting: dict, hamiltonian: Observable | None = None):
    """ansatzを取得"""
    if setting["ansatz_name"] == AnsatzName.Hamiltonian:
        if hamiltonian is None:
            raise RuntimeError(
                "Hamiltonian ansatzを使用するのであれば、ハミルトニアンを指定してください"
            )

        if hamiltonian:
            ansatz = hamiltonian_ansatz(
                depth=setting["reps"], hamiltonian=hamiltonian, setting=setting
            )
        else:
            hamiltonian = get_defined_hamiltonian(setting=setting)
            ansatz = hamiltonian_ansatz(
                depth=setting["reps"], hamiltonian=hamiltonian, setting=setting
            )
            # assert False, "Hamiltonian ansatzを使用するのであれば、ハミルトニアンを指定してください"

    elif setting["ansatz_name"] == AnsatzName.HardwareEfficient:
        ansatz = he_chain_ansatz(reps=setting["reps"], n_qubits=setting["n_qubits"])
    elif setting["ansatz_name"] == AnsatzName.HardwareEfficientFull:
        ansatz = he_complete_ansatz(reps=setting["reps"], n_qubits=setting["n_qubits"])
    elif setting["ansatz_name"] == AnsatzName.HardwareEfficientLadder:
        ansatz = he_ladder_ansatz(reps=setting["reps"], n_qubits=setting["n_qubits"])
    elif setting["ansatz_name"] == AnsatzName.HardwareEfficientCrossLadder:
        ansatz = he_cross_ladder_ansatz(
            reps=setting["reps"], n_qubits=setting["n_qubits"]
        )

    elif setting["ansatz_name"] == AnsatzName.TwoLocal:
        ansatz = bb_chain_ansatz(reps=setting["reps"], n_qubits=setting["n_qubits"])
    elif setting["ansatz_name"] == AnsatzName.TwoLocalStair:
        ansatz = bb_stair_ansatz(reps=setting["reps"], n_qubits=setting["n_qubits"])
    elif setting["ansatz_name"] == AnsatzName.TwoLocalFull:
        ansatz = bb_complete_ansatz(reps=setting["reps"], n_qubits=setting["n_qubits"])
    elif setting["ansatz_name"] == AnsatzName.TwoLocalLadder:
        ansatz = bb_ladder_ansatz(reps=setting["reps"], n_qubits=setting["n_qubits"])
    elif setting["ansatz_name"] == AnsatzName.TwoLocalCrossLadder:
        ansatz = bb_cross_ladder_ansatz(
            reps=setting["reps"], n_qubits=setting["n_qubits"]
        )
    else:
        assert False, "ansatz_nameをミスっている"
    return ansatz


def get_parameter_set_ansatz(__ansatz: ParametricQuantumCircuit, _param):
    """パラメータをansatzに設定する"""
    ansatz = __ansatz.copy()
    assert ansatz.get_parameter_count() == len(
        _param
    ), "ansatzのパラメータ数と与えられたパラメータの数が一致していません"
    for i in range(ansatz.get_parameter_count()):
        ansatz.set_parameter(i, _param[i])
    return ansatz


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from qulacsvis import circuit_drawer  # type: ignore

    ansatz = he_ladder_ansatz(reps=1, n_qubits=8)
    circuit_drawer(ansatz, "mpl")
    plt.show()
