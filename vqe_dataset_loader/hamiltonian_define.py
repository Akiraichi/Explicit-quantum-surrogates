import numpy as np
import openfermion as of
from openfermion import jordan_wigner
from qulacs import Observable
from qulacs.observable import create_observable_from_openfermion_text

from vqe_dataset_loader.hamiltonian_name import HamiltonianName

def get_defined_hamiltonian(setting):
    n_qubits = setting["n_qubits"]
    assert n_qubits % 4 == 0, "n_qubitは4の倍数である必要があります"
    _lattice = setting["lattice"]

    operator = Observable(n_qubits)

    if setting["hamiltonian_name"] == HamiltonianName.Ising_1d:
        """一次元格子の横磁場イジングモデル"""
        for k in range(len(_lattice)):
            operator.add_operator(1, f"Z {_lattice[k][0]}Z {_lattice[k][1]}")
        for i in range(n_qubits):
            operator.add_operator(setting["h"], f"X {i}")
    elif setting["hamiltonian_name"] == HamiltonianName.Heisenberg_1d:
        """一次元格子のハイゼンベルグモデル"""
        for k in range(len(_lattice)):
            operator.add_operator(1, f"X {_lattice[k][0]}X {_lattice[k][1]}")
            operator.add_operator(1, f"Y {_lattice[k][0]}Y {_lattice[k][1]}")
            operator.add_operator(1, f"Z {_lattice[k][0]}Z {_lattice[k][1]}")
        for i in range(n_qubits):
            operator.add_operator(setting["h"], f"Z {i}")
    elif setting["hamiltonian_name"] == HamiltonianName.SSH_1d:
        """一次元格子のSSHモデル"""
        for k in range(len(_lattice)):  # あとで確認する
            coef = (1 + ((-1) ** _lattice[k][0]) * setting["delta"])
            operator.add_operator(coef, f"X {_lattice[k][0]}X {_lattice[k][1]}")
            operator.add_operator(coef, f"Y {_lattice[k][0]}Y {_lattice[k][1]}")
            operator.add_operator(coef, f"Z {_lattice[k][0]}Z {_lattice[k][1]}")
    elif setting["hamiltonian_name"] == HamiltonianName.Kitaex_chain:
        """一次元格子のKitaexモデル"""
        Jx = np.cos(setting["theta"])
        Jy = np.sin(setting["theta"])
        for k in range(len(_lattice)):
            operator.add_operator(Jx, f"X {_lattice[k][0]}X {_lattice[k][1]}")
            operator.add_operator(Jy, f"Y {_lattice[k][0]}Y {_lattice[k][1]}")
    elif setting["hamiltonian_name"] == HamiltonianName.J1J2model:
        """一次元格子のJ1-J2モデル"""
        J2 = setting["J2"]
        for k in range(len(_lattice)):
            operator.add_operator(1, f"X {_lattice[k][0]}X {_lattice[k][1]}")
            operator.add_operator(1, f"Y {_lattice[k][0]}Y {_lattice[k][1]}")
            operator.add_operator(1, f"Z {_lattice[k][0]}Z {_lattice[k][1]}")
        for k in range(len(_lattice) - 1):
            operator.add_operator(J2, f"X {_lattice[k][0]}X {_lattice[k][1] + 1}")
            operator.add_operator(J2, f"Y {_lattice[k][0]}Y {_lattice[k][1] + 1}")
            operator.add_operator(J2, f"Z {_lattice[k][0]}Z {_lattice[k][1] + 1}")
    elif setting["hamiltonian_name"] == HamiltonianName.Hubbard_1d:
        """周期的境界条件なし, harf-filling, -Jと定義している点に注意"""
        J = -1.0
        hubbard = of.fermi_hubbard(x_dimension=n_qubits // 2, y_dimension=1, tunneling=-J, coulomb=setting["U"],
                                   periodic=False, spinless=False, particle_hole_symmetry=True)
        jw_hamiltonian = jordan_wigner(hubbard)
        jw_hamiltonian.compress()
        operator = create_observable_from_openfermion_text(str(jw_hamiltonian))
    elif setting["hamiltonian_name"] == HamiltonianName.Hubbard_2d:
        """周期的境界条件なし, harf-filling, -Jと定義している点に注意"""
        J = -1.0
        assert n_qubits == 8 or n_qubits == 12 or n_qubits == 16 or n_qubits == 20 or n_qubits == 24 or n_qubits == 28, "2Dハバードモデルを定義するには、qubit数の割り当てがよくないと思います"
        hubbard = of.fermi_hubbard(x_dimension=n_qubits // 4, y_dimension=2, tunneling=-J, coulomb=setting["U"],
                                   periodic=False, spinless=False, particle_hole_symmetry=True)
        jw_hamiltonian = jordan_wigner(hubbard)
        jw_hamiltonian.compress()
        operator = create_observable_from_openfermion_text(str(jw_hamiltonian))
        if n_qubits != 8:
            assert False, "このハミルトニアン設定で大丈夫か？"

    elif setting["hamiltonian_name"] == HamiltonianName.Debug:
        """一次元格子の横磁場イジングモデル"""
        for k in range(len(_lattice)):
            operator.add_operator(1, f"X {_lattice[k][0]}X {_lattice[k][1]}")
            # M += np.dot(Z[_lattice[k][0]], Z[_lattice[k][1]])
        for i in range(n_qubits):
            operator.add_operator(setting["h"], f"X {i}")
        for i in range(n_qubits):
            operator.add_operator(setting["h"], f"Z {i}")
    elif setting["hamiltonian_name"] == HamiltonianName.Hubbard_2d_change:
        """周期的境界条件なし, harf-filling, -Jと定義している点に注意"""
        if n_qubits == 8 or n_qubits == 12 or n_qubits == 16 or n_qubits == 20 or n_qubits == 24:
            # 8qubitの場合
            _ham = '''-0.5 [X0 Z1 X2] +
            -0.5 [Y0 Z1 Y2] +
            -0.5 [X0 Z1 Z2 Z3 X4] +
            -0.5 [Y0 Z1 Z2 Z3 Y4] +
            -0.5 [X1 Z2 X3] +
            -0.5 [Y1 Z2 Y3] +
            -0.5 [X1 Z2 Z3 Z4 X5] +
            -0.5 [Y1 Z2 Z3 Z4 Y5] +
            -0.5 [X2 Z3 Z4 Z5 X6] +
            -0.5 [Y2 Z3 Z4 Z5 Y6] +
            -0.5 [X3 Z4 Z5 Z6 X7] +
            -0.5 [Y3 Z4 Z5 Z6 Y7] +
            -0.5 [X4 Z5 X6] +
            -0.5 [Y4 Z5 Y6] +
            -0.5 [X5 Z6 X7] +
            -0.5 [Y5 Z6 Y7] +
            0.25 [Z0 Z1] +
            0.25 [Z2 Z3] +
            0.25 [Z4 Z5] +
            0.25 [Z6 Z7]'''
        else:
            assert False, "qubit数を確認してください"

        if n_qubits == 12 or n_qubits == 16 or n_qubits == 20 or n_qubits == 24:
            # 12qubitの場合
            _ham += ''' +
            -0.5 [X4 Z5 Z6 Z7 X8] +
            -0.5 [Y4 Z5 Z6 Z7 Y8] +
            -0.5 [X5 Z6 Z7 Z8 X9] +
            -0.5 [Y5 Z6 Z7 Z8 Y9] +
            -0.5 [X6 Z7 Z8 Z9 X10] +
            -0.5 [Y6 Z7 Z8 Z9 Y10] +
            -0.5 [X7 Z8 Z9 Z10 X11] +
            -0.5 [Y7 Z8 Z9 Z10 Y11] +
            -0.5 [X8 Z9 X10] +
            -0.5 [Y8 Z9 Y10] +
            -0.5 [X9 Z10 X11] +
            -0.5 [Y9 Z10 Y11] +
            0.25 [Z8 Z9] +
            0.25 [Z10 Z11]'''

        if n_qubits == 16 or n_qubits == 20 or n_qubits == 24:
            # 16qubitの場合
            _ham += ''' +
            -0.5 [X8 Z9 Z10 Z11 X12] +
            -0.5 [Y8 Z9 Z10 Z11 Y12] +
            -0.5 [X9 Z10 Z11 Z12 X13] +
            -0.5 [Y9 Z10 Z11 Z12 Y13] +
            -0.5 [X10 Z11 Z12 Z13 X14] +
            -0.5 [Y10 Z11 Z12 Z13 Y14] +
            -0.5 [X11 Z12 Z13 Z14 X15] +
            -0.5 [Y11 Z12 Z13 Z14 Y15] +
            -0.5 [X12 Z13 X14] +
            -0.5 [Y12 Z13 Y14] +
            -0.5 [X13 Z14 X15] +
            -0.5 [Y13 Z14 Y15] +
            0.25 [Z12 Z13] +
            0.25 [Z14 Z15]'''
        if n_qubits == 20 or n_qubits == 24:
            # 20qubitの場合
            _ham += ''' +
            -0.5 [X12 Z13 Z14 Z15 X16] +
            -0.5 [Y12 Z13 Z14 Z15 Y16] +
            -0.5 [X13 Z14 Z15 Z16 X17] +
            -0.5 [Y13 Z14 Z15 Z16 Y17] +
            -0.5 [X14 Z15 Z16 Z17 X18] +
            -0.5 [Y14 Z15 Z16 Z17 Y18] +
            -0.5 [X15 Z16 Z17 Z18 X19] +
            -0.5 [Y15 Z16 Z17 Z18 Y19] +
            -0.5 [X16 Z17 X18] +
            -0.5 [Y16 Z17 Y18] +
            -0.5 [X17 Z18 X19] +
            -0.5 [Y17 Z18 Y19] +
            0.25 [Z16 Z17] +
            0.25 [Z18 Z19]'''

        if n_qubits == 24:
            # 24qubitの場合
            _ham += ''' +
        -0.5 [X16 Z17 Z18 Z19 X20] +
        -0.5 [Y16 Z17 Z18 Z19 Y20] +
        -0.5 [X17 Z18 Z19 Z20 X21] +
        -0.5 [Y17 Z18 Z19 Z20 Y21] +
        -0.5 [X18 Z19 Z20 Z21 X22] +
        -0.5 [Y18 Z19 Z20 Z21 Y22] +
        -0.5 [X19 Z20 Z21 Z22 X23] +
        -0.5 [Y19 Z20 Z21 Z22 Y23] +
        -0.5 [X20 Z21 X22] +
        -0.5 [Y20 Z21 Y22] +
        -0.5 [X21 Z22 X23] +
        -0.5 [Y21 Z22 Y23] +
        0.25 [Z20 Z21] +
        0.25 [Z22 Z23]'''

        operator = create_observable_from_openfermion_text(_ham)
    else:
        assert False, "ハミルトニアン名を間違えている"
    return operator