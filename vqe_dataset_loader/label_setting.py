import networkx as nx # type: ignore

from vqe_dataset_loader.hamiltonian_name import HamiltonianName


def get_label_setting(n_qubits: int, label: int, ansatz_name: str, reps, vqe_index):
    """ラベルを設定する。"""
    assert 0 <= label <= 9, "ラベル番号が範囲外です"
    if n_qubits >= 12:
        assert label != 5, "ラベル番号が5になっていますが大丈夫ですか？"
    if label == 0:
        """一次元格子の横磁場イジングモデル"""
        h = 2.0
        setting = {
            "n_qubits": n_qubits,
            "hamiltonian_name": HamiltonianName.Ising_1d,
            "label": label,
            "lattice": lattice_map(n=n_qubits),
            "h": h,
            "reps": reps,
            "ansatz_name": ansatz_name,
            "vqe_index": vqe_index,
        }
    elif label == 1:
        """一次元格子のハイゼンベルグモデル"""
        h = 2.0
        setting = {
            "n_qubits": n_qubits,
            "hamiltonian_name": HamiltonianName.Heisenberg_1d,
            "label": label,
            "lattice": lattice_map(n=n_qubits),
            "h": h,
            "reps": reps,
            "ansatz_name": ansatz_name,
            "vqe_index": vqe_index,
        }
    elif label == 2:
        """一次元格子のSSHモデル"""
        delta = 1.5
        setting = {
            "n_qubits": n_qubits,
            "hamiltonian_name": HamiltonianName.SSH_1d,
            "label": label,
            "lattice": lattice_map(n=n_qubits),
            "delta": delta,
            "reps": reps,
            "ansatz_name": ansatz_name,
            "vqe_index": vqe_index,
        }
    elif label == 3:
        """一次元格子のJ1-J2モデル"""
        J2 = 3
        setting = {
            "n_qubits": n_qubits,
            "hamiltonian_name": HamiltonianName.J1J2model,
            "label": label,
            "lattice": lattice_map(n=n_qubits),
            "J2": J2,
            "reps": reps,
            "ansatz_name": ansatz_name,
            "vqe_index": vqe_index,
        }
    elif label == 4:
        """1次元格子, harf-fillingのHubbard模型"""
        U = 1.0
        setting = {
            "n_qubits": n_qubits,
            "hamiltonian_name": HamiltonianName.Hubbard_1d,
            "label": label,
            "lattice": lattice_map(n=n_qubits),
            "U": U,
            "reps": reps,
            "ansatz_name": ansatz_name,
            "vqe_index": vqe_index,
        }
    elif label == 5:
        """2次元ラダー格子, harf-fillingのHubbard模型"""
        U = 1.0
        setting = {
            "n_qubits": n_qubits,
            "hamiltonian_name": HamiltonianName.Hubbard_2d,
            "label": label,
            "lattice": lattice_map(n=n_qubits),  # 使用していない。
            "U": U,
            "reps": reps,
            "ansatz_name": ansatz_name,
            "vqe_index": vqe_index,
        }
    elif label == 6:
        """デバッグ用の一次元格子の横磁場イジングモデル"""
        h = 2.0
        setting = {
            "n_qubits": n_qubits,
            "hamiltonian_name": HamiltonianName.Debug,
            "label": label,
            "lattice": lattice_map(n=n_qubits),
            "h": h,
            "reps": reps,
            "ansatz_name": ansatz_name,
            "vqe_index": vqe_index,
        }
    elif label == 7:
        """2次元ラダー格子, harf-fillingのHubbard模型, spinのindexを変えた"""
        U = 1.0
        setting = {
            "n_qubits": n_qubits,
            "hamiltonian_name": HamiltonianName.Hubbard_2d_change,
            "label": label,
            "lattice": lattice_map(n=n_qubits),  # 使用していない
            "U": U,
            "reps": reps,
            "ansatz_name": ansatz_name,
            "vqe_index": vqe_index,
        }
    else:
        assert False, "ラベル番号が正しくありません"

    return setting


def lattice_map(n):
    G = nx.path_graph(n=n, create_using=None)
    G = nx.convert_node_labels_to_integers(G)
    return list(G.edges)
