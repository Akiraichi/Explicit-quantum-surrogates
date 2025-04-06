from dataclasses import dataclass

from mnisq import (  # type: ignore
    load_Fashion_original_f80,
    load_Fashion_original_f90,
    load_Fashion_original_f95,
    load_Kuzushiji_original_f80,
    load_Kuzushiji_original_f90,
    load_Kuzushiji_original_f95,
    load_mnist_original_f80,
    load_mnist_original_f90,
    load_mnist_original_f95,
)


@dataclass(frozen=True)
class Dataset:
    class MNISQ:
        class Name:
            MNIST: str = "mnisq_mnist"
            FashionMNIST: str = "mnisq_fashionmnist"
            Kuzusizi: str = "mnisq_kuzusizi"

            @classmethod
            def values(cls):
                return {cls.MNIST, cls.FashionMNIST, cls.Kuzusizi}

        class Fidelity:
            F80: float = 0.8
            F90: float = 0.9
            F95: float = 0.95

        class Qubits:
            q10: int = 10  # MNISQデータセットは10量子ビットのみ選択可能

        @staticmethod
        def load_data_for_mnisq(ds_name: str, fidelity: float):
            load_functions = {
                (
                    Dataset.MNISQ.Name.MNIST,
                    Dataset.MNISQ.Fidelity.F80,
                ): load_mnist_original_f80,
                (
                    Dataset.MNISQ.Name.MNIST,
                    Dataset.MNISQ.Fidelity.F90,
                ): load_mnist_original_f90,
                (
                    Dataset.MNISQ.Name.MNIST,
                    Dataset.MNISQ.Fidelity.F95,
                ): load_mnist_original_f95,
                (
                    Dataset.MNISQ.Name.FashionMNIST,
                    Dataset.MNISQ.Fidelity.F80,
                ): load_Fashion_original_f80,
                (
                    Dataset.MNISQ.Name.FashionMNIST,
                    Dataset.MNISQ.Fidelity.F90,
                ): load_Fashion_original_f90,
                (
                    Dataset.MNISQ.Name.FashionMNIST,
                    Dataset.MNISQ.Fidelity.F95,
                ): load_Fashion_original_f95,
                (
                    Dataset.MNISQ.Name.Kuzusizi,
                    Dataset.MNISQ.Fidelity.F80,
                ): load_Kuzushiji_original_f80,
                (
                    Dataset.MNISQ.Name.Kuzusizi,
                    Dataset.MNISQ.Fidelity.F90,
                ): load_Kuzushiji_original_f90,
                (
                    Dataset.MNISQ.Name.Kuzusizi,
                    Dataset.MNISQ.Fidelity.F95,
                ): load_Kuzushiji_original_f95,
            }

            load_func = load_functions.get((ds_name, fidelity))
            assert not (load_func is None), "Invalid dataset name or fidelity"

            return load_func()

    class VQEGeneratedDataset:
        class Name:
            VQEGeneratedDataset: str = "vqe_generated_dataset"

            @classmethod
            def values(cls):
                return {cls.VQEGeneratedDataset}

        class Qubits:
            q4: int = 4
            q8: int = 8
            q12: int = 12
            q16: int = 16
            q20: int = 20

        @staticmethod
        def get_hamiltonian_labels(n_qubits):
            if n_qubits == Dataset.VQEGeneratedDataset.Qubits.q4:
                hamiltonian_labels = [0, 1, 2, 3, 4]
            elif n_qubits == Dataset.VQEGeneratedDataset.Qubits.q8:
                hamiltonian_labels = [0, 1, 2, 3, 4, 5]
            elif n_qubits in [
                Dataset.VQEGeneratedDataset.Qubits.q12,
                Dataset.VQEGeneratedDataset.Qubits.q16,
                Dataset.VQEGeneratedDataset.Qubits.q20,
            ]:
                hamiltonian_labels = [0, 1, 2, 3, 4, 7]
            else:
                assert False, "Invalid number of qubits"
            return hamiltonian_labels
