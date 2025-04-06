import copy
import datetime
import sys
import time
from typing import List

import joblib  # type: ignore
import numpy as np
from jaxtyping import Complex128
from qulacs import QuantumGateMatrix, QuantumState
from qulacs.gate import DenseMatrix, Identity, X, Y, Z
from qulacs.state import inner_product
from scipy.optimize import minimize  # type: ignore

from aqce_util.AQCEConditionClass import AQCECondition
from aqce_util.AQCEDictType import AQCESaveDataDict, MatrixCDict
from misc.util import (
    AQCECounter,
    calc_fidelity_qulacs,
    inner_product_exp,
    repeat_save_until_success,
)
from qnn_util.errors import ConfigurationError

sys.path.append("/home/akimoto/mimi/")


class ExtendedAQCE:
    """Embed orthogonal quantum states specified in target_state_list into a quantum circuit.

    This class implements the AQCE algorithm,
    which finds quantum circuits that represent target quantum states.
    """

    _pauli_matrix = [
        np.array([[1, 0], [0, 1]], dtype=np.complex128),  # I
        np.array([[0, 1], [1, 0]], dtype=np.complex128),  # X
        np.array(
            [
                [0, -1j],
                [1j, 0],
            ],
            dtype=np.complex128,
        ),  # Y
        np.array([[1, 0], [0, -1]], dtype=np.complex128),  # Z
    ]

    def __init__(
        self,
        target_state_list: List[QuantumState],
        condition: AQCECondition,
        coef_ndarray: Complex128[np.ndarray, "k n_support_vectors"] | None = None,
        data_list: list[QuantumState] | None = None,
    ):
        """
        Initialize the Extended AQCE (Adaptive Quantum Circuit Embedding) algorithm.

        This class implements the Extended AQCE algorithm, which finds quantum circuits
        that represent target quantum states. It can be used to convert implicit models
        to explicit quantum circuits.

        Args:
            target_state_list: List of target quantum states to be represented by the circuit
            condition: Configuration object for AQCE execution
            coef_ndarray: Coefficients needed when representing quantum states as linear
                         combinations of training data (used for noisy simulations)
            data_list: Training data needed when representing quantum states as linear
                      combinations of training data (used for noisy simulations)
        """
        # Used for simulations with noise
        self.coef_ndarray = coef_ndarray  # Coefficients needed when representing quantum states as linear combinations of training data
        self.data_list = data_list  # Training data needed when representing quantum states as linear combinations of training data
        self.noisy = condition.noisy
        self.n_shots = condition.n_shots

        # AQCE execution condition settings
        self.n_qubits: int = condition.n_qubits
        self.M_0: int = condition.M_0
        self.M_delta: int = condition.M_delta
        self.M_max: int = condition.M_max
        self.N: int = condition.N
        self.max_fidelity: float = condition.Max_fidelity
        self.optimize_method_str: str = condition.optimize_method_str

        # Initialization of variables used in the algorithm
        self.M: int = 0  # Current number of quantum gates
        self.C: list[QuantumGateMatrix]
        self.C_inv: list[QuantumGateMatrix]
        self.target_state_list: List[QuantumState] = target_state_list

        # Simulation settings
        self.print_debug: bool = condition.print_debug
        self._pauli_matrix_product_cache = None  # Cache

        # For recording
        self.matrixC_list: list[MatrixCDict]
        self.t_list: list[tuple[int, int, float]] = []
        self.fidelity2target_list_list: list[tuple[int, int, list[float]]] = []
        self.unitary_fidelity_lists: list[tuple[int, int, list[float]]] = (
            []
        )  # Fidelity between optimal U and U obtained by gradient method
        self.k: int = len(self.target_state_list)
        self.time_list: list[float] = []  # List for storing time for each loop

        # Count various metrics related to AQCE
        self.counter: AQCECounter = AQCECounter(n_qubits=self.n_qubits)
        self.counter_list: list[AQCECounter] = []

    @property
    def pauli_matrix_product(self):
        """
        Calculate and cache the tensor product of Pauli matrices.

        This property computes the tensor product of all combinations of Pauli matrices
        (I, X, Y, Z) and caches the result for efficient reuse.

        Returns:
            List of lists containing the tensor products of Pauli matrices
        """
        if self._pauli_matrix_product_cache is None:
            self._pauli_matrix_product_cache = [
                [
                    np.kron(self._pauli_matrix[j], self._pauli_matrix[i])
                    for j in range(4)
                ]
                for i in range(4)
            ]
        return self._pauli_matrix_product_cache

    def entanglement_step(self) -> None:
        """
        Perform the entanglement step of the AQCE algorithm.

        This method adds M_delta new quantum gates to the circuit. Each gate is initialized
        as an identity matrix and will be optimized later.
        """
        # Initialize the new quantum gate as identity
        V: Complex128[np.ndarray, "4 4"] = np.eye(4, dtype=np.complex128)
        idx = (0, 1)
        # Add M_delta new quantum gates
        for _ in range(self.M_delta):
            self.C.insert(
                0, DenseMatrix([idx[0], idx[1]], V)
            )  # TODO: Inserting at the beginning of the list is inefficient, but not a major issue
            self.C_inv.insert(
                0, DenseMatrix([idx[0], idx[1]], np.conjugate(V.T))
            )  # TODO: Inserting at the beginning of the list is inefficient, but not a major issue
            self.matrixC_list.insert(0, {"i": idx[0], "j": idx[1], "U": V.copy()})
            # Increment the gate count
            self.M += 1

        # For analysis: update the counter
        self.counter.n_gates = self.M

    def _prepare_phi_states(self, m: int) -> list[QuantumState]:
        """
        Create Phi_state_list with gates to the left of m applied.

        This method creates a list of quantum states by applying the gates to the left
        of the m-th gate to computational basis states.

        Args:
            m: Index of the gate

        Returns:
            List of quantum states with gates to the left of m applied
        """
        phi_list: list[QuantumState] = []
        for i in range(len(self.target_state_list)):
            phi = QuantumState(self.n_qubits)
            phi.set_computational_basis(i)
            for gate_idx in range(m):
                self.C[gate_idx].update_quantum_state(phi)
            phi_list.append(phi)
        return phi_list

    def _prepare_psi_states(self, m: int) -> list[QuantumState]:
        """
        Create Psi_state_list with inverse gates to the right of m applied.

        This method creates a list of quantum states by applying the inverse gates to the right
        of the m-th gate to the target states.

        Args:
            m: Index of the gate

        Returns:
            List of quantum states with inverse gates to the right of m applied
        """
        psi_list: list[QuantumState] = []
        for target_state in self.target_state_list:
            psi = QuantumState(self.n_qubits)
            psi.load(target_state)
            for gate_idx in reversed(range(m + 1, self.M)):
                self.C_inv[gate_idx].update_quantum_state(psi)
            psi_list.append(psi)
        return psi_list

    def _prepare_data_states_exp(self, m: int) -> list[QuantumState]:
        """
        Create data states list with inverse gates to the right of m applied.

        This method creates a list of quantum states by applying the inverse gates to the right
        of the m-th gate to the data states. Used for noisy simulations.

        Args:
            m: Index of the gate

        Returns:
            List of data quantum states with inverse gates to the right of m applied
        """
        assert self.data_list is not None, "self.data_list is None"

        psi_list: list[QuantumState] = []
        for data in self.data_list:
            psi = data.copy()
            for gate_idx in reversed(range(m + 1, self.M)):
                self.C_inv[gate_idx].update_quantum_state(psi)
            psi_list.append(psi)
        return psi_list

    def extended_aqce_optimization_step(self) -> None:
        """
        Perform the optimization step of the Extended AQCE algorithm.

        This method optimizes each gate in the circuit for N sweeps. For each gate,
        it calculates the optimal unitary matrix and its indices, then updates the
        circuit with the new gate.
        """
        for sweep_count in range(self.N):  # Perform N sweeps
            t_start = time.time()
            for m in range(self.M):  # For each of the M gates
                # Calculate the optimal U and its index
                U, idx = self.extended_aqce_optimization_step_inner(m=m)

                self.C[m] = DenseMatrix([idx[0], idx[1]], U)
                self.C_inv[m] = DenseMatrix([idx[0], idx[1]], np.conjugate(U.T))
                self.matrixC_list[m] = {
                    "i": idx[0],
                    "j": idx[1],
                    "U": U,
                }
                if self.print_debug:
                    print("m: ", m)
            if self.print_debug:
                self.print_info(sweep_count=sweep_count)
            self.t_list.append((self.M, sweep_count, time.time() - t_start))

        # For analysis: update the counter
        self.counter.n_sweeps += self.N

    @staticmethod
    def _calculate_phase_angles(
        F_list: List[Complex128[np.ndarray, "4 4"]],
        U: Complex128[np.ndarray, "4 4"],
    ) -> np.ndarray:
        """
        Calculate phase angles for updating the unitary matrix.

        Args:
            F_list: List of F matrices
            U: Current value of the unitary matrix to be updated

        Returns:
            theta_list: List of calculated phase angles
        """
        theta_list = np.zeros(len(F_list))  # Initialize theta_list
        for k, F_k in enumerate(F_list):
            value_k = np.trace(F_k @ U.conj().T)
            theta_k = np.angle(
                value_k
            )  # Calculate the phase angle (θ) of the complex number (in radians)
            theta_list[k] = theta_k
        return theta_list

    def extended_aqce_optimization_step_inner(
        self,
        m: int,
    ) -> tuple[Complex128[np.ndarray, "4 4"], tuple[int, int]]:
        """
        Inner optimization step for the Extended AQCE algorithm.

        This method calculates the optimal unitary matrix and its indices for the m-th gate
        in the circuit. It tries all possible qubit index combinations and selects the one
        that maximizes the fidelity.

        Args:
            m: Index of the gate to optimize

        Returns:
            Tuple containing:
            - The optimal unitary matrix
            - The tuple of qubit indices (i,j) for the gate
        """
        idx = (-1, -1)
        max_value = -np.inf

        Phi_state_list = self._prepare_phi_states(m=m)
        Psi_state_list = self._prepare_psi_states(m=m)
        if self.noisy:
            Psi_data_state_list = self._prepare_data_states_exp(m=m)

        # Calculate optimal fidelity for all index combinations
        tmp_U: Complex128[np.ndarray, "4 4"] = self.matrixC_list[m][
            "U"
        ]  # Current value of the unitary matrix to be updated
        for _i in range(self.n_qubits):
            for _j in range(_i + 1, self.n_qubits):
                # Calculate the Fidelity tensor
                F_list = self._calculate_F_list(
                    target_state_list=self.target_state_list,
                    Phi_state_list=Phi_state_list,
                    Psi_state_list=Psi_state_list,
                    idx=(_i, _j),
                    Psi_data_state_list=Psi_data_state_list if self.noisy else None,
                )

                # Update the phase
                theta_list = self._calculate_phase_angles(F_list=F_list, U=tmp_U)

                # Calculate F_all
                F_all = np.zeros((4, 4), dtype=np.complex128)
                for F_k, theta_k in zip(F_list, theta_list):
                    F_all += np.exp(-1j * theta_k) * F_k

                # Check if this is the optimal index
                D = np.linalg.svd(F_all, compute_uv=False)
                if sum(D) > max_value:
                    max_value = sum(D)
                    idx = (_i, _j)
                    max_F = F_all
        # Calculate the optimal U
        X_, D, Y_ = np.linalg.svd(max_F, full_matrices=True)
        Unew: Complex128[np.ndarray, "4 4"] = X_ @ Y_
        return Unew, idx

    def print_info(self, sweep_count: int) -> None:
        """
        Print information about the current state of the optimization.

        This method calculates and prints the fidelity between the current circuit
        and each target state. It's used for debugging and monitoring the progress
        of the optimization.

        Args:
            sweep_count: Current sweep count
        """
        print(
            self.M,
            sweep_count,
            "_________________________________________________________________________________",
        )
        # Calculate the Fidelity with the target state
        f_list = []
        for i in range(len(self.target_state_list)):
            # Since this is for debugging, calculate without noise
            _f = calc_fidelity_qulacs(
                target_state=self.target_state_list[i],
                C=self.C,
                i=i,
                noisy=False,
                n_shots=None,
            )
            f_list.append(_f)
            print(f"Fidelity of the {i}th state", self.M, sweep_count, _f)
        self.fidelity2target_list_list.append((self.M, sweep_count, f_list))

    def run(self) -> None:
        """
        Run the Extended AQCE algorithm.

        This method initializes the quantum circuit and performs optimization
        to find a circuit that represents the target quantum states.
        """
        self.initialise()
        self.perform_optimization()

    def initialise(self) -> None:
        """
        Define the quantum circuit before starting optimization with AQCE.

        This method initializes the quantum circuit with identity gates and
        sets up the necessary data structures for optimization.
        """
        # Initialize with identity gates
        idx = (0, 1)
        I_gate = DenseMatrix(
            index_list=[idx[0], idx[1]], matrix=np.eye(4, dtype=np.complex128)  # type: ignore
        )
        # Define a list of quantum gates that make up the quantum circuit U
        self.C = [I_gate for _ in range(self.M_0)]
        # Hold the inverse gate of each quantum gate in self.C
        # TODO: Currently qulacs has an inverse function, so it may not be necessary to store these
        self.C_inv = [I_gate for _ in range(self.M_0)]
        # Store quantum gates as matrices since there was no function to convert quantum gates to matrices
        # TODO: Current version of qulacs might have this functionality
        self.matrixC_list = [
            {"i": idx[0], "j": idx[1], "U": np.eye(4, dtype=np.complex128)}
            for _ in range(self.M_0)
        ]
        self.M = self.M_0  # Update the current number of quantum gates

        # For analysis: update the counter
        self.counter.n_gates = self.M

    def perform_optimization(self) -> None:
        """
        Execute optimization until the number of quantum gates exceeds M_max or fidelity exceeds max_fidelity.

        This method iteratively adds gates to the circuit and optimizes them until either
        the maximum number of gates is reached or the desired fidelity is achieved.
        """
        fidelity_list: list[float] = [0.0] * len(self.target_state_list)
        while self.M < self.M_max and any(
            fidelity < self.max_fidelity for fidelity in fidelity_list
        ):
            t1 = time.time()
            if self.optimize_method_str == "extended_aqce":
                self.entanglement_step()
                self.extended_aqce_optimization_step()
            else:
                raise ConfigurationError("Please check optimize_method_str")

            # Calculate fidelity for each quantum state and update the list to determine whether to continue optimization
            for i in range(len(fidelity_list)):
                fidelity_list[i] = calc_fidelity_qulacs(
                    target_state=self.target_state_list[i],
                    C=self.C,
                    i=i,
                    noisy=self.noisy,
                    n_shots=self.n_shots,
                )
                self.counter.n_hadamard_test += (
                    1  # Since we're calculating fidelity, it only increases once
                )

            # For analysis: update the counter
            self.counter.fidelity_list = fidelity_list.copy()
            self.counter_list.append(copy.deepcopy(self.counter))
            self.time_list.append(time.time() - t1)

    def _calc_F(
        self,
        Phi_state: QuantumState,
        Psi_state: QuantumState,
        idx: tuple[int, int],
    ) -> Complex128[np.ndarray, "4 4"]:
        """
        Calculate the Fidelity tensor at index idx.

        Args:
            Phi_state: First quantum state
            Psi_state: Second quantum state
            idx: Tuple of indices (i,j) for the qubits to apply Pauli operators

        Returns:
            4x4 complex matrix representing the Fidelity tensor
        """
        i = idx[0]
        j = idx[1]

        pauli_i = [Identity(i), X(i), Y(i), Z(i)]
        pauli_j = [Identity(j), X(j), Y(j), Z(j)]
        pauli_psi_state = QuantumState(Psi_state.get_qubit_count())

        F = np.zeros((4, 4), dtype=np.complex128)
        for k in range(4):
            for l in range(4):
                pauli_psi_state.load(Psi_state)
                pauli_i[k].update_quantum_state(pauli_psi_state)
                pauli_j[l].update_quantum_state(pauli_psi_state)
                f_kl = inner_product(Phi_state, pauli_psi_state) / 4
                F += f_kl * self.pauli_matrix_product[k][l]

        return F

    def _calculate_F_list(
        self,
        target_state_list: list[QuantumState],
        Phi_state_list: list[QuantumState],
        Psi_state_list: list[QuantumState],
        idx: tuple[int, int],
        Psi_data_state_list: list[QuantumState] | None = None,
    ) -> list[Complex128[np.ndarray, "4 4"]]:
        """
        Function to create F_list.

        This function calculates a list of Fidelity tensors for each target state.

        Args:
            target_state_list: List of target quantum states
            Phi_state_list: List of Phi quantum states
            Psi_state_list: List of Psi quantum states
            idx: Tuple of indices (i,j) for the qubits to apply Pauli operators
            Psi_data_state_list: Optional list of data quantum states with gates applied, used for noisy simulations

        Returns:
            List of 4x4 complex matrices representing the Fidelity tensors
        """
        if self.noisy:
            assert Psi_data_state_list is not None, "Psi_data_state_list is None"
            assert self.coef_ndarray is not None, "coef_ndarray is None"

            F_list = []
            for index, (Phi_state, Psi_state) in enumerate(
                zip(
                    Phi_state_list,
                    Psi_state_list,
                )
            ):
                result = self._calc_F_exp(
                    Phi_state=Phi_state,
                    Psi_state__=Psi_state,
                    Psi_data_state_list=Psi_data_state_list,
                    idx=idx,
                    gamma_array=self.coef_ndarray[index],
                )
                F_list.append(result)
        else:
            F_list = [
                self._calc_F(
                    Phi_state=Phi_state,
                    Psi_state=Psi_state,
                    idx=idx,
                )
                for Psi_state, Phi_state in zip(
                    Psi_state_list,
                    Phi_state_list,
                )
            ]
        return F_list

    def _calc_F_exp(
        self,
        Phi_state: QuantumState,
        Psi_data_state_list: List[QuantumState],
        idx: tuple[int, int],
        gamma_array: Complex128[np.ndarray, "n_support_vectors"],
        check: bool = False,
        Psi_state__: QuantumState | None = None,  # For debugging
    ) -> Complex128[np.ndarray, "4 4"]:
        """
        Calculate the Fidelity tensor at index idx for states represented as linear combinations.

        This method calculates the Fidelity tensor when the quantum states to be embedded
        are represented as linear combinations of training data.

        Args:
            Phi_state: First quantum state
            Psi_data_state_list: List of training data quantum states with gates applied
            idx: Tuple of indices (i,j) for the qubits to apply Pauli operators
            gamma_array: Coefficients when eigenvectors are represented as linear combinations of training data
            check: Whether to perform verification checks
            Psi_state__: Optional state for debugging

        Returns:
            4x4 complex matrix representing the Fidelity tensor
        """
        # Turn off noise when verifying operation
        if check:
            noisy = False
        else:
            noisy = self.noisy

        # Define variables in advance
        i = idx[0]
        j = idx[1]
        pauli_i = [Identity(i), X(i), Y(i), Z(i)]
        pauli_j = [Identity(j), X(j), Y(j), Z(j)]

        """Simple implementation"""
        F = np.zeros((4, 4), dtype=np.complex128)
        for k in range(4):
            for l in range(4):
                f_kl = 0
                for index, Psi_state in enumerate(Psi_data_state_list):
                    pauli_psi_state = Psi_state.copy()
                    pauli_i[k].update_quantum_state(pauli_psi_state)
                    pauli_j[l].update_quantum_state(pauli_psi_state)

                    f_kl_gamma = inner_product_exp(
                        vec1=Phi_state.get_vector(),
                        vec2=pauli_psi_state.get_vector(),
                        noisy=noisy,
                        n_shots=self.n_shots,
                    )
                    f_kl += gamma_array[index] * f_kl_gamma / 4

                F += f_kl * self.pauli_matrix_product[k][l]

        """Synthesized Pauli operators but it became slower"""
        # pauli_psi_state = QuantumState(self.n_qubits)
        #
        # F = np.zeros((4, 4), dtype=np.complex128)
        # for k in range(4):
        #     for l in range(4):
        #         f_kl = 0
        #         for gamma, Psi_state in zip(gamma_list, Psi_data_state_list):
        #             pauli_psi_state.load(Psi_state)
        #             self.marge_pauli_list[i][j][k][l].update_quantum_state(pauli_psi_state)
        #
        #             f_kl_gamma = inner_product_exp(vec1=Phi_state.get_vector(),
        #                                            vec2=pauli_psi_state.get_vector(),
        #                                            noisy=noisy, n_shots=self.n_shots)
        #             f_kl += gamma * f_kl_gamma / 4
        #
        #         F += f_kl * self.pauli_matrix_product[k][l]

        """Speed-up version. It didn't get faster at all."""
        # F = np.zeros((4, 4), dtype=np.complex128)
        # Phi_state_vector = Phi_state.get_vector()
        #
        # # Pre-calculate to reduce computational cost
        # pauli_psi_state = QuantumState(self.n_qubits)
        # for gamma, Psi_state in zip(gamma_list, Psi_data_state_list):
        #     # First load the state and make a copy
        #     pauli_psi_state.load(Psi_state)
        #     original_psi_state = pauli_psi_state.copy()
        #     for k in range(4):
        #         # Apply the k-th Pauli operator
        #         pauli_psi_state.load(original_psi_state)
        #         pauli_i[k].update_quantum_state(pauli_psi_state)
        #         # Make a copy of the state vector created here
        #         vec_after_k = pauli_psi_state.copy()
        #         for l in range(4):
        #             pauli_psi_state.load(vec_after_k)
        #             pauli_j[l].update_quantum_state(pauli_psi_state)
        #             small_f_gamma_kl = inner_product_exp(vec1=Phi_state_vector,
        #                                                  vec2=pauli_psi_state.get_vector(),
        #                                                  noisy=noisy, n_shots=self.n_shots)
        #             f_kl = gamma * small_f_gamma_kl / 4
        #
        #             F += f_kl * self.pauli_matrix_product[k][l]
        #
        #             # For analysis: update the counter
        #             self.counter.n_hadamard_test += 2  # Two times for real and imaginary parts

        if check:
            assert Psi_state__ is not None, "Psi_state__ is None"
            F_true = self._calc_F(
                Phi_state=Phi_state,
                Psi_state=Psi_state__,
                idx=idx,
            )
            print("Display norm difference:", np.linalg.norm(F_true - F, ord="fro"))

        return F

    def save(self, save_path) -> AQCESaveDataDict:
        """
        Save the results of the AQCE algorithm to a file.

        This method saves the quantum circuit, target states, fidelity values,
        and other data from the AQCE algorithm execution to a file for later use.

        Args:
            save_path: Path where the results will be saved

        Returns:
            AQCESaveDataDict: Dictionary containing the saved data

        Raises:
            RuntimeError: If saving fails after multiple attempts
        """
        # Convert to ndarray for saving
        target_state_list = [state.get_vector() for state in self.target_state_list]
        if self.data_list is not None:
            data_list = [data.get_vector() for data in self.data_list]
        else:
            data_list = None

        save_data: AQCESaveDataDict = {
            "fidelity2target_list_list": self.fidelity2target_list_list,
            # Fidelity between quantum states created by self.C and target_state
            "matrixC_list": self.matrixC_list,
            "t_list": self.t_list,
            "target_state_list": target_state_list,
            "unitary_fidelity_list": self.unitary_fidelity_lists,
            "k": self.k,
            "max_fidelity": self.max_fidelity,
            "time_list": self.time_list,
            "today": datetime.datetime.today().strftime(
                "%Y-%m-%d-%H-%M-%S"
            ),  # Example: '2024-06-18-14-40-24'
            #
            "noisy": self.noisy,
            "n_shots": self.n_shots,
            "coef_array_array": self.coef_ndarray,
            "data_list": data_list,
            "counter_list": self.counter_list,
        }

        if repeat_save_until_success(data=save_data, path=save_path, retry_count=10):
            print(f"Data saved successfully to {save_data}.")
        else:
            raise RuntimeError(
                f"Failed to save data to {save_data} after multiple attempts."
            )
        # joblib.dump(value=save_data, filename=file_name)
        return save_data
