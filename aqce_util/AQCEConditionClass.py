import datetime
import glob
import os
from dataclasses import dataclass

import joblib  # type: ignore
from qulacs import QuantumGateMatrix
from qulacs.gate import DenseMatrix

from aqce_util.AQCEDictType import MatrixCDict


@dataclass(frozen=True)
class AQCECondition:
    """
    Configuration class for AQCE.

    This class stores parameters and settings for the AQCE algorithm, which is used
    to find quantum circuits that represent target quantum states. It includes settings
    for circuit size, optimization parameters, and noise simulation.

    Attributes:
        n_qubits: Number of qubits in the target quantum circuit
        k: Number of quantum states to embed in the quantum circuit
        M_0: Initial number of quantum gates in the AQCE-created circuit
        M_max: Maximum number of quantum gates during optimization
        M_delta: Number of quantum gates to add in each optimization step
        N: Sweep count for optimization
        Max_fidelity: Maximum fidelity threshold for optimization
        data_index: Index to distinguish different AQCEs for multi-class SVM
        optimize_method_str: Optimization method selection (currently only "extended_aqce" is available)
        print_debug: Whether to output debug information
        n_data_by_label: Number of data points per label
        noisy: Whether to include noise in simulations
        n_shots: Number of measurement shots (when including noise)
    """
    n_qubits: int  # Number of qubits in the target quantum circuit
    k: int  # Number of quantum states to embed in the quantum circuit
    M_0: int  # Initial number of quantum gates in the AQCE-created circuit
    M_max: int  # Maximum number of quantum gates during optimization
    M_delta: int  # Number of quantum gates to add in each optimization step
    N: int  # sweep count
    Max_fidelity: float  # Maximum fidelity threshold for optimization
    data_index: int  # Index to distinguish different AQCEs for multi-class SVM
    optimize_method_str: (
        str  # Optimization method selection (currently only "extended_aqce" is available)
    )
    print_debug: bool  # Whether to output debug information
    n_data_by_label: int

    noisy: bool  # Whether to include noise in simulations
    n_shots: int | None  # Number of measurement shots (when including noise)

    def __post_init__(self):
        """
        Validate the Max_fidelity value after initialization.

        Raises:
            ValueError: If Max_fidelity is not between 0 and 1
        """
        if not (0.0 <= self.Max_fidelity <= 1.0):
            raise ValueError("fidelity must be between 0 and 1")

    @staticmethod
    def get_top_folder_name():
        """
        Get the top-level folder name for AQCE data storage.

        Returns:
            str: The name of the top-level folder
        """
        return "aqce_data"

    def get_file_name(self, today, data_index=None):
        """
        Generate a filename for AQCE data based on configuration parameters.

        Args:
            today: Date string to include in the filename
            data_index: Optional index to use instead of self.data_index

        Returns:
            str: Formatted filename for AQCE data
        """
        if data_index is None:
            data_index = self.data_index
        return f"n_qubits={str(self.n_qubits).zfill(2)}-k={str(self.k).zfill(3)}-M0={self.M_0}-Mmax={self.M_max}-Mdelta={self.M_delta}-N={self.N}-Maxf={int(self.Max_fidelity * 100)}-{today}-noisy={self.noisy}-n_shots={self.n_shots}-data_index{str(data_index).zfill(2)}.jb"

    def get_save_path(
        self, svm_filename, ed_cfg_filename, today: str | None = None
    ) -> str:
        """
        Generate the full path for saving AQCE data.

        This method creates a hierarchical directory structure based on the SVM filename,
        eigenvalue decomposition configuration filename, and AQCE parameters.

        Args:
            svm_filename: Path to the SVM model file
            ed_cfg_filename: Path to the eigenvalue decomposition configuration file
            today: Optional date string (if None, current date/time will be used)

        Returns:
            str: Full path for saving AQCE data
        """
        # Directory structure
        first_folder_name = AQCECondition.get_top_folder_name()
        second_folder_name = os.path.basename(svm_filename)[:-3]  # Remove .jb extension
        third_folder_name = os.path.basename(ed_cfg_filename)[:-3]  # Remove .jb extension
        forth_folder_name = f"aqce-n_qubits={str(self.n_qubits).zfill(2)}-k={str(self.k).zfill(3)}-M0={self.M_0}-Mmax={self.M_max}-Mdelta={self.M_delta}-N={self.N}-Maxf={int(self.Max_fidelity * 100)}-noisy={self.noisy}-n_shots={self.n_shots}"
        folder_path = os.path.join(
            first_folder_name, second_folder_name, third_folder_name, forth_folder_name
        )
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)

        # File
        # Include today's date in the filename
        if today is None:
            today = datetime.datetime.today().strftime(
                "%Y-%m-%d-%H-%M-%S"
            )  # Example: '2024-06-18-14-40-24'
        file_name = self.get_file_name(today=today)
        path = os.path.join(folder_path, file_name)
        return path

    def search_aqce_data_path_to_file(
        self, data_index: int, svm_filename: str, ed_cfg_filename: str
    ):
        """
        Get the path to AQCE data with the specified conditions.

        This method searches for AQCE data files matching the given parameters
        and returns the path to the matching file.

        Args:
            data_index: Index to distinguish different AQCEs
            svm_filename: Path to the SVM model file
            ed_cfg_filename: Path to the eigenvalue decomposition configuration file

        Returns:
            str: Path to the matching AQCE data file

        Raises:
            AssertionError: If no file or multiple files match the search criteria
        """
        file_name = self.get_file_name(today="*", data_index=data_index)
        folder_path = os.path.dirname(
            self.get_save_path(
                svm_filename=svm_filename, ed_cfg_filename=ed_cfg_filename
            )
        )
        path_to_file = os.path.join(folder_path, file_name)
        files = glob.glob(path_to_file)
        assert len(files) == 1, f"Please verify: path_to_file={path_to_file}"

        file = files[0]
        return file

    @staticmethod
    def load_matrixC_list(path_to_file: str) -> list[MatrixCDict]:
        """
        Get the quantum circuit constructed by AQCE stored at the specified path.

        This method loads the quantum circuit data from a file and returns the list
        of matrix C dictionaries that define the circuit.

        Args:
            path_to_file: Path to the file containing AQCE data

        Returns:
            list[MatrixCDict]: List of matrix C dictionaries defining the quantum circuit

        Raises:
            AssertionError: If no file or multiple files match the path
        """
        # file_name = f"q{str(aqce_condition.n_qubits).zfill(2)}_k{str(aqce_condition.k).zfill(3)}_M0{aqce_condition.M_0}_Mmax{aqce_condition.M_max}_Mdelta{aqce_condition.M_delta}_N{aqce_condition.N}_Maxf{int(aqce_condition.Max_fidelity * 100)}_*_di{str(aqce_condition.data_index).zfill(2)}.jb"
        # file_path = os.path.join(folder_path, file_name)
        files = glob.glob(path_to_file)
        assert len(files) == 1, f"Please verify: {path_to_file}"
        data = joblib.load(files[0])
        matrixC_list = data["matrixC_list"]

        return matrixC_list

    @staticmethod
    def load_AQCE_circuit_C_dagger(path_to_file: str) -> list[QuantumGateMatrix]:
        """
        Load the quantum circuit constructed by AQCE and return its dagger (adjoint).

        This method loads the quantum circuit data from a file, takes the dagger of each
        gate in the circuit, and returns the list of quantum gates representing C^†.

        Args:
            path_to_file: Path to the file containing AQCE data

        Returns:
            list[QuantumGateMatrix]: List of quantum gates representing C^†
        """
        # Load the quantum circuit according to the path
        matrixC_list = AQCECondition.load_matrixC_list(path_to_file=path_to_file)

        C_dagger = []  # Store quantum gates contained in the quantum circuit
        for C_data in matrixC_list:
            U = C_data["U"]
            i = C_data["i"]
            j = C_data["j"]
            C_dagger_quantum_gate = DenseMatrix([i, j], U.conj().T.copy())
            C_dagger.append(C_dagger_quantum_gate)

        return C_dagger
