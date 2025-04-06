import os
from dataclasses import dataclass
from typing import List, Optional

import joblib  # type: ignore
import numpy as np
from jaxtyping import Complex128, Int64

from misc.DatasetClass import Dataset
from misc.util import list2str
from svm_util.SVMDictType import SVMParametersDict, SVMSaveDataDict


@dataclass(frozen=True)
class SVMCondition:
    """
    Configuration class for SVM experiments.

    This class stores parameters and settings for Support Vector Machine (SVM) experiments
    with quantum data. It includes dataset selection, model configuration, and file paths
    for saving and loading results.

    Attributes:
        seed: Random seed for reproducibility
        n_qubits: Number of qubits in the quantum system
        ds_name: Dataset name
        labels: List of labels to use from the dataset
        n_data_by_label: Number of data points per label
        test_size: Proportion of test data (default 0.5)
        C: SVM regularization parameter (default 1)
        fidelity: Dataset fidelity setting (optional)
        start_index: Starting index for data retrieval (default 0)
    """
    seed: int
    n_qubits: int
    ds_name: str
    labels: List[int]
    n_data_by_label: int
    test_size: float = 0.5  # Proportion of test data
    C: float = 1  # SVM regularization parameter
    fidelity: Optional[float] = None
    start_index: int = 0

    def __post_init__(self):
        """
        Validate parameters after initialization.

        This method checks that test_size and fidelity (if provided) are between 0 and 1,
        and that the combination of n_qubits and ds_name is valid.

        Raises:
            ValueError: If test_size or fidelity is not between 0 and 1, or if the
                       combination of n_qubits and ds_name is invalid
        """
        if not (0.0 <= self.test_size <= 1.0):
            raise ValueError("test_size must be between 0 and 1")
        if self.fidelity is not None:
            if not (0.0 <= self.fidelity <= 1.0):
                raise ValueError("fidelity must be between 0 and 1")

        # TODO: This conditional statement should be coded to be unnecessary
        if (
            self.n_qubits in [4, 8, 12, 16, 20]
            and self.ds_name == Dataset.VQEGeneratedDataset.Name.VQEGeneratedDataset
        ):
            pass
        elif self.n_qubits == 10 and self.ds_name in [
            Dataset.MNISQ.Name.MNIST,
            Dataset.MNISQ.Name.FashionMNIST,
            Dataset.MNISQ.Name.Kuzusizi,
        ]:
            pass
        else:
            raise ValueError("Please check the number of qubits or the dataset name")


    def get_top_folder_name(self) -> str:
        """
        Get the top-level folder name for SVM data storage.

        Returns:
            str: The name of the top-level folder
        """
        return "svm_data"

    def get_file_name_(self) -> str:
        """
        Generate a filename for SVM data based on configuration parameters.

        This method creates a filename that includes all relevant parameters to uniquely
        identify the SVM model, including dataset name, fidelity, number of qubits,
        labels, and other configuration settings.

        Returns:
            str: Formatted filename for SVM data
        """
        if self.fidelity is None:
            fidelity_str = "None"
        else:
            fidelity_str = str(int(self.fidelity * 100))

        return f"svm_model-ds_name{self.ds_name}-fidelity{fidelity_str}-n_qubits{str(self.n_qubits).zfill(2)}-labels{list2str(self.labels)}-n_data_by_label{self.n_data_by_label}-seed{self.seed}-test_size{self.test_size}-C{self.C}-start_index{self.start_index}.jb"

    def get_save_path(self) -> str:
        """
        Generate the full path for saving SVM data.

        This method creates a directory structure based on the top folder name
        and generates a full path for saving SVM data.

        Returns:
            str: Full path for saving SVM data
        """
        first_folder_name = self.get_top_folder_name()
        # Create directory only if it doesn't exist
        if not os.path.exists(first_folder_name):
            os.makedirs(first_folder_name, exist_ok=True)

        file_name = self.get_file_name_()
        path = os.path.join(first_folder_name, file_name)
        return path


    def get_observable_save_path(self) -> str:
        """
        Generate the full path for saving SVM observable data.

        This method creates a directory structure for SVM observable data
        and generates a full path for saving it. The filename includes all
        relevant parameters to uniquely identify the SVM model.

        Returns:
            str: Full path for saving SVM observable data
        """
        if self.fidelity is None:
            fidelity_str = "None"
        else:
            fidelity_str = str(int(self.fidelity * 100))

        folder_name = "svm_observable"
        file_name = f"svm_observable-ds_name{self.ds_name}-fidelity{fidelity_str}-n_qubits{str(self.n_qubits).zfill(2)}-labels{list2str(self.labels)}-n_data_by_label{self.n_data_by_label}-seed{self.seed}-test_size{self.test_size}-C{self.C}-start_index{self.start_index}.jb"
        path = os.path.join(folder_name, file_name)
        # Create directory only if it doesn't exist
        if not os.path.exists(folder_name):
            os.makedirs(folder_name, exist_ok=True)
        return path

    def load_saved_data(self) -> SVMSaveDataDict:
        """
        Load saved SVM data from disk.

        This method loads the SVM data from the path generated by get_save_path().

        Returns:
            SVMSaveDataDict: Dictionary containing the saved SVM data
        """
        return joblib.load(self.get_save_path())

    def load_b_list(self) -> list[float]:
        """
        Load the list of bias terms from saved SVM data.

        This method loads the saved SVM data and extracts the bias terms (b values)
        from each SVM model.

        Returns:
            list[float]: List of bias terms, one for each SVM model
        """
        svm_data_dict: SVMSaveDataDict = self.load_saved_data()
        svm_datas: list[SVMParametersDict] = svm_data_dict["datas"]
        b_list = [svm_data["b_array"][0] for svm_data in svm_datas]
        return b_list

    def load_dataset(
        self,
    ) -> tuple[
        Complex128[np.ndarray, "n_train_samples n_features"],
        Complex128[np.ndarray, "n_test_samples n_features"],
        Int64[np.ndarray, "n_train_samples"],
        Int64[np.ndarray, "n_test_samples"],
    ]:
        """
        Load the training and test datasets from saved SVM data.

        This method loads the saved SVM data and extracts the training and test
        datasets, ensuring they have the correct data types.

        Returns:
            tuple: A tuple containing:
                - X_train: Training feature data
                - X_test: Test feature data
                - y_train: Training label data
                - y_test: Test label data
        """
        svm_data_dict = self.load_saved_data()
        X_train = np.asarray(svm_data_dict["X_train"], dtype=np.complex128)
        y_train = np.asarray(svm_data_dict["y_train"], dtype=np.int64)
        X_test = np.asarray(svm_data_dict["X_test"], dtype=np.complex128)
        y_test = np.asarray(svm_data_dict["y_test"], dtype=np.int64)

        return X_train, X_test, y_train, y_test
