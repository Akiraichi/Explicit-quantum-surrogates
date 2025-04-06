import os
from collections import Counter
from typing import List, Optional

import joblib  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from jaxtyping import Complex128, Int64
from qulacs import QuantumState
from sklearn.model_selection import train_test_split  # type: ignore

from misc import util
from misc.DatasetClass import Dataset
from misc.util import debug_print, list2str
from qnn_util.QNNConditionClass import QNNCondition
from svm_util.SVMConditionClass import SVMCondition
from vqe_dataset_loader.vqe_dataclass import VQEData
from vqe_dataset_loader.vqe_dataset_loader import load_vqe_dataset


def get_cache_file_path(
    ds_name: str,
    labels: list[int],
    n_data_by_label: int,
    fidelity: Optional[float],
    n_qubits: int,
    state_type: str,
    start_index: int,
    cache_dir: str = "cache_files",
) -> str:
    """
    Returns the path to the cache file.
    """
    cache_file_name = f"ds_name{ds_name}-labels{list2str(labels)}-n_data_by_label{n_data_by_label}-fidelity{fidelity}-n_qubits{n_qubits}-state_type{state_type}-start_index{start_index}.jb"

    os.makedirs(cache_dir, exist_ok=True)
    cache_file_path = os.path.join(cache_dir, cache_file_name)
    return cache_file_path


def get_cache_file_path_for_vqe_dataset(
    n_qubits: int,
    cache_dir: str = os.path.join("cache_files", "vqe_generated_dataset"),
) -> str:
    """
    Returns the path to the cache file for VQE dataset.
    """
    cache_file_name = f"n_qubits={n_qubits}.jb"

    os.makedirs(cache_dir, exist_ok=True)
    cache_file_path = os.path.join(cache_dir, cache_file_name)
    return cache_file_path


def load_from_cache(
    cache_file_path: str,
) -> (
    tuple[
        Complex128[np.ndarray, "n_samples n_features"], Int64[np.ndarray, "n_samples"]
    ]
    | None
):
    """
    Loads and returns data from the cache file if it exists.
    Returns None if the file doesn't exist.
    """
    if os.path.exists(cache_file_path):
        print(f"Loading data from joblib cache: {cache_file_path}")
        return joblib.load(cache_file_path)
    return None


def load_VQEData_from_cache(
    cache_file_path: str,
) -> List[VQEData] | None:
    """
    Loads and returns VQE data from the cache file if it exists.
    Returns None if the file doesn't exist.
    """
    if os.path.exists(cache_file_path):
        print(f"Loading data from joblib cache: {cache_file_path}")
        return joblib.load(cache_file_path)
    return None


def save_to_cache(
    data: tuple[
        Complex128[np.ndarray, "n_samples n_features"], Int64[np.ndarray, "n_samples"]
    ],
    cache_file_path: str,
):
    """
    Saves a tuple of (X, y) to the cache file.
    """
    print(f"Saving to joblib cache: {cache_file_path}")
    joblib.dump(data, cache_file_path)


def save_VQEdata_to_cache(
    data: List[VQEData],
    cache_file_path: str,
):
    """
    Saves VQE data to the cache file.
    """
    print(f"Saving to joblib cache: {cache_file_path}")
    joblib.dump(data, cache_file_path)


def label_rewrite(
    X: Complex128[np.ndarray, "n_samples n_features"],
    y: Int64[np.ndarray, "n_samples"],
    target_label: int,
    debug: bool = False,
) -> tuple[
    Complex128[np.ndarray, "n_samples n_features"], Int64[np.ndarray, "n_samples"]
]:
    """
    Converts target_label to 1 and all other labels to 0 for binary classification.

    Args:
        X: Feature data
        y: Label data
        target_label: The label to be converted to 1
        debug: Whether to print debug information

    Returns:
        Tuple of (X, y) with modified labels
    """
    for i in range(len(y)):
        if y[i] == target_label:
            y[i] = 1
        else:
            y[i] = 0

    debug_print(str(Counter(y)), debug_print=debug)
    return X, y


def load_dataset(
    ds_name: str,
    labels: list[int],
    n_data_by_label: int,
    fidelity: float | None,
    n_qubits: int,
    state_type="ndarray",  # only ndarray is implemented
    start_index: int = 0,
    use_cache: bool = True,
) -> tuple[
    Complex128[np.ndarray, "n_samples n_features"], Int64[np.ndarray, "n_samples"]
]:
    """
    Load a dataset.

    TODO: If this function's processing is heavy, it can be made lighter by rewriting it using generators.
    Actually, I wish the mnisq library had used generators.

    Args:
        ds_name: Dataset name. Can be selected from "mnisq_mnist", "mnisq_fashionmnist", "mnisq_kuzusizi", "vqe_generated_dataset".
        labels: Labels to load from the dataset.
                For VQE dataset, this parameter is ignored and hamiltonian_labels are used.
        n_data_by_label: Number of data points per label.
                         For VQE dataset, this parameter is ignored.
        fidelity: Dataset fidelity. Only effective for MNISQ dataset.
                  For VQE dataset, this parameter is ignored.
        n_qubits: Number of qubits for the data to be loaded.
                  For MNISQ dataset, this parameter is ignored.
        state_type: Whether to get data as ndarray or Quantum State. Currently only "ndarray" is implemented.
        start_index: The index to start retrieving data from the dataset. For example, if you want to get data
                    from the 5th item, set start_index=4 (0 start).
        use_cache: Whether to use cached data if available.

    Returns:
        Tuple of (X, y) where X is the feature data and y is the label data.
    """

    # 1) Get the cache file path
    cache_file_path = get_cache_file_path(
        ds_name=ds_name,
        labels=labels,
        n_data_by_label=n_data_by_label,
        fidelity=fidelity,
        n_qubits=n_qubits,
        state_type=state_type,
        start_index=start_index,
    )

    # 2) If use_cache=True and the file exists, load from cache
    if use_cache:
        cached_data = load_from_cache(cache_file_path)
        if cached_data is not None:
            return cached_data  # Return (X, y) from cache

    # Load data from dataset and perform preprocessing
    if ds_name in Dataset.MNISQ.Name.values():
        print("When selecting MNISQ dataset, the n_qubits argument is ignored")
        assert fidelity is not None, "fidelity is None"

        items = Dataset.MNISQ.load_data_for_mnisq(ds_name=ds_name, fidelity=fidelity)
        if labels is None:
            labels = list(
                range(10)
            )  # TODO: Hardcoding that there are 10 labels [0,1,...,9]
        X, y = mnisq_data_preprocessing(
            data_dict=items,
            labels=labels,
            n_data_by_label=n_data_by_label,
            state_type=state_type,
            start_index=start_index,
        )
    elif ds_name in Dataset.VQEGeneratedDataset.Name.values():
        print("When selecting VQE dataset, the n_data_by_label argument is ignored.")
        print("When selecting VQE dataset, the fidelity argument is ignored.")
        print("When selecting VQE dataset, the labels argument is ignored.")

        hamiltonian_labels = Dataset.VQEGeneratedDataset.get_hamiltonian_labels(
            n_qubits=n_qubits  # TODO: 内部的な事情により、最後のラベル番号が7になる場合があるが、ラベル番号を5に修正しても問題ないと思う。
        )
        if labels is None:
            labels = hamiltonian_labels
        else:
            assert (
                labels in hamiltonian_labels
            ), f"Invalid labels specified. labels: {labels}, available labels: {hamiltonian_labels}"

        # Create a cache for VQE dataset because of its large size
        cache_file_path_vqe = get_cache_file_path_for_vqe_dataset(n_qubits=n_qubits)
        vqe_data_list: List[VQEData] | None = load_VQEData_from_cache(cache_file_path)
        if vqe_data_list is None:
            # Cache file doesn't exist, so create it
            vqe_data_list = load_vqe_dataset(n_qubits=n_qubits, labels=labels)
            save_VQEdata_to_cache(
                data=vqe_data_list, cache_file_path=cache_file_path_vqe
            )

        X, y = vqe_data_preprocessing(vqe_data_list=vqe_data_list)
    else:
        raise RuntimeError("Invalid ds_name specified")

    # Save to cache
    save_to_cache((X, y), cache_file_path)
    return X, y


def vqe_data_preprocessing(
    vqe_data_list: list[VQEData],
) -> tuple[
    Complex128[np.ndarray, "n_samples n_features"], Int64[np.ndarray, "n_samples"]
]:
    """
    Preprocesses VQE data for machine learning.

    Args:
        vqe_data_list: List of VQEData objects

    Returns:
        Tuple of (X, y) where X contains quantum states and y contains target labels
    """
    X = []  # Quantum states of the data
    y = []  # Target variables based on Hamiltonian labels
    for vqe_data in vqe_data_list:
        X.append(vqe_data.state)
        y.append(vqe_data.label)
    return np.asarray(X), np.asarray(y)


def mnisq_data_preprocessing(
    data_dict: dict,
    labels: list[int],
    n_data_by_label: int,
    start_index: int,
    state_type="ndarray",  # Not strictly necessary, but kept because it's included in the filename.
) -> tuple[
    Complex128[np.ndarray, "n_samples n_features"], Int64[np.ndarray, "n_samples"]
]:
    """
    Preprocesses MNISQ dataset for machine learning.

    The preprocessing includes:
    1. Extract only data with labels specified in the 'labels' parameter
    2. Ensure equal number of data points for each label
    3. Convert quantum states obtained with QuantumState to ndarray

    Args:
        data_dict: Dictionary containing the dataset
        labels: List of labels to extract from the dataset
        n_data_by_label: Number of data points to extract for each label
        start_index: Starting index for data extraction
        state_type: Type of state representation (currently only "ndarray" is supported)

    Returns:
        Tuple of (X, y) where X contains quantum states and y contains labels
    """
    # Preprocessing steps for the loaded dataset:
    # (1) Extract only data with labels specified in 'labels'
    # (2) Ensure equal number of data points for each label
    # (3) Convert quantum states obtained with QuantumState to ndarray
    X: list[QuantumState | np.complex128] = []
    y: list = []
    df = pd.DataFrame(data_dict)
    for i in labels:
        filtered_df = df[
            df["label"] == i
        ]  # (1) Extract only data with labels specified in 'labels'
        result_df = filtered_df.iloc[
            start_index : start_index + n_data_by_label
        ]  # Get n_data_by_label items starting from start_index
        state_array: list = []
        for _circuit in list(result_df.circuit):
            _state = QuantumState(qubit_count=10)  # MNISQ dataset uses 10 qubits
            _circuit.update_quantum_state(_state)
            if state_type == "ndarray":
                state_array.append(
                    _state.get_vector()
                )  # (3) Convert quantum states obtained with QuantumState to ndarray
            else:
                raise RuntimeError(f"Please check state_type: {state_type}")
        # Verify data counts
        assert (
            len(state_array) == n_data_by_label
        ), f"Number of X data: {len(X)} does not match n_data_by_label: {n_data_by_label}. Label ID: {i}"
        assert (
            len(result_df.label) == n_data_by_label
        ), f"Number of y data: {len(y)} does not match n_data_by_label: {n_data_by_label}. Label ID: {i}"

        y += list(result_df.label)
        X += state_array

    return np.asarray(X, np.complex128), np.asarray(y, np.int64)


def load_and_split_dataset_for_additional_learning(
    qnn_condition: QNNCondition,
    svm_condition: SVMCondition,
) -> tuple[
    Complex128[np.ndarray, "n_train_samples n_features"],
    Complex128[np.ndarray, "n_test_samples n_features"],
    Int64[np.ndarray, "n_train_samples"],
    Int64[np.ndarray, "n_test_samples"],
    Complex128[np.ndarray, "n_train_samples n_features"],
    Complex128[np.ndarray, "n_test_samples n_features"],
    Int64[np.ndarray, "n_train_samples"],
    Int64[np.ndarray, "n_test_samples"],
]:
    """
    Loads a dataset and splits it into training and test data for additional learning.
    Labels are rewritten to 0 and 1 for binary classification.

    Parameters
    ----------
    qnn_condition : QNNCondition
        Condition class containing dataset name, label information, etc.
    svm_condition : SVMCondition
        Condition class for SVM training.

    Returns
    -------
    X_train_original, X_test_original, y_train_original, y_test_original, X_train, X_test, y_train, y_test
        Tuple containing both original and new training and test data.
    """

    # Load the data used for training
    X_train_original, X_test_original, y_train_original, y_test_original = (
        svm_condition.load_dataset()
    )
    # For additional learning, convert labels to 0 and 1 for binary classification
    X_train_original, y_train_original = label_rewrite(
        X=X_train_original, y=y_train_original, target_label=qnn_condition.target_label
    )

    X_test_original, y_test_original = label_rewrite(
        X=X_test_original, y=y_test_original, target_label=qnn_condition.target_label
    )

    # For additional learning, we don't use the data that was used to train the implicit model. Therefore, new data is needed.
    # MNISQ MNIST dataset has abundant data, but VQE-generated dataset has limited data, so we branch the processing
    if qnn_condition.ds_name in Dataset.MNISQ.Name.values():
        X, y = load_dataset(
            ds_name=qnn_condition.ds_name,
            labels=qnn_condition.svm_trained_label,  # TODO Check this. It was originally 'labels'.
            n_data_by_label=qnn_condition.n_data_by_label,
            fidelity=qnn_condition.fidelity,
            n_qubits=qnn_condition.n_qubits,
            start_index=qnn_condition.start_index,
            state_type="ndarray",
            # For additional learning, specify start_index to avoid overlap with data used for training the implicit model
            use_cache=True,
        )

        # Shuffle the dataset and split into training and test data (using stratify option)
        X, y = util.data_shuffle(
            X=X, y=y, seed=qnn_condition.seed
        )  # TODO: Enable seed value
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=qnn_condition.test_size,
            random_state=qnn_condition.seed,
            stratify=y,
        )

        X_test, y_test = label_rewrite(
            X=X_test, y=y_test, target_label=qnn_condition.target_label
        )
        X_train, y_train = label_rewrite(
            X=X_train, y=y_train, target_label=qnn_condition.target_label
        )

    elif qnn_condition.ds_name in Dataset.VQEGeneratedDataset.Name.values():
        # VQE-generated dataset has limited data
        # We only consider gradient evaluation, not additional learning. This works even with limited data
        # Therefore, we can use the data as is
        X_train, X_test, y_train, y_test = (
            X_test_original.copy(),
            X_test_original.copy(),
            y_test_original.copy(),
            y_test_original.copy(),
        )  # All test data
    else:
        raise RuntimeError("Invalid ds_name specified")

    return (
        X_train_original,
        X_test_original,
        y_train_original,
        y_test_original,
        X_train,
        X_test,
        y_train,
        y_test,
    )
