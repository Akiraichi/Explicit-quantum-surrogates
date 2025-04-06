import os
import pathlib
from typing import Dict, List

import joblib  # type: ignore
import pandas as pd  # type: ignore

from vqe_dataset_loader.ansatz_name import AnsatzName
from vqe_dataset_loader.label_setting import get_label_setting
from vqe_dataset_loader.vqe_dataclass import VQEData


def load_vqe_dataset(n_qubits, labels) -> List[VQEData]:
    """
    Load VQE-generated Dataset.

    Note: Please create appropriate quantum state objects from the VQE generated dataset published on Github (in QASM format) according to your environment.
    Alternatively, you can load data stored in cache_files. By default, if the relevant data exists in cache_files, it will be loaded.
    The data stored in cache_files has the same content as the VQE generated dataset published on Github, but is organized using the VQEData class for easier handling.

    The following code works only in my personal environment.

    """

    # Load data with labels included in the 'labels' parameter and with the lowest energy.

    # 1. Load the data
    vqe_index_list = list(range(10))
    ansatz_names = [
        AnsatzName.Hamiltonian,
        AnsatzName.HardwareEfficient,
        AnsatzName.HardwareEfficientFull,
        AnsatzName.HardwareEfficientLadder,
        AnsatzName.HardwareEfficientCrossLadder,
        AnsatzName.TwoLocal,
        AnsatzName.TwoLocalStair,
        AnsatzName.TwoLocalFull,
        AnsatzName.TwoLocalLadder,
        AnsatzName.TwoLocalCrossLadder,
    ]
    reps_list = list(range(3, 33))

    vqe_data_list = []
    for label in labels:
        for ansatz_name in ansatz_names:
            for reps in reps_list:
                print(f"label:{label}, ansatz: {ansatz_name}")
                # Get the data with the minimum energy from the dataset
                minimum_energy_data = get_minimum_energy_data(
                    n_qubits=n_qubits,
                    label=label,
                    ansatz_name=ansatz_name,
                    reps=reps,
                    vqe_index_list=vqe_index_list,
                )
                vqe_data = VQEData(minimum_energy_data)
                vqe_data_list.append(vqe_data)

    return vqe_data_list


def get_minimum_energy_data(n_qubits, label, ansatz_name, reps, vqe_index_list):
    def sort_vqe_results(vqe_results: List[Dict], key: str):
        """Takes a list of vqe_results as input and returns them sorted by vqe_result['energy'] in ascending order"""
        # Convert to DataFrame
        df = pd.DataFrame(vqe_results)

        # Sort by energy in ascending order
        df.sort_values(key, inplace=True)

        # Convert to list of dictionaries
        dicts_in_list = df.to_dict("records")
        return dicts_in_list

    # Load data for all indices in vqe_index_list
    loaded_datas = []
    for vqe_index in vqe_index_list:
        # Get settings
        setting = get_label_setting(
            n_qubits=n_qubits,
            label=label,
            ansatz_name=ansatz_name,
            reps=reps,
            vqe_index=vqe_index,
        )
        # Get data
        loaded_data = load_vqe_data(
            path_str=get_vqe_data_path(setting=setting, mkdir=False)
        )
        loaded_datas.append(loaded_data)

    # Sort the list by energy in ascending order
    sorted_loaded_datas = sort_vqe_results(loaded_datas, key="energy")

    # Get the data with the lowest energy
    minimum_energy_data = sorted_loaded_datas[0]
    return minimum_energy_data


def get_vqe_data_path(setting, mkdir=True):
    """
    vqe_datasetに保存されているデータへのパスを取得する。
    TODO: あなたの環境に合わせてfolder_pathを調整してください。
    :param setting:
    :param vqe_index:
    :param mkdir:
    :return:
    """
    # settingの展開。*使って変数展開させてもいいけど。
    hamiltonian_name = setting["hamiltonian_name"]
    n_qubits = setting["n_qubits"]
    label = setting["label"]
    folder_name = "vqe_dataset"
    ansatz_name = setting["ansatz_name"]
    reps = setting["reps"]
    vqe_index = setting["vqe_index"]

    # 0埋めする
    label = str(label).zfill(2)
    n_qubits = str(n_qubits).zfill(2)
    reps = str(reps).zfill(2)
    vqe_index = str(vqe_index).zfill(2)

    #
    folder_path = f"../ao/{folder_name}/{n_qubits}qubit/{label}/{ansatz_name}"
    if mkdir:
        os.makedirs(folder_path, exist_ok=True)

    file_name = f"label={label}_reps={reps}_index={vqe_index}_{hamiltonian_name}_{ansatz_name}_qubit={n_qubits}.jb"
    return f"{folder_path}/{file_name}"


def load_vqe_data(path_str: str) -> dict:
    """
    Load dataset and return as a qulacs circuit.
    Currently, this loads data saved in jb format and returns it as a Parametric Quantum Circuit class.

    """
    # Convert path:str to pathlib object for easier handling. https://note.nkmk.me/python-pathlib-usage/
    path = pathlib.Path(path_str)

    # Check if the path is valid
    assert path.exists(), "Invalid path"
    # Check that the file extension is .jb
    assert path.suffix == ".jb", "Not a .jb file"

    # Load the data
    loaded_data = joblib.load(str(path))

    # Check if the loaded data is valid
    assert (
        type(loaded_data) is dict
    ), "Loaded data type is not as expected"

    return loaded_data
