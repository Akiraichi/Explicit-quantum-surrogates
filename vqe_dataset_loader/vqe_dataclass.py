from vqe_dataset_loader.ansatz_define import get_ansatz, get_parameter_set_ansatz
from vqe_dataset_loader.hamiltonian_define import get_defined_hamiltonian
from vqe_dataset_loader.label_setting import get_label_setting


class VQEData:
    def __init__(self, rawdata):
        self.state = rawdata["state"]
        self.parameter = rawdata["parameter"]
        self.energy = rawdata["energy"]
        self.depth = rawdata["depth"]
        self.n_qubits = rawdata["n_qubits"]
        self.hamiltonian_name = rawdata["hamiltonian_name"]
        self.label = rawdata["label"]
        self.reps = rawdata["reps"]
        self.ansatz_name = rawdata["ansatz_name"]
        self.vqe_index = rawdata["vqe_index"]
        self.param_not_set_ansatz = self.__get_param_not_set_ansatz()
        self.param_set_ansatz = self.__get_param_set_ansatz()

    def __get_param_not_set_ansatz(self):
        # 設定の取得
        setting = get_label_setting(n_qubits=self.n_qubits, label=self.label, ansatz_name=self.ansatz_name,
                                    reps=self.reps,
                                    vqe_index=self.vqe_index)  # 設定を取得する
        # ハミルトニアンの取得
        hamiltonian = get_defined_hamiltonian(setting=setting)  # ハミルトニアンを取得
        return get_ansatz(setting=setting, hamiltonian=hamiltonian)

    def __get_param_set_ansatz(self):
        # 設定の取得
        return get_parameter_set_ansatz(__ansatz=self.param_not_set_ansatz.copy(), _param=self.parameter)
