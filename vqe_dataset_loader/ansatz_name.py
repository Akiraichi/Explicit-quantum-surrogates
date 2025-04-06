from dataclasses import dataclass

@dataclass(frozen=True)
class AnsatzName:
    Hamiltonian: str = "Hamiltonian"

    HardwareEfficient: str = "HardwareEfficient"
    HardwareEfficientFull: str = "HardwareEfficient_full"
    HardwareEfficientLadder: str = "HardwareEfficient_ladder"
    HardwareEfficientCrossLadder: str = "HardwareEfficient_cross_ladder"

    TwoLocal: str = "TwoLocal"
    TwoLocalStair: str = "TwoLocal_stair"
    TwoLocalFull: str = "TwoLocal_full"
    TwoLocalLadder: str = "TwoLocal_ladder"
    TwoLocalCrossLadder: str = "TwoLocal_cross_ladder"