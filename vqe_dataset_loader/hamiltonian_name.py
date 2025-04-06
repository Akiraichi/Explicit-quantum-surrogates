from dataclasses import dataclass

@dataclass(frozen=True)
class HamiltonianName:
    Heisenberg_1d: str = "Heisenberg_1d"
    Ising_1d: str = "Ising_1d"
    SSH_1d: str = "SSH_1d"
    Kitaex_chain: str = "Kitaex_chain"
    J1J2model: str = "J1J2model"
    Hubbard_1d: str = "Hubbard_1d"
    Hubbard_2d: str = "Hubbard_2d"
    # spinのインデックスを変えたやつ。
    Hubbard_2d_change: str = "Hubbard_2d_change"
    # debug用のハミルトニアン
    Debug: str = "Debug"