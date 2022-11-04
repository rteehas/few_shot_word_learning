from dataclasses import dataclass

@dataclass
class Config:
    weight_decay: float
    epochs: int
    lr: float
    mlm_prob: float