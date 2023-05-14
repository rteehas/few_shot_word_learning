from dataclasses import dataclass

@dataclass
class Config:
    weight_decay: float
    epochs: int
    lr: float
    mlm_prob: float

@dataclass
class AggregatorConfig:
    agg_method: str = "mean"

@dataclass
class RNNAggConfig(AggregatorConfig):
    agg_method: str = "RNN"
    input_size: int = 768
    output_size: int = 384

@dataclass
class TransformerCLSConfig(AggregatorConfig):
    agg_method: str = "CLS"
    input_size: int = 768
    hidden_size = 512
    nhead: int = 8
    num_layers: int = 2
    num_positions: int = 50 #corresponds to the max number of contextual examples