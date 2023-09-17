import torch
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import MaskedLMOutput, CausalLMOutputWithPast
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

from modules.memory import OnlineProtoNet


class MaskLMOutputWithNewToken(MaskedLMOutput):

    def __init__(self, loss, logits, hidden_states, attentions, new_token_loss, memories):

        super().__init__(loss=loss, logits=logits, hidden_states=hidden_states, attentions=attentions)
        self.new_token_loss = new_token_loss
        self.memories=memories

@dataclass
class CausalLMOutputWithNewToken(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    new_token_loss: Optional[torch.FloatTensor] = None
    memories: Optional[List[Dict[str, OnlineProtoNet]]] = None