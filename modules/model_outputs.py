import torch
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import MaskedLMOutput, CausalLMOutputWithPast
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

from modules.memory import OnlineProtoNet
# import train_with_llama as train


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
    memories: Optional[List[Dict[str, Any]]] = None

@dataclass
class CausalLMOutputWithNewTokenNegatives(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    positive_loss: Optional[torch.FloatTensor]= None
    negative_loss: Optional[torch.FloatTensor]= None
    positive_logits: torch.FloatTensor = None
    negative_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    new_token_loss: Optional[torch.FloatTensor] = None
    memories: Optional[List[Dict[str, Any]]] = None

@dataclass
class CausalLMOutputWithRegressionLoss(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    base_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    base_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    new_token_loss: Optional[torch.FloatTensor] = None
    memories: Optional[List[Dict[str, Any]]] = None
    regression_loss: Optional[torch.FloatTensor] = None
    distillation_loss: Optional[torch.FloatTensor] = None

@dataclass
class CausalLMOutputWithRegressionAndNegativeLoss(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    base_logits: torch.FloatTensor = None
    positive_logits: torch.FloatTensor = None
    negative_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    base_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    new_token_loss: Optional[torch.FloatTensor] = None
    memories: Optional[List[Dict[str, Any]]] = None
    regression_loss: Optional[torch.FloatTensor] = None
    distillation_loss: Optional[torch.FloatTensor] = None
    positive_loss: Optional[torch.FloatTensor] = None
    negative_loss: Optional[torch.FloatTensor] = None
