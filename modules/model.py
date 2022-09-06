import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel, RobertaForMaskedLM

from mlm import *
from prototype_memory import *

class MorphMemoryModel(nn.Module):

    def __init__(self, model, tokenizer):
        super(MorphMemoryModel, self).__init__()
        self.model = model
        # if self.model_name == "roberta":
        #     self.config = RobertaConfig
        #     self
        self.tokenizer = tokenizer
        self.firstLM = RobertaForMaskedLM.from_pretrained('roberta-base')
        self.secondLM = RobertaForMaskedLM.from_pretrained('roberta-base')
        # self.firstLM.embeddings = self.secondLM.embeddings # share embeddings

        self.memory = OnlineProtoNet()
        self.morph_model = MaskedAttention(tokenizer)
        self.mask_attn_linear = nn.Linear(768, len(self.tokenizer))
        self.l1 = nn.Linear(self.morph_model.d + self.firstLM.model)
        self.opt = opt


    def forward(self, seq, input, nonce):
        lm_labels = self.tokenizer(seq, return_tensors="pt")["input_ids"]
        lm_labels[lm_labels != self.tokenizer.mask_token_id] = -100
        first_out = self.firstLM(**input, labels=lm_labels, output_hidden_states=True)
        first_loss = first_out.loss
        first_hidden = first_out.hidden_states

        masked_layer_out = self.morph_model(first_hidden[-2], input, seq, nonce)
        embs = self.mask_attn_linear(masked_layer_out)
        msk = self.morph_model.mask_nonce(self.morph_model.calc_nonce_span(nonce, input, seq))
        masked_layer_labels = input.input_ids * msk
        masked_layer_labels[masked_layer_labels==0] = -100
        masked_layer_logits = nn.functional.log_softmax(embs)
        masked_layer_loss = nn.functional.cross_entropy(masked_layer_logits.view(-1, len(self.tokenizer), masked_layer_labels.view(-1)))

        last_loss, last_hidden = self.secondLM(input_embeds = first_hidden[-2], labels=lm_labels)
        return first_loss + last_loss + masked_layer_loss
