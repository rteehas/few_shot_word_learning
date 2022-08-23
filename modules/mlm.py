import torch
from torch import nn


class BERTWrapper(nn.Module):

    def __init__(self, model, tokenizer):
        # assumes bert-style model (i.e. BERT or RoBERTa)
        self.model = model
        self.tokenizer = tokenizer
        pass

    def forward(self, inputs, attn_mask, **kwargs):
        if kwargs:
            if kwargs["memory_aug"]:
                labels = inputs["input_ids"]
                input_embeds = inputs["input_embeds"]
                labels[labels != self.tokenizer.mask_token_id] = -100

                outputs = self.model(input_embeds=input_embeds,
                           labels=labels,
                           attn_mask=attn_mask,
                           output_hidden_states=True)

            else:
                labels = inputs["input_ids"].clone()
                labels[labels != self.tokenizer.mask_token_id] = -100

                outputs = self.model(input_embeds=inputs["input_ids"],
                                          labels=labels,
                                          attn_mask=attn_mask,
                                          output_hidden_states=True)

        return outputs.loss, outputs.hidden_states

