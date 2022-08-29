import torch
from torch import nn
from functools import reduce



class BERTWrapper(nn.Module):

    def __init__(self, model, tokenizer):
        super(BERTWrapper, self).__init__()
        # assumes bert-style model (i.e. BERT or RoBERTa)
        self.model = model
        self.tokenizer = tokenizer

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

class MaskedAttention(nn.Module):

    def __init__(self, tokenizer, num_words, embed_dim):
        self.tokenizer = tokenizer
        self.morph = nn.TransformerEncoderLayer(embed_dim, 1)

    def forward(self, input, encoded):
        spans = self.calculate_spans(encoded)
        masks = self.calculate_masks(spans)
        out = []
        for mask in masks:
            tmp_input = input.clone()
            outputs = self.morph(tmp_input, src_mask = mask)
            out.append(outputs)
        return out

    def calculate_spans(self, encoded):
        # taken from https://github.com/huggingface/tokenizers/issues/447
        desired_output = []
        for word_id in encoded.words():
            if word_id is not None:
                start, end = encoded.word_to_tokens(word_id)
                desired_output.append((start, end))

        return desired_output

    def calculate_masks(self, spans):
        broken_words = filter(lambda tup: abs(tup[1] - tup[0] > 1), spans)
        masks = []
        for span in broken_words:
            maskless = torch.full_like(len(spans), False, dtype=torch.bool)
            masked = maskless.index_fill(0, [i for i in range(span[0], span[1])], True)
            masks.append(masked)
        return masks
