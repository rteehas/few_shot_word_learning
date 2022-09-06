import torch
from torch import nn
from functools import reduce


# class BERTWrapper(nn.Module):

#     def __init__(self, model, tokenizer):
#         super(BERTWrapper, self).__init__()
#         # assumes bert-style model (i.e. BERT or RoBERTa)
#         self.model = model
#         self.tokenizer = tokenizer

#     def forward(self, inputs, nonce, **kwargs):
#         # if kwargs:
#         #     if kwargs["memory_aug"]:
#         labels = inputs["input_ids"]
#         # print(inputs.input_ids.size(),inputs.input_ids.shape)
#         # ids = inputs.input_ids.unsqueeze(0)
#         # attn_mask = inputs.attention_mask.unsqueeze(0)
#         # print(ids.size(), ids.shape)
#         # print(attn_mask.size())

#         # input_embeds = inputs["input_embeds"]
#         # labels[labels != self.tokenizer.mask_token_id] = -100

#         outputs = self.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, output_hidden_states=True)

#             # else:
#             #     labels = inputs["input_ids"].clone()
#             #     labels[labels != self.tokenizer.mask_token_id] = -100

#             #     outputs = self.model(input_embeds=inputs["input_ids"],
#             #                               labels=labels,
#             #                               attn_mask=inputs["attention_mask"],
#             #                               output_hidden_states=True)

#         return outputs

class MaskedAttention(nn.Module):

    def __init__(self, tokenizer, embed_dim):
        super(MaskedAttention, self).__init__()
        self.tokenizer = tokenizer
        self.morph = nn.TransformerEncoderLayer(embed_dim, 1)
        self.d = embed_dim

    def forward(self, input, encoded, seq, nonce):
        span = self.calc_nonce_span(nonce, encoded, seq)
        mask = self.mask_nonce(span, encoded)
        tmp_input = input.clone()
        msk_dim = mask.size()[1]
        # tmp_input = tmp_input * mask
        return self.morph(tmp_input, src_mask=torch.reshape(mask, (msk_dim, 1, 1)))

    # def calculate_spans(self, encoded):
    #     # taken from https://github.com/huggingface/tokenizers/issues/447
    #     desired_output = []
    #     for word_id in encoded.words():
    #         if word_id is not None:
    #             start, end = encoded.word_to_tokens(word_id)
    #             word = self.tokenizer.decode(encoded[start:end])
    #             new_tok_name = word + "_merged"
    #             desired_output.append((start, end, new_tok_name))

    #     return desired_output

    def calc_nonce_span(self, nonce, encoded, seq):
        for word_id in encoded.words():
            if word_id is not None:
                start, end = encoded.word_to_tokens(word_id)
                s, e = encoded.word_to_chars(word_id)
                # print(seq[s:e+1])
                if seq[s:e] == nonce:
                    return (start, end)

    # def calculate_masks(self, spans):
    #     broken_words = filter(lambda tup: abs(tup[1] - tup[0] > 1), spans)
    #     masks = []
    #     for span in broken_words:
    #         maskless = torch.full_like(len(spans), False, dtype=torch.bool)
    #         masked = maskless.index_fill(int(0), [i for i in range(span[0], span[1])], True)
    #         masks.append((masked, span[2]))
    #     return masks

    def mask_nonce(self, span, encoded):
        maskless = torch.full_like(enc.input_ids, False, dtype=torch.bool)
        for i in range(span[0], span[1]):
            maskless[0][i] = True
        # masked = maskless.index_fill(0, torch.Tensor([i for i in range(span[0], span[1])]), 1)
        return maskless
