import torch
from torch import nn
from functools import reduce


class AttnSeq2Seq(nn.Module):
    def __init__(self, tokenizer, dim):
        super(AttnSeq2Seq, self).__init__()
        self.embedding = nn.Embedding(len(tokenizer), dim)
        # self.pos_encoder = PositionalEncoding(INPUT_DIM, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=1)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        self.linear = nn.Linear(dim, len(tokenizer))
        self.softmax = nn.Softmax(dim=-1)
        self.d = dim

    def get_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def calc_nonce_span(self, nonce, encoded, seq):
        for word_id in encoded.words():
            if word_id is not None:
                start, end = encoded.word_to_tokens(word_id)
                s, e = encoded.word_to_chars(word_id)
                # print(seq[s:e+1])
                if seq[s:e] == nonce:
                    return (start, end)

    def mask_nonce(self, span, encoded):
        maskless = torch.full_like(encoded.input_ids, False, dtype=torch.bool)
        for i in range(span[0], span[1]):
            maskless[0][i] = True
        # masked = maskless.index_fill(0, torch.Tensor([i for i in range(span[0], span[1])]), 1)
        return ~maskless

    def extract_nonce_embeds(self, span, embeds):
        new_embeds = torch.zeros_like(embeds)
        diff = span[1] - span[0]
        new_embeds[:, :diff] = embeds[:, span[0]:span[1]]  # assumes embeds shape [Batch, Seq, Embed Dim]
        return new_embeds

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        # src = self.embedding(src)
        # src = self.pos_encoder(src)
        src = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)

        tgt_mask = self.get_mask(tgt.size(0)).to(device)
        tgt = self.embedding(tgt)
        # tgt = self.pos_encoder(tgt)

        output = self.transformer_decoder(
            tgt=tgt,
            memory=src,
            tgt_mask=tgt_mask,  # to avoid looking at the future tokens (the ones on the right)
            tgt_key_padding_mask=tgt_key_padding_mask,  # to avoid working on padding
            memory_key_padding_mask=src_key_padding_mask  # avoid looking on padding of the src
        )

        # output = self.linear(output)
        return output