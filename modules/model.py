import torch
from torch import nn
from transformers import RobertaForMaskedLM
from prototype_memory import OnlineProtoNet
from mlm import AttnSeq2Seq
from utils import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MorphMemoryModel(nn.Module):

    def __init__(self, tokenizer, nonces, device, layers):
        super(MorphMemoryModel, self).__init__()

        self.tokenizer = tokenizer

        self.firstLM = RobertaForMaskedLM.from_pretrained('roberta-base').to(device)
        self.secondLM = RobertaForMaskedLM.from_pretrained('roberta-base').to(device)
        self.firstLM.resize_token_embeddings(len(self.tokenizer))
        self.secondLM.resize_token_embeddings(len(self.tokenizer))
        self.secondLM.roberta.embeddings = self.firstLM.roberta.embeddings  # share weights

        self.memory = OnlineProtoNet()
        self.morph_model = AttnSeq2Seq(self.tokenizer, 768).to(device)
        self.mask_attn_linear = nn.Linear(768, len(self.tokenizer))  # weight shape = (len(tokenizer), 768)

        self.nonces = {}  # for referencing embedding location
        for i, val in enumerate(reversed(nonces)):
            self.nonces[val] = -(1 + i)

        self.layers = layers

    def forward(self, seq, second_seq, input, second_input, second_labels, nonce):
        seq_enc = self.tokenizer(seq, return_tensors="pt").to(device)

        first_out = self.firstLM(**input, output_hidden_states=True)

        first_hidden = first_out.hidden_states
        # print(first_hidden[-2].shape)

        nonce_tok = "<{}_nonce>".format(nonce)
        # seq_to_seq_tgt = "<s> {} </s>".format(nonce_tok)
        # print(seq_enc)
        span = self.morph_model.calc_nonce_span(nonce, seq_enc, seq)
        # print(nonce, span, seq)

        combined = combine_layers(first_hidden, self.layers).unsqueeze(0)
        # print(combined.shape)
        nonce_embeds_first = self.morph_model.extract_nonce_embeds(span, combined)
        tgt_enc = self.tokenizer(nonce_tok, return_tensors="pt", padding="max_length",
                                 max_length=input["input_ids"].size(1)).to(device)  # batch size 1

        msk = self.morph_model.mask_nonce((0, span[1] - span[0]), tgt_enc)
        tgt_pad = tgt_enc["attention_mask"] > 0
        tgt_pad = ~tgt_pad

        seq_out = self.morph_model(nonce_embeds_first,
                                   tgt_enc["input_ids"],
                                   msk.view(input["input_ids"].size(1), 1),
                                   tgt_pad.view(input["input_ids"].size(1), 1))

        seq_lin = self.morph_model.linear(seq_out)

        nonce_morph_embed = self.morph_model.embedding(tgt_enc["input_ids"][:, 1])
        nonce_idx = get_word_idx(seq, nonce)
        first_lm_embed = extract_word_embeddings(first_hidden, self.layers, nonce_idx, seq_enc)

        nonce_ind = self.nonces[nonce_tok]

        self.secondLM.roberta.embeddings.word_embeddings.weight.data[nonce_ind, :] = first_lm_embed

        second_out = self.secondLM(**second_input, labels=second_labels, output_hidden_states=True)

        second_loss = second_out.loss
        second_hidden = second_out.hidden_states

        hiddens = (first_hidden, second_hidden)

        return second_loss, hiddens, second_out.logits


class OnlineProtoNet(nn.Module):

    def __init__(self):
        super(OnlineProtoNet, self).__init__()
        self.memory = {}

    def store(self, word, embed):
        if word not in self.memory:
            self.memory[word] = (embed, 1)
        else:
            self.memory[word] = (torch.div(embed + self.memory[word][0], self.memory[word][1] + 1), self.memory[word][1] + 1)

    def retrieve(self, word):  # retrieves embed after linear layer to fix dims if needed
        return self.memory[word][0]


def init_new_dim(size, method="random"):
    if method == "zeros":
        return torch.zeros(size)
    elif method == "random":
        return torch.rand(size)

