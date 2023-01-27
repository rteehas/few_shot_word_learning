import torch
from torch import nn
from transformers import RobertaForMaskedLM, AutoModelForCausalLM
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

    def initialize_with_def(self, nonce, definition):
        with torch.no_grad():
            toks = self.tokenizer(definition, return_tensors="pt").to(device)
            outs = self.firstLM(**toks, output_hidden_states=True)



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


class MorphMemoryModelAutoregressive(nn.Module):
    def __init__(self, mlmTokenizer, autoregressiveTokenizer, nonces, device, layers):
        super(MorphMemoryModelAutoregressive, self).__init__()

        self.mlmTokenizer = mlmTokenizer
        self.autoregressiveTokenizer = autoregressiveTokenizer

        self.firstLM = RobertaForMaskedLM.from_pretrained('roberta-base').to(device)
        self.secondLM = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        self.firstLM.resize_token_embeddings(len(self.mlmTokenizer))
        self.secondLM.resize_token_embeddings(len(self.autoregressiveTokenizer))
        # self.secondLM.roberta.embeddings = self.firstLM.roberta.embeddings  # share weights

        #         self.memory = OnlineProtoNet()
        self.morph_model = AttnSeq2Seq(self.mlmTokenizer, 768).to(device)
        self.mask_attn_linear = nn.Linear(768, len(self.mlmTokenizer))  # weight shape = (len(tokenizer), 768)

        self.nonces = {}  # for referencing embedding location
        for nonce in nonces:
            nonce_tok_id = (
            self.mlmTokenizer.convert_tokens_to_ids(nonce), self.autoregressiveTokenizer.convert_tokens_to_ids(nonce))
            self.nonces[nonce] = nonce_tok_id

        # initialize with mean embedding
        indices = self.get_zero_grads_idx()
        m = torch.mean(self.firstLM.roberta.embeddings.word_embeddings.weight[indices[0], :], dim=0)
        for nonce in nonces:
            with torch.no_grad():
                self.firstLM.roberta.embeddings.word_embeddings.weight[self.nonces[nonce][0], :] = m

        m = torch.mean(self.secondLM.transformer.wte.weight[indices[1],], dim=0)
        for nonce in nonces:
            with torch.no_grad():
                self.secondLM.transformer.wte.weight[self.nonces[nonce][1], :] = m

        self.layers = layers

    def freeze_roberta(self):
        for parameter in self.firstLM.parameters():
            parameter.requires_grad = False

        self.firstLM.roberta.embeddings.word_embeddings.weight.requires_grad = True

    def freeze_gpt(self):
        for parameter in self.secondLM.parameters():
            parameter.requires_grad = False

        for parameter in self.secondLM.lm_head.parameters():
            parameter.requires_grad = True

    def get_zero_grads_idx(self):
        index_grads_to_zero1 = torch.empty(len(self.mlmTokenizer), dtype=torch.bool).fill_(True)
        for nonce in self.nonces:
            idx = self.nonces[nonce][0]
            index_grads_to_zero1 = index_grads_to_zero1 & (torch.arange(len(self.mlmTokenizer)) != idx)

        index_grads_to_zero2 = torch.empty(len(self.autoregressiveTokenizer), dtype=torch.bool).fill_(True)
        for nonce in self.nonces:
            idx = self.nonces[nonce][1]
            index_grads_to_zero2 = index_grads_to_zero2 & (torch.arange(len(self.autoregressiveTokenizer)) != idx)

        return [index_grads_to_zero1, index_grads_to_zero2]

    def forward(self, seq, second_seq, inputs, second_input, second_labels, nonce):
        seq_enc = self.mlmTokenizer(seq, return_tensors="pt").to(device)

        first_out = self.firstLM(**inputs, output_hidden_states=True)

        first_hidden = first_out.hidden_states
        # print(first_hidden[-2].shape)

        #         print("here", second_input["input_ids"])
        #         print("here",second_labels)
        #         print("here",second_input["attention_mask"])

        if nonce in seq:
            nonce_tok = nonce
            # seq_to_seq_tgt = "<s> {} </s>".format(nonce_tok)
            # print(seq_enc)
            span = self.morph_model.calc_nonce_span(nonce, seq_enc, seq)
            # print(nonce, span, seq)

            combined = combine_layers(first_hidden, self.layers).unsqueeze(0)
            # print(combined.shape)
            nonce_embeds_first = self.morph_model.extract_nonce_embeds(span, combined)
            tgt_enc = self.mlmTokenizer(nonce_tok, return_tensors="pt", padding="max_length",
                                        max_length=inputs["input_ids"].size(1)).to(device)  # batch size 1

            msk = self.morph_model.mask_nonce((0, span[1] - span[0]), tgt_enc)
            tgt_pad = tgt_enc["attention_mask"] > 0
            tgt_pad = ~tgt_pad
            #             print("here2", second_input["input_ids"])
            #             print("here2",second_labels)
            #             print("here2",second_input["attention_mask"])
            seq_out = self.morph_model(nonce_embeds_first,
                                       tgt_enc["input_ids"],
                                       msk.view(inputs["input_ids"].size(1), 1),
                                       tgt_pad.view(inputs["input_ids"].size(1), 1))

            #         seq_lin = self.morph_model.linear(seq_out)
            #             print("here3", second_input["input_ids"])
            #             print("here3",second_labels)
            #             print("here3",second_input["attention_mask"])
            nonce_morph_embed = seq_out.squeeze(0)[1, :]
            nonce_morph_embed = self.morph_model.embedding(tgt_enc["input_ids"][:, 1])
            nonce_idx = get_word_idx(seq, nonce)
            #             print("here")
            first_lm_embed = extract_word_embeddings(first_hidden, self.layers, nonce_idx, seq_enc)
            #             print(first_lm_embed)
            nonce_ind = self.nonces[nonce_tok][1]
            #             self.secondLM.transformer.wte.weight[nonce_ind,:] = self.secondLM.transformer.wte.weight[nonce_ind,:] + first_lm_embed
            #             cl = self.secondLM.transformer.wte.weight.clone()
            #             cl[nonce_ind, :] = first_lm_embed
            #             print(cl)
            #             print(self.secondLM.transformer.wte.weight)
            #             print(self.firstLM.roberta.embeddings.word_embeddings.weight)
            # https://github.com/huggingface/transformers/issues/1458
            #         print(second_input["input_ids"])
            #         print(second_labels)
            #         print(second_input["attention_mask"])
            # todo, make this less hacky, update the embedding matrix with no grad
            a = []
            for s in second_inputs["input_ids"][0]:
                if s != 50257:
                    a.append(model.secondLM.transformer.wte(s).reshape((1, 768)))
                else:
                    a.append(first_lm_embed.reshape((1, 768)))
            #             print(first_lm_embed.shape)
            input_embeds = torch.cat(a, dim=0)
        if nonce in seq:

            second_out = self.secondLM(inputs_embeds=input_embeds.unsqueeze(0),  # second_input["input_ids"],
                                       labels=second_labels,
                                       attention_mask=second_input["attention_mask"],
                                       token_type_ids=None,
                                       output_hidden_states=True)
        else:
            second_out = self.secondLM(second_input["input_ids"],
                                       labels=second_labels,
                                       attention_mask=second_input["attention_mask"],
                                       token_type_ids=None,
                                       output_hidden_states=True)

        second_loss = second_out.loss
        second_hidden = second_out.hidden_states

        hiddens = (first_hidden, second_hidden)

        return second_loss, hiddens, second_out.logits

    def generate(self, input_seq):
        input_ids = self.autoregressiveTokenizer.encode(input_sequence, add_special_tokens=True,
                                                        return_tensors='pt').to(device)

        beam_outputs = self.secondLM.generate(
            input_ids,
            max_length=50,
            num_beams=5,
            no_repeat_ngram_size=2,
            num_return_sequences=3,
            early_stopping=True
            # pad_token_id = tokenizer.pad_token_id
        )
        return beam_outputs