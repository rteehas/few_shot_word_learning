import torch
from torch import nn
import torch.nn.functional as F
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

        w = self.secondLM.roberta.embeddings.word_embeddings.weight.clone()
        msk = torch.nn.functional.one_hot(torch.LongTensor([self.nonces[nonce][1]]),
                                          num_classes=len(self.autoregressiveTokenizer))

        masked = w * (1 - msk).T
        new_w = masked + (first_lm_embed * msk.T)

        input_embeds = F.embedding_bag(new_w, second_input["input_ids"])
        if nonce not in seq:
            second_out = self.secondLM(**second_input, labels=second_labels, output_hidden_states=True)
        else:
            second_out = self.secondLM(inputs_embeds=input_embeds, attention_mask=second_input["attention_mask"], labels=second_labels, output_hidden_states=True)

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
        self.lin = nn.Linear(768, 768, bias=False)
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
        print("first loss", first_out.loss)
        first_hidden = first_out.hidden_states

        if nonce in seq:
            nonce_tok = nonce
            span = self.morph_model.calc_nonce_span(nonce, seq_enc, seq)

            combined = combine_layers(first_hidden, self.layers).unsqueeze(0)
            nonce_embeds_first = self.morph_model.extract_nonce_embeds(span, combined)
            tgt_enc = self.mlmTokenizer(nonce_tok, return_tensors="pt", padding="max_length",
                                        max_length=inputs["input_ids"].size(1)).to(device)  # batch size 1

            msk = self.morph_model.mask_nonce((0, span[1] - span[0]), tgt_enc)
            tgt_pad = tgt_enc["attention_mask"] > 0
            tgt_pad = ~tgt_pad

            seq_out = self.morph_model(nonce_embeds_first,
                                       tgt_enc["input_ids"],
                                       msk.view(inputs["input_ids"].size(1), 1),
                                       tgt_pad.view(inputs["input_ids"].size(1), 1))

            nonce_morph_embed = seq_out.squeeze(0)[1, :]
            nonce_idx = get_word_idx(seq, nonce)
            first_lm_embed = extract_word_embeddings(first_hidden, self.layers, nonce_idx, seq_enc)
            nonce_ind = self.nonces[nonce_tok][1]
            w = self.secondLM.transformer.wte.weight.clone()
            msk = torch.nn.functional.one_hot(torch.LongTensor([self.nonces[nonce][1]]),
                                              num_classes=len(self.autoregressiveTokenizer))

            masked = w * (1 - msk).T
            new_w = masked + (first_lm_embed * msk.T)

            input_embeds = F.embedding_bag(new_w, second_input["input_ids"])

        if nonce in seq:

            second_out = self.secondLM(inputs_embeds=input_embeds,  # second_input["input_ids"],
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
        # taken from modeling_gpt2.py
        second_hidden = second_out.hidden_states
        logits = F.linear(second_hidden[-1], new_w)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = second_labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        hiddens = (first_hidden, second_hidden)

        return loss, hiddens, logits

    def generate(self, input_seq, max_len, num_beams):
        # fix generate
        input_ids = self.autoregressiveTokenizer.encode(input_seq, add_special_tokens=True,
                                                        return_tensors='pt').to(device)

        beam_outputs = self.secondLM.generate(
            input_ids,
            max_length=max_len,
            num_beams=num_beams,
            no_repeat_ngram_size=2,
            num_return_sequences=3,
            early_stopping=True
            # pad_token_id = tokenizer.pad_token_id
        )
        return beam_outputs
