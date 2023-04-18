import torch
from torch import nn
import torch.nn.functional as F
from transformers import RobertaForMaskedLM, AutoModelForCausalLM
from prototype_memory import OnlineProtoNet
from mlm import AttnSeq2Seq
from utils import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class OnlineProtoNet(nn.Module):

    def __init__(self, agg="mean"):
        super(OnlineProtoNet, self).__init__()
        self.agg_method = agg
        self.memory = {}
        if agg == "CLS":
            input_size = 768
            nhead = 8
            num_layers = 2
            hidden_size = 512
            num_positions = 25
            self.agg = TransformerSummarizerWithCLS(input_size, nhead, num_layers, hidden_size, num_positions)
        elif agg == "RNN":
            self.agg = BiGRUSummarizer(768, 384)

    def store(self, word, embed):
        if word not in self.memory:
            self.memory[word] = [embed]
        else:
            #             print(embed.shape, self.memory[word][0].shape, self.memory[word][0].requires_grad)
            self.memory[word].append(embed)

    def retrieve(self, word):  # retrieves embed after linear layer to fix dims if needed
        ctx = torch.cat(self.memory[word], dim=0)
        if self.agg_method == "mean":

            #             print(ctx.requires_grad, "ctx")
            return torch.mean(ctx, dim=0)
        else:
            return self.agg(ctx.unsqueeze(0))


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


class MorphMemoryModel(nn.Module):

    def __init__(self, tokenizer, nonces, device, layers, agg="mean"):
        super(MorphMemoryModel, self).__init__()

        self.tokenizer = tokenizer

        self.firstLM = RobertaForMaskedLM.from_pretrained('roberta-base').to(device)
        self.secondLM = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-roberta-base').to(device)
        #         self.firstLM.resize_token_embeddings(len(self.tokenizer))
        #         self.secondLM.resize_token_embeddings(len(self.tokenizer))
        self.freeze_roberta()

        #         self.secondLM.roberta.embeddings.word_embeddings = self.firstLM.roberta.embeddings.word_embeddings  # share weights

        self.memory = OnlineProtoNet(agg=agg)
        self.morph_model = AttnSeq2Seq(self.tokenizer, 768).to(device)
        #         self.mask_attn_linear = nn.Linear(768, len(self.tokenizer))  # weight shape = (len(tokenizer), 768)
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=1)
        self.lin = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.lin2 = nn.Linear(768, 768, bias=False)
        self.nonces = {}  # for referencing embedding location
        for nonce in nonces:
            nonce_tok_id = self.tokenizer.convert_tokens_to_ids(nonce)
            self.nonces[nonce] = nonce_tok_id

        #         initialize with mean embedding
        #         indices = self.get_zero_grads_idx()
        #         m = torch.mean(self.secondLM.roberta.embeddings.word_embeddings.weight[indices, :], dim=0)
        #         for nonce in nonces:
        #             with torch.no_grad():
        #                 self.secondLM.roberta.embeddings.word_embeddings.weight[self.nonces[nonce],:] = m
        #                 self.firstLM.roberta.embeddings.word_embeddings.weight[self.nonces[nonce],:] = m

        #         with torch.no_grad():
        #             self.morph_model.embedding.weight = self.secondLM.roberta.embeddings.word_embeddings.weight
        #             self.morph_model.embedding.weight = torch.nn.parameter.Parameter(model.secondLM.roberta.embeddings.word_embeddings.weight.clone())

        self.layers = layers
        self.morph_decoder = nn.Linear(768, len(self.tokenizer), bias=False)

    def freeze_roberta(self):
        for parameter in self.firstLM.parameters():
            parameter.requires_grad = False

        for parameter in self.secondLM.parameters():
            parameter.requires_grad = False

    #         self.firstLM.roberta.embeddings.word_embeddings.weight.requires_grad = True
    #         self.secondLM.roberta.embeddings.word_embeddings.weight.requires_grad = True
    def build_memory(self, seq, inputs, nonce):
        seq_enc = self.tokenizer(seq, return_tensors="pt").to(device)

        first_out = self.firstLM(**inputs, output_hidden_states=True)
        first_hidden = first_out.hidden_states

        combined = combine_layers(first_hidden, self.layers).unsqueeze(0)

        if nonce != "":
            nonce_tok = nonce
            # seq_to_seq_tgt = "<s> {} </s>".format(nonce_tok)
            # print(seq_enc)
            if nonce in seq:
                span = self.morph_model.calc_nonce_span(nonce, seq_enc, seq)

                selects = (inputs["input_ids"] == self.nonces[nonce]).nonzero(as_tuple=True)
                #                 print(selects, selects[0])
                first_lm_embed = combined[:, selects[0], :]
                #                 print(first_lm_embed.shape)
                first_lm_embed = first_lm_embed.sum(1)
                #                 print(first_lm_embed.shape)
                self.memory.store(nonce, self.lin(first_lm_embed).clone().reshape((1, 768)))

                #             print(first_lm_embed.shape)

    #                 input_embeds = torch.cat(a, dim=0).unsqueeze(0)

    def forward(self, seq, second_seq, inputs, second_input, label, nonce):
        seq_enc = self.tokenizer(seq, return_tensors="pt").to(device)
        #         print(self.firstLM(**inputs, output_hidden_states=True))
        first_out = self.firstLM(**inputs, output_hidden_states=True)
        #         print("first loss", first_out.loss)
        first_hidden = first_out.hidden_states
        combined = combine_layers(first_hidden, self.layers).unsqueeze(0)
        #         print(combined.shape)
        #         print(combined.shape)
        # print(first_hidden[-2].shape)
        if nonce != "":
            nonce_tok = nonce
            # seq_to_seq_tgt = "<s> {} </s>".format(nonce_tok)
            # print(seq_enc)
            if nonce in seq:
                span = self.morph_model.calc_nonce_span(nonce, seq_enc, seq)

                selects = (inputs["input_ids"] == self.nonces[nonce]).nonzero(as_tuple=True)
                #                 print(selects, selects[0])
                first_lm_embed = combined[:, selects[0], :]
                #                 print(first_lm_embed.shape)
                first_lm_embed = first_lm_embed.sum(1)

                self.memory.store(nonce, self.lin(first_lm_embed).clone().reshape((1, 768)))

                nonce_token_id = self.nonces[nonce]

                # Create a boolean mask indicating where the input token ids match the nonce token id
                nonce_mask = (second_input["input_ids"] == nonce_token_id)

                # Get embeddings from the secondLM model (using dynamic attribute access if necessary)

                all_embeddings = self.secondLM.roberta.embeddings.word_embeddings(second_input["input_ids"])

                # Use the mask to replace embeddings corresponding to the nonce with the memory retrieval
                retrieved_nonce_embed = self.memory.retrieve(nonce).expand_as(all_embeddings)
                input_embeds = torch.where(nonce_mask.unsqueeze(-1), retrieved_nonce_embed, all_embeddings)


        else:
            input_embeds = self.lin(combined)

        if nonce not in seq and nonce != "":
            raise Exception("Nonce Not in Sequence")
        else:
            second_out = self.secondLM(inputs_embeds=input_embeds, attention_mask=second_input["attention_mask"],
                                       labels=label, output_hidden_states=True)

        second_loss = second_out.loss
        #         print(second_loss)
        second_hidden = second_out.hidden_states

        #         hiddens = (first_hidden, second_hidden)
        #         hiddens = []
        #         print(second_loss, hiddens)
        return second_loss, second_out

    def get_zero_grads_idx(self):
        index_grads_to_zero = torch.empty(len(self.tokenizer), dtype=torch.bool).fill_(True)
        for nonce in self.nonces:
            idx = self.nonces[nonce]
            index_grads_to_zero = index_grads_to_zero & (torch.arange(len(self.tokenizer)) != idx)
        return index_grads_to_zero

    def initialize_with_def(self, nonce, definition):
        with torch.no_grad():
            toks = self.tokenizer(definition, return_tensors="pt").to(device)
            outs = self.firstLM(**toks, output_hidden_states=True)

class BiGRUSummarizer(nn.Module):
    # batch, seq, hidden
    def __init__(self, input_size, hidden_size):
        super(BiGRUSummarizer, self).__init__()
        self.bigru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, token_embeddings):
        _, hidden = self.bigru(token_embeddings)
        fwd_hidden = hidden[0:1]  # Forward hidden state
        bwd_hidden = hidden[1:2]  # Backward hidden state
        output_embedding = torch.cat((fwd_hidden, bwd_hidden), dim=2).squeeze(0)
        return output_embedding

class TransformerSummarizerWithCLS(nn.Module):
    def __init__(self, input_size, nhead, num_layers, hidden_size, num_positions):
        super(TransformerSummarizerWithCLS, self).__init__()
        self.embedding = nn.Embedding(num_positions, input_size)
        self.cls_embedding = nn.Parameter(torch.randn(1, 1, input_size))
        self.pos_encoder = self.positional_encoding(input_size, num_positions + 1)  # +1 for the [CLS] token
        encoder_layers = TransformerEncoderLayer(input_size, nhead, hidden_size)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

    def forward(self, token_embeddings):
        batch_size = token_embeddings.shape[0]
        cls_token = self.cls_embedding.expand(batch_size, -1, -1)  # Create [CLS] tokens for each item in the batch
        token_embeddings = torch.cat([cls_token, token_embeddings], dim=1)  # Prepend [CLS] tokens to the input

        seq_len = token_embeddings.shape[1]
        pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        pos_embeddings = self.embedding(pos)
        token_embeddings = token_embeddings + pos_embeddings

        token_embeddings = token_embeddings.transpose(0, 1)  # TransformerEncoder expects (S, N, E) format
        output = self.transformer_encoder(token_embeddings)
        cls_embedding = output[0]  # [CLS] embedding is at position 0
        return cls_embedding

    @staticmethod
    def positional_encoding(input_size, num_positions):
        pe = torch.zeros(num_positions, input_size)
        position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / input_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return nn.Parameter(pe, requires_grad=False)


class MetaLearner(nn.Module):
    def __init__(self, base_model):

        self.base_model = base_model

    def forward_example(self, inputs, nonce):
        nonce_token_id = self.base_model.nonces[nonce]
        nonce_mask = (inputs["input_ids"] == nonce_token_id)
        all_embeddings = self.base_model.secondLM.roberta.embeddings.word_embeddings(inputs["input_ids"])
        retrieved_nonce_embed = self.base_model.memory.retrieve(nonce).expand_as(all_embeddings)
        input_embeds = torch.where(nonce_mask.unsqueeze(-1), retrieved_nonce_embed, all_embeddings)
        second_out = self.base_model.secondLM(inputs_embeds=input_embeds, attention_mask=inputs["attention_mask"])
        return second_out

    def forward_task(self, seq, second_seq, inputs, second_input, label, nonce):
        return self.base_model.forward(seq, second_seq, inputs, second_input, label, nonce)