import torch
from torch import nn
from transformers import RobertaForMaskedLM
from prototype_memory import OnlineProtoNet
from mlm import AttnSeq2Seq

class MorphMemoryModel(nn.Module):

    def __init__(self, tokenizer, nonces, device):
        super(MorphMemoryModel, self).__init__()
        # self.model = model
        # if self.model_name == "roberta":
        #     self.config = RobertaConfig
        #     self
        self.device = device
        self.tokenizer = tokenizer
        # self.second_tokenizer = copy.deepcopy(self.tokenizer)
        self.firstLM = RobertaForMaskedLM.from_pretrained('roberta-base').to(device)
        self.secondLM = RobertaForMaskedLM.from_pretrained('roberta-base').to(device)
        self.firstLM.resize_token_embeddings(len(self.tokenizer))
        self.secondLM.resize_token_embeddings(len(self.tokenizer))
        self.secondLM.roberta.embeddings = self.firstLM.roberta.embeddings  # share weights

        self.memory = OnlineProtoNet()
        self.morph_model = AttnSeq2Seq(self.tokenizer, 768).to(device)
        self.mask_attn_linear = nn.Linear(768, len(self.tokenizer))  # weight shape = (len(tokenizer), 768)
        # self.morph_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(768, 1),1)
        # self.l1 = nn.Linear(self.morph_model.d + self.firstLM.model)

        self.nonces = {}  # for referencing embedding location
        for i, val in enumerate(reversed(nonces)):
            self.nonces[val] = -(1 + i)

    def forward(self, seq, second_seq, input, second_input, seq_labels, second_labels, nonce):
        seq_enc = self.tokenizer(seq, return_tensors="pt").to(self.device)
        lm_labels = seq_labels
        # lm_labels[lm_labels != self.tokenizer.mask_token_id] = -100
        # print(lm_labels)
        first_out = self.firstLM(**input, labels=lm_labels, output_hidden_states=True)
        first_loss = first_out.loss
        # print(first_loss)
        first_hidden = first_out.hidden_states

        nonce_tok = "<{}_nonce>".format(nonce)
        # seq_to_seq_tgt = "<s> {} </s>".format(nonce_tok)
        # print(seq_enc)
        span = self.morph_model.calc_nonce_span(nonce, seq_enc, seq)
        # print(nonce, span, seq)
        nonce_embeds_first = self.morph_model.extract_nonce_embeds(span, first_hidden[-2])
        tgt_enc = self.tokenizer(nonce_tok, return_tensors="pt", padding="max_length",
                                 max_length=input["input_ids"].size(1)).to(self.device)  # batch size 1
        # print(tgt_enc)
        # print(input["input_ids"].size())
        msk = self.morph_model.mask_nonce((0, span[1] - span[0]), tgt_enc)
        tgt_pad = tgt_enc["attention_mask"] > 0
        tgt_pad = ~tgt_pad

        seq_out = self.morph_model(nonce_embeds_first,
                                   tgt_enc["input_ids"],
                                   msk.view(input["input_ids"].size(1), 1),
                                   tgt_pad.view(input["input_ids"].size(1), 1))

        seq_lin = self.morph_model.linear(seq_out)
        ce = nn.CrossEntropyLoss(ignore_index=1)
        mask_seq_loss = ce(seq_lin.view(-1, len(self.tokenizer)), tgt_enc["input_ids"].view(-1))

        nonce_morph_embed = self.morph_model.embedding(tgt_enc["input_ids"][:, 1])
        # print(nonce_morph_embed)
        # nonce_first_lm_embed = first_hidden[-2]
        # nonce_joint_embed =
        # masked_layer_out = masked_layer(first_hidden[-2], input, seq, nonce)
        # tgt = torch.roll(input.input_ids, 1)
        self.memory.store(nonce_tok, nonce_morph_embed)
        nonce_ind = self.nonces[nonce_tok]

        # with torch.no_grad():
        self.secondLM.roberta.embeddings.word_embeddings.weight.data[nonce_ind, :] = self.memory.retrieve(
            nonce_tok).clone()

        # self.second_tokenizer.add_special_tokens({"nonce": "<{}_nonce".format(nonce)})
        # self.secondLM.resize_token_embeddings(len(self.second_tokenizer))
        # self.resize_masked_attn_linear("random")
        # embs = self.mask_attn_linear(masked_layer_out)
        # msk = masked_layer.mask_nonce(masked_layer.calc_nonce_span(nonce, inputs, seq))
        # masked_layer_labels = input.input_ids * msk
        # masked_layer_labels[masked_layer_labels==0] = -100
        # masked_layer_logits = nn.functional.log_softmax(embs)
        # masked_layer_loss = nn.functional.cross_entropy(logits.view(-1, len(tokenizer), masked_layer_labels.view(-1)))
        # second_seq = seq.replace(nonce, nonce_tok)

        # second_labels = self.tokenizer(second_seq, return_tensors="pt")["input_ids"].to(device)
        # second_labels[second_labels != self.tokenizer.mask_token_id] = -100
        second_out = self.secondLM(**second_input, labels=second_labels, output_hidden_states=True)

        second_loss = second_out.loss
        second_hidden = second_out.hidden_states

        losses = (first_loss, mask_seq_loss, second_loss)
        hiddens = (first_hidden, second_hidden)
        # last_loss, last_hidden = self.secondLM(, )
        # return first_loss + last_loss
        return losses, hiddens
