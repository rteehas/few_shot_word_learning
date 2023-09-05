import json
import argparse
import glob
import logging
import os
import sys
import random
import shutil
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import default_collate
# from tensorboardX import SummaryWriter
import re
from torch import nn
from tqdm import tqdm, trange
import wandb
from accelerate import Accelerator
from torch.utils.data import Dataset
from transformers import AutoTokenizer, RobertaForMultipleChoice, RobertaForMaskedLM
import logging
# from
from transformers import AdamW, get_linear_schedule_with_warmup
from nltk.corpus import wordnet as wn

from configs.config import AggregatorConfig
from modules.memory import OnlineProtoNet
from modules.model import MorphMemoryModel
from modules.utils import combine_layers

logger = logging.getLogger(__name__)

class MyDataset(Dataset):
    # TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label, all_mlms, all_mlm_masks, all_nonce)
    def __init__(self, all_input_ids, all_input_mask, all_segment_ids, all_label, all_mlms, all_mlm_masks, all_nonce):
        self.base_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        self.mlm_input = all_mlms
        self.mlm_mask = all_mlm_masks
        self.nonceMLM = all_nonce
    def __getitem__(self, idx):

        b = self.base_dataset.__getitem__(idx)
        mlm_input = self.mlm_input[idx]
        mlm_mask = self.mlm_mask[idx]
        nonceMLM = self.nonceMLM[idx]

        return (b[0], b[1], b[2], b[3], mlm_input, mlm_mask, nonceMLM)

    def __len__(self):
        return len(self.base_dataset)

    def collate(self, batch):
        #for i in range(len(batch[4])):
         #   for k in range(len(batch)):
          #      if k != 6:
           #         print(batch[k][i].shape)
        # print(batch, "batch")
        out_batch = []
        for i in range(4):
            out_batch.append(default_collate([b[i] for b in batch]))
        for i in range(4,7):
            out_batch.append([b[i] for b in batch])
        return out_batch



def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {"acc": simple_accuracy(preds, labels)}

class MorphMemoryModelMC(MorphMemoryModel):

    def __init__(self, firstLM, secondLM, nonces, device, layers, mask_token_id, memory_config, emb_gen, rescale):
        super(MorphMemoryModelMC, self).__init__(firstLM, secondLM, nonces, device, layers, mask_token_id,
                                                   memory_config, emb_gen)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.firstLM.config.hidden_size,
                                                   nhead=self.firstLM.config.num_attention_heads,
                                                   activation='relu',
                                                   batch_first=True).to(self.device)
        self.emb_gen = nn.TransformerEncoder(encoder_layer, num_layers=1).to(self.device)
        self.cls_token = nn.Parameter(torch.randn(1, self.firstLM.config.hidden_size, device=self.device))
        self.dropout = nn.Dropout(0.2)
        initial_first_ind = int(self.firstLM.config.vocab_size - len(self.nonces))
        initial_second_ind = int(self.secondLM.config.vocab_size - len(self.nonces))

        self.first_list = list(range(initial_first_ind, self.firstLM.config.vocab_size))
        self.second_list = list(range(initial_second_ind, self.secondLM.config.vocab_size))

        self.std_second = self.secondLM.get_input_embeddings().weight[:initial_second_ind, :].norm(dim=1).std()
        self.mean_norm = self.secondLM.get_input_embeddings().weight[:initial_second_ind, :].norm(dim=1).mean()

        self.memory_config = memory_config
        self.rescale = rescale

    def swap_with_mask(self, inputs):
        inp = inputs.clone()
        for nonce in self.first_list:
            inp[inp == nonce] = self.mask_token_id
        return inp

    def get_new_weights_new(self, task, memory):

        if task == 'MLM':
            ref_model = self.firstLM

        elif task == "Task":
            ref_model = self.secondLM
        else:
            raise NotImplementedError

        w = ref_model.get_input_embeddings().weight.clone()
        n, hidden = w.shape
        if not ref_model.get_input_embeddings().weight.requires_grad:
            w.requires_grad = True

        msk = torch.zeros_like(w).to(self.device)
        msk2 = torch.zeros_like(w).to(self.device)
        for key in memory.memory:
            #             print(key)
            #             print(hidden)
            #             print(torch.tensor([key]).to(self.device).expand(1, hidden).shape)
            if self.rescale:
                msk = msk.scatter(0, torch.tensor([key]).to(self.device).expand(1, hidden), memory.retrieve(key,
                                                                                                            std=self.std_second,
                                                                                                            mean=self.mean_norm,
                                                                                                            normalize=True))
            else:
                msk = msk.scatter(0, torch.tensor([key]).to(self.device).expand(1, hidden), memory.retrieve(key))
            msk2[key, :] = 1.

        return w * (~msk2.bool()) + msk

    def forward(self, batch):

        mlm_inputs = batch["mlm_inputs"]
        attn_mask = batch['mlm_mask']
        # task_inputs = {'input_ids': batch["task_inputs"]['input_ids'].to(self.device),
        #                'attention_mask': batch["task_inputs"]['attention_mask'].to(self.device)}
        nonceMLM = batch["nonceMLM"]
        # nonceTask = batch["nonceTask"]
        task_labels = batch['labels']

        b_task = len(mlm_inputs)
        k_task, l_task = mlm_inputs[0].shape

        task_ids = batch['input_ids']
        task_attn = batch["attention_mask"]

        outs = []
        memories = []
        embs = []
        #         attns = []
        for i in range(b_task):
            contexts = mlm_inputs[i]
            new_token = nonceMLM[i]
            memory = OnlineProtoNet(self.memory_config, self.device)
            
            new_inputs = self.swap_with_mask(contexts)
            attn = attn_mask[i]
            #print(attn.shape, new_inputs.shape)
            with torch.no_grad():
                first_out = self.firstLM(input_ids=new_inputs, attention_mask=attn,
                                         output_hidden_states=True)

            first_hidden = first_out.hidden_states
            combined = self.dropout(combine_layers(first_hidden, self.layers))
            if len(combined.shape) == 2:
                combined = combined.unsqueeze(0)

            cls = self.cls_token.unsqueeze(0).expand(combined.shape[0], -1, -1)
            attn = torch.cat([torch.tensor([1], device=self.device).unsqueeze(0).expand(attn.shape[0], -1),
                              attn], dim=1)
            embed_inputs = torch.cat([cls, combined], dim=1)
            embeds = self.emb_gen(embed_inputs, src_key_padding_mask=~attn.bool())
            nonce_embeds = embeds[:, 0]
            memory.store(new_token, nonce_embeds)

            new_w = self.get_new_weights_new(task="Task", memory=memory)

            input_embeds = F.embedding(task_ids[i], new_w)
            embs.append(input_embeds)

        inputs_embeds = torch.stack(embs)
        #         print(inputs_embeds.shape)
        #         print(task_attn.shape)
        #         print(task_labels.shape)
        out_vals = self.secondLM(
            inputs_embeds=inputs_embeds,
            attention_mask=task_attn,
            labels=task_labels,
            output_hidden_states=True
        )
        return out_vals



class ARCExample(object):
    """A single training/test example for the ARC dataset."""

    def __init__(self,
                 arc_id,
                 question,
                 para,
                 choices,
                 num_choices,
                 label=None):
        self.arc_id = arc_id
        self.question = question
        self.para = para
        if len(choices) > num_choices:
            raise ValueError("More choices: {} in question: {} than allowed: {}".format(
                choices, question, num_choices
            ))
        self.choices = [choice["text"] for choice in choices]
        self.choice_paras = [choice.get("para") for choice in choices]
        if len(choices) < num_choices:
            add_num = num_choices - len(choices)
            self.choices.extend([""] * add_num)
            self.choice_paras.extend([None] * add_num)
        label_id = None
        if label is not None:
            for (idx, ch) in enumerate(choices):
                if ch["label"] == label:
                    label_id = idx
                    break
            if label_id is None:
                raise ValueError("No answer found matching the answer key:{} in {}".format(
                    label, choices
                ))
        self.label = label_id

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"arc_id: {self.arc_id}",
            f"para: {self.para}",
            f"question: {self.question}",
            f"choices: {self.choices}"
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label

class InputFeaturesMorph(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 mlm_inputs,
                 mlm_mask,
                 nonceMLM,
                 label):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
            }
            for _, input_ids, input_mask, segment_ids  in choices_features
        ]
        self.label = label
        self.mlm_inputs = mlm_inputs
        self.mlm_mask = mlm_mask
        self.nonceMLM = nonceMLM

class ARCExampleMorph(object):
    """A single training/test example for the ARC dataset."""

    def __init__(self,
                 arc_id,
                 question,
                 para,
                 choices,
                 num_choices,
                 sentences,
                 label=None):
        self.arc_id = arc_id
        self.question = question
        self.para = para
        self.sentences = sentences
        if len(choices) > num_choices:
            raise ValueError("More choices: {} in question: {} than allowed: {}".format(
                choices, question, num_choices
            ))
        self.choices = [choice["text"] for choice in choices]
        self.choice_paras = [choice.get("para") for choice in choices]
        if len(choices) < num_choices:
            add_num = num_choices - len(choices)
            self.choices.extend([""] * add_num)
            self.choice_paras.extend([None] * add_num)
        label_id = None
        if label is not None:
            for (idx, ch) in enumerate(choices):
                if ch["label"] == label:
                    label_id = idx
                    break
            if label_id is None:
                raise ValueError("No answer found matching the answer key:{} in {}".format(
                    label, choices
                ))
        self.label = label_id

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"arc_id: {self.arc_id}",
            f"para: {self.para}",
            f"question: {self.question}",
            f"choices: {self.choices}"
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)

def convert_examples_to_features(examples, max_seq_length,
                                 tokenizer,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    features = []
    for (ex_index, example) in enumerate(examples):
        if example is None:
            continue
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % ex_index)

        choices_features = []
        for choice_idx, choice in enumerate(example.choices):
            context = example.para
            if example.choice_paras[choice_idx] is not None:
                context += " " + example.choice_paras[choice_idx]
            context += " " + example.question
            context_tokens = tokenizer.tokenize(context)
            context_tokens_choice = context_tokens[:]
            choice_tokens = tokenizer.tokenize(choice)
            # Modifies `context_tokens_choice` and `ending_tokens` in
            # place so that the total length is less than the
            # specified length.  Account for [CLS], [SEP], [SEP] with
            # "- 3"
            _truncate_seq_pair(context_tokens_choice, choice_tokens, max_seq_length - 3)
            tokens_a = context_tokens_choice
            tokens_b = choice_tokens

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = tokens_a + [sep_token]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if tokens_b:
                tokens += tokens_b + [sep_token]
                segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

            if cls_token_at_end:
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label_id = example.label
        if isinstance(example, ARCExampleMorph):
            if len(example.sentences) == 0 or len(example.sentences) > 5:
                continue
            mlm_inputs = tokenizer(example.sentences, padding="max_length", max_length=max_seq_length, return_tensors='pt')
            nonceMLM = tokenizer.convert_tokens_to_ids("<nonce>")
            input_features = InputFeaturesMorph(
                example_id=example.arc_id,
                choices_features=choices_features,
                label=label_id,
                mlm_inputs=mlm_inputs['input_ids'],
                mlm_mask=mlm_inputs['attention_mask'],
                nonceMLM=nonceMLM
            )
        else:
            input_features = InputFeatures(
                example_id=example.arc_id,
                choices_features=choices_features,
                label=label_id
            )
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info(f"arc_id: {example.arc_id}")
            for choice_idx, (tokens, input_ids, input_mask, segment_ids) in enumerate(
                    choices_features):
                logger.info(f"choice: {choice_idx}")
                logger.info(f"tokens: {' '.join(tokens)}")
                logger.info(f"input_ids: {' '.join(map(str, input_ids))}")
                logger.info(f"input_mask: {' '.join(map(str, input_mask))}")
                logger.info(f"segment_ids: {' '.join(map(str, segment_ids))}")
                logger.info(f"label: {label_id}")

        features.append(
            input_features
        )
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class ARCExampleReader:
    """Reader for ARC dataset format"""

    ## older versions
    # def get_train_examples(self, data_dir, max_choices,other_name=""):
    #     return self._create_examples(
    #         self._read_jsonl(os.path.join(data_dir, "train.jsonl")), for_training=True,
    #         max_choices=max_choices)

    # def get_dev_examples(self, data_dir, max_choices,other_name=""):
    #     return self._create_examples(
    #         self._read_jsonl(os.path.join(data_dir, "dev.jsonl")), for_training=False,
    #         max_choices=max_choices)

    ## reads files directly
    def get_train_examples(self, data_dir, max_choices, exclusion=""):
        return self._create_examples(
            self._read_jsonl(data_dir, exclusion), for_training=True,
            max_choices=max_choices)

    def get_dev_examples(self, data_dir, max_choices, new_token, morph, exclusion=""):
        print("dev","new", new_token)
        print("dev", "morph", morph)
        return self._create_examples(
            self._read_jsonl(data_dir, exclusion), for_training=False,
            max_choices=max_choices, new_tok=new_token, morph=morph)

    def _read_jsonl(self, filepath, exclusion):
        excluded_set = set([d.strip() for d in exclusion.split(',')]) if exclusion else None

        with open(filepath, 'r') as fp_reader:
            for line in fp_reader:
                ## exclude something?
                if excluded_set:
                    json_line = json.loads(line.strip())
                    dataset = json_line.get("notes", {})
                    try:
                        if dataset["source"].strip() not in excluded_set:
                            # print(dataset["source"])
                            continue
                    except KeyError:
                        raise ValueError('Included dataset does not appear to contain multiple datasets!')
                    #
                    # print(dataset["source"])
                yield json.loads(line.strip())

    def _create_examples(self, json_stream, for_training, max_choices, new_tok=False, morph=False):
        """Creates examples for the training and dev sets."""
        assert (morph and new_tok) or (not morph), "Morph is only used when doing new token eval"
        for input_json in json_stream:
            if "answerKey" not in input_json and for_training:
                raise ValueError("No answer key provided for training in {}".format(input_json))

            ###
            if new_tok:
                word = input_json['notes']['surface_form']
                new_token = "<nonce>"

                question = re.sub(r"\b({})\b".format(word), new_token, input_json["question"]["stem"], flags=re.I)
                choices = input_json["question"]["choices"]
                new_choices = []
                for c in choices:
                    new_c = {'text': re.sub(r"\b({})\b".format(word), new_token, c['text'], flags=re.I),
                             'label': c['label']}
                    new_choices.append(new_c)

                if morph:
                    gold = input_json['notes']['gold_synset']
                    examples = wn.synset(gold).examples()
                    sentences = [s for s in examples if word in s]
                    sentences = [re.sub(r"\b({})\b".format(word), new_token, s, flags=re.I) for s in sentences]
                    arc_example = ARCExampleMorph(
                        arc_id=input_json["id"],
                        question=question,
                        para=input_json.get("para", ""),
                        choices=new_choices,
                        num_choices=max_choices,
                        label=input_json.get("answerKey"),
                        sentences=sentences
                    )
                else:
                    gold = input_json['notes']['gold_synset']
                    examples = wn.synset(gold).examples()
                    sentences = [s for s in examples if word in s]
                    sentences = [re.sub(r"\b({})\b".format(word), new_token, s, flags=re.I) for s in sentences]
                    if len(sentences) == 2:
                        arc_example = ARCExample(
                            arc_id=input_json["id"],
                            question=question,
                            para=input_json.get("para", ""),
                            choices=new_choices,
                            num_choices=max_choices,
                            label=input_json.get("answerKey")
                        )
                    else:
                        arc_example = None
            else:
                arc_example = ARCExample(
                    arc_id=input_json["id"],
                    question=input_json["question"]["stem"],
                    para=input_json.get("para", ""),
                    choices=input_json["question"]["choices"],
                    num_choices=max_choices,
                    label=input_json.get("answerKey")
                )
            yield arc_example

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, reader, tokenizer, accelerator):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size * 2
    # train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size)
    train_dataloader = accelerator.prepare(train_dataloader)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
    accelerator.register_for_checkpointing(optimizer)
    accelerator.register_for_checkpointing(scheduler)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    # global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    # train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    # set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    for epoch in range(args.num_train_epochs):
        # epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(train_dataloader):
            log_dict = {}
            model.train()
            #batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      #'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                      # XLM don't use segment_ids
                      'labels': batch[3]}
            outputs = model(**inputs)

            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)


            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            log_dict['train loss'] = loss.item()
            scheduler.step()  # Update learning rate schedule
            optimizer.step()
            model.zero_grad()
            # global_step += 1

            # if (step + 1) % args.gradient_accumulation_steps == 0:
            #     scheduler.step()  # Update learning rate schedule
            #     optimizer.step()
            #     model.zero_grad()
            #     global_step += 1

                # if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                #     # Log metrics
                #
                #     if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                #         eval_log_dict = {}
                #         results = evaluate(args, model=model, reader=reader, tokenizer=tokenizer)
                #         for key, value in results.items():
                #             eval_log_dict['eval_{}'.format(key)] = value
                #             eval_log_dict['global step'] = global_step
                #         # wandb.log(eval_log_dict)
            #                     tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
            #                     tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
            #                     logging_loss = tr_loss

            #                 if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0 and not args.save_no_checkpoints:
            #                     # Save model checkpoint
            #                     output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
            #                     if not os.path.exists(output_dir):
            #                         os.makedirs(output_dir)
            #                     model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            #                     model_to_save.save_pretrained(output_dir)
            #                     torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            #                     logger.info("Saving model checkpoint to %s", output_dir)
            log_dict['epoch'] = epoch
            accelerator.log(log_dict)
    #     if args.local_rank in [-1, 0]:
    #         tb_writer.close()
        eval_log_dict = {}
        results = evaluate(args, model=model, reader=reader, tokenizer=tokenizer, accelerator=accelerator)
        for key, value in results.items():
            eval_log_dict['eval_{}'.format(key)] = value
            eval_log_dict['epoch'] = epoch

        accelerator.log(eval_log_dict)
        accelerator.save_state(args.output_dir + args.dataset + "checkpoint_{}".format(epoch))

    return args.num_train_epochs, tr_loss / args.num_train_epochs


def evaluate(args, model, reader, tokenizer, accelerator, prefix="", next_fname=""):
    eval_task_names = ["qa"]
    eval_outputs_dirs = [args.output_dir]

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, reader, tokenizer, evaluate=True, next_fname=next_fname)

        # if not os.path.exists(eval_output_dir) :
        #     os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * 2
        # Note that DistributedSampler samples randomly
        # eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size,
                                     collate_fn = eval_dataset.collate if isinstance(eval_dataset, MyDataset) else default_collate)
        eval_dataloader = accelerator.prepare(eval_dataloader)
        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            #batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          #'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                          # XLM don't use segment_ids
                          'labels': batch[3]}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

            # erint(preds)
            # print(np.argmax(preds,axis=1))
            # print(out_label_ids)
            # print("============")
            ## record out labels

        eval_loss = eval_loss / nb_eval_steps
        ## print out logits before
        final_logits = preds
        preds = np.argmax(preds, axis=1)

        #         model_output_file = os.path.join(args.output_dir,"model_output.txt")
        #         if next_fname:
        #             model_output_file = os.path.join(args.output_dir,"model_output_next.txt")

        #         with open(model_output_file,'w') as s_output:
        #             logger.info('Printing model (i.e., raw logits, label predictions) output to file...')
        #             for i in range(preds.shape[0]):
        #                 print("%s\t%s\t%s\t%s" % (out_label_ids[i],preds[i],
        #                                               str(out_label_ids[i]==preds[i]),
        #                                               ' '.join([str(l) for l in final_logits[i]])),file=s_output)

        ## spit out predictions to file
        result = compute_metrics(preds, out_label_ids)
        results.update(result)
    #         output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    #         if next_fname:
    #             output_eval_file = os.path.join(eval_output_dir, "eval_results_next.txt")

    #         with open(output_eval_file, "w") as writer:
    #             logger.info("***** Eval results {} *****".format(prefix))
    #             for key in sorted(result.keys()):
    #                 logger.info("  %s = %s", key, str(result[key]))
    #                 writer.write("%s = %s\n" % (key, str(result[key])))

    return results

def evaluate_morph(args, model, reader, tokenizer, accelerator, prefix="", next_fname=""):
    eval_task_names = ["qa"]
    eval_outputs_dirs = [args.output_dir]

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, reader, tokenizer, evaluate=True, next_fname=next_fname)

        # if not os.path.exists(eval_output_dir) :
        #     os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * 2
        # Note that DistributedSampler samples randomly
        # eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, collate_fn=eval_dataset.collate)
        eval_dataloader = accelerator.prepare(eval_dataloader)
        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            #batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          #'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                          # XLM don't use segment_ids
                          'labels': batch[3],
                          'mlm_inputs': batch[4],
                          'mlm_mask': batch[5],
                          'nonceMLM': batch[6]}
                outputs = model(inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

            # print(preds)
            # print(np.argmax(preds,axis=1))
            # print(out_label_ids)
            # print("============")
            ## record out labels

        eval_loss = eval_loss / nb_eval_steps
        ## print out logits before
        final_logits = preds
        preds = np.argmax(preds, axis=1)

        #         model_output_file = os.path.join(args.output_dir,"model_output.txt")
        #         if next_fname:
        #             model_output_file = os.path.join(args.output_dir,"model_output_next.txt")

        #         with open(model_output_file,'w') as s_output:
        #             logger.info('Printing model (i.e., raw logits, label predictions) output to file...')
        #             for i in range(preds.shape[0]):
        #                 print("%s\t%s\t%s\t%s" % (out_label_ids[i],preds[i],
        #                                               str(out_label_ids[i]==preds[i]),
        #                                               ' '.join([str(l) for l in final_logits[i]])),file=s_output)

        ## spit out predictions to file
        result = compute_metrics(preds, out_label_ids)
        results.update(result)
    #         output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    #         if next_fname:
    #             output_eval_file = os.path.join(eval_output_dir, "eval_results_next.txt")

    #         with open(output_eval_file, "w") as writer:
    #             logger.info("***** Eval results {} *****".format(prefix))
    #             for key in sorted(result.keys()):
    #                 logger.info("  %s = %s", key, str(result[key]))
    #                 writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def load_model_partial(fname, model):
    model_dict = model.state_dict()
    state_dict = torch.load(fname)
    partial_states = {k:v for k, v in state_dict.items() if "emb_gen" in k or "memory" in k or "cls" in k}
#     print(partial_states)
    model_dict.update(partial_states)
    model.load_state_dict(model_dict)
    return model


def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]


def load_and_cache_examples(args, reader, tokenizer, evaluate=False, next_fname=""):
    # Load data features from cache or dataset file
    #     cached_features_file = os.path.join(args.output_dir, 'cached_{}_{}_{}_{}'.format(
    #         'dev' if evaluate else 'train',
    #         list(filter(None, args.model_name_or_path.split('/'))).pop(),
    #         str(args.max_seq_length),
    #         "arc"))
    #     if os.path.exists(cached_features_file) and not next_fname:
    if False:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)

        ### find the correct dataset paths
        if evaluate:
            if args.dev_name:
                data_file = args.dev_name
            else:
                data_file = os.path.join(args.data_dir, "dev.jsonl")
        ## training
        else:
            if args.train_name:
                data_file = args.train_name
            else:
                data_file = os.path.join(args.data_dir, "train.jsonl")

        # examples = reader.get_dev_examples(args.data_dir, args.num_choices) if evaluate else reader.get_train_examples(args.data_dir, args.num_choices)
        examples = reader.get_dev_examples(data_file, args.num_choices, exclusion=args.limit_test, morph=args.use_morph, new_token=args.eval_new_token) if evaluate else \
            reader.get_train_examples(data_file, args.num_choices, exclusion=args.limit_train)
        features = convert_examples_to_features(examples, args.max_seq_length, tokenizer,
                                                cls_token_at_end=False,
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                sep_token=tokenizer.sep_token,
                                                cls_token_segment_id=0,
                                                pad_on_left=False,
                                                # pad on the left for xlnet
                                                pad_token_segment_id= 0)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
    all_label = torch.tensor([f.label for f in features], dtype=torch.long)
    if isinstance(features[0], InputFeaturesMorph):
        all_mlms = [f.mlm_inputs for f in features]
        all_mlm_masks = [f.mlm_mask for f in features]
        all_nonce = [f.nonceMLM for f in features]
        print("here")
        dataset = MyDataset(all_input_ids, all_input_mask, all_segment_ids, all_label, all_mlms, all_mlm_masks, all_nonce)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default="../semantic_fragments/data_mcqa/definitions/", type=str,
                        help="The input data dir. Should contain the .jsonl files (or other data files) for the task.")
    parser.add_argument("--output_dir", default="semantic_outputs/", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--dataset", default="definition", type=str)
    ## Other parameters
    parser.add_argument("--use_morph", action="store_true")
    parser.add_argument("--eval_new_token", action="store_true")
    parser.add_argument("--eval_mask", action="store_true")
    parser.add_argument('--num_choices',
                        type=int,
                        default=4,
                        help="Number of answer choices (will pad if less, throw exception if more)")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--load_baseline_checkpoint", default="", type=str)

    ###########################
    # ### KYLES NEW SETTINGS  #
    ###########################

    parser.add_argument('--dev_name2',
                        default='',
                        help="the name of the dev experiment")

    parser.add_argument('--dev_name',
                        default='',
                        help="the name of the dev experiment")

    parser.add_argument("--remove_model",
                        default=False,
                        action='store_true',
                        help="Remove the pytorch model after done training")

    parser.add_argument("--override",
                        default=False,
                        action='store_true',
                        help="Override the existing directory")

    parser.add_argument("--no_save_checkpoints",
                        default=False,
                        action='store_true',
                        help="Don't save the model after each checkpoint ")

    parser.add_argument("--run_existing",
                        default='',
                        help="Run in eval model with an existing model, points to output_model_file")

    parser.add_argument("--bert_config",
                        default='',
                        help="Location of the existing BERT configuration")


    parser.add_argument("--exclude_dataset",
                        default='',
                        type=str,
                        help="Datasets to exclude (in case of Aristo dataset with multiple datasets built in)")

    parser.add_argument("--train_name",
                        default='',
                        type=str,
                        help="the number of multiple choice options")

    parser.add_argument("--limit_train",
                        default='',
                        type=str,
                        help="(for multi-dataset datasets) the datasets to use for training")

    parser.add_argument("--limit_test",
                        default='',
                        type=str,
                        help="(for multi-dataset datasets) the datasets to use for testing")


    args = parser.parse_args()

    # Setup logging
    log_file = "semantic_logger.log"
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO, filename=log_file)
    #     logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
    #                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    # set_seed(args)
    accelerator = Accelerator(log_with="wandb")
    device = accelerator.device

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    if args.use_morph:
        firstLM = RobertaForMaskedLM.from_pretrained("roberta-base")
        secondLM = RobertaForMultipleChoice.from_pretrained("tmp/test-mlm2/checkpoint-22000")
        logger.info("initial state dict: {}".format(secondLM.state_dict()))
        state_dict = torch.load(args.load_baseline_checkpoint + "/" + "pytorch_model.bin")
        secondLM.load_state_dict(state_dict)
        logger.info("first lm weights {}".format(firstLM.get_input_embeddings().weight))
        logger.info("second lm weights {}".format(secondLM.get_input_embeddings().weight))

    else:
        model = RobertaForMultipleChoice.from_pretrained("tmp/test-mlm2/checkpoint-22000")
    assert not (args.do_train and args.eval_new_token), "Only eval if you're evaluating with new tokens"

    if args.eval_new_token:
        tokenizer.add_tokens(['<nonce>'])
        if not args.use_morph:
            with torch.no_grad():
                mean_embed = torch.mean(model.get_input_embeddings().weight, dim=0)
        else:
            firstLM.resize_token_embeddings(len(tokenizer))
            secondLM.resize_token_embeddings(len(tokenizer))

        #    model.resize_token_embeddings(len(tokenizer))
        #    model.get_input_embeddings().weight[-1, :] = mean_embed


    logger.info('loaded a pre-trained model..')
    #for param in model.parameters():
    #    param.requires_grad = False
    if not args.use_morph:
        model.get_input_embeddings().weight.requires_grad=False
    #for param in model.classifier.parameters():
    #    param.requires_grad = True
    else:
        memory_config = AggregatorConfig()
        new_toks = list(set(tokenizer.convert_tokens_to_ids(['<nonce>'])))
        path = "model_checkpoints/e20_smallevalind_rescaleFalse_wd0.03_resample_False__full_gelu_online_6examples_0.003_mean_Transformer_bs8_modified_mamlFalse_randomTrue_finetuneFalse_cat_Falselayers4_binary_False_mask_newFalse/MLMonline_memory_model_roberta_roberta_mean_memory_NormedOutput/checkpoints/"
        print(path)
        if "rescaleTrue" in path:
            rescale = True

        else:
            rescale = False
        chkpt = "checkpoint_8"
        chkpt_path = path + "{}/".format(chkpt)
        name = "pytorch_model.bin"
        model=MorphMemoryModelMC(firstLM, secondLM, new_toks, device, [-1, -2, -3, -4],
                                              tokenizer.mask_token_id, memory_config, 'Transformer', rescale=rescale)
        with torch.no_grad():
            norm = model.cls_token.norm()
        logger.info("pre loading norm {}".format(norm.item()))
        model = load_model_partial(chkpt_path + name, model)
        with torch.no_grad():
            logger.info("post loading norm {}".format(model.cls_token.norm().item()))

        logger.info("secondLM state dict after {}".format(model.secondLM.state_dict()))

    project = "semantics"

    # run = wandb.init(project=project, reinit=True)

    accelerator.init_trackers(
        project_name=project,
        config={"learning_rate": args.learning_rate,
                },
    )
    model = accelerator.prepare(model)
    logger.info("Training/evaluation parameters %s", args)

    reader = ARCExampleReader()
    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, reader, tokenizer, evaluate=False)


        global_step, tr_loss = train(args, train_dataset=train_dataset, model=model, reader=reader,
                                     tokenizer=tokenizer, accelerator=accelerator)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        if args.do_train:

            logger.info("Saving model checkpoint to %s", args.output_dir)
            accelerator.save_state(args.output_dir + args.dataset + "checkpoint_{}".format(global_step + 1))
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`

    ###################################
    # ## EVALUATION (KYLE's VERSION)  #
    ###################################

    results = {}
    if args.load_baseline_checkpoint != "" and not args.use_morph:
        logger.info("loading checkpoint {}".format(args.load_baseline_checkpoint))
        print("loading checkpoint")
        accelerator.load_state(args.load_baseline_checkpoint)
    if args.eval_new_token:
        logger.info("Resizing embeds for new token")

        with torch.no_grad():
            if not args.use_morph:
                model.module.resize_token_embeddings(len(tokenizer))
                if args.eval_mask:
                    model.module.get_input_embeddings().weight[-1, :] = model.module.get_input_embeddings().weight[tokenizer.mask_token_id, :]
                else:
                    model.module.get_input_embeddings().weight[-1, :] = mean_embed
    if args.do_eval:

        ## the actual evaluation
        if args.use_morph:
            global_step = ""
            result = evaluate_morph(args, model, reader, tokenizer, prefix=global_step, accelerator=accelerator)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)
            eval_log_dict = {}
            for key, value in results.items():
                eval_log_dict['eval_{}'.format(key)] = value
        else:
            global_step = ""
            result = evaluate(args, model, reader, tokenizer, prefix=global_step, accelerator=accelerator)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)
            eval_log_dict = {}
            for key, value in results.items():
                eval_log_dict['eval_{}'.format(key)] = value
            # eval_log_dict['epoch'] = epoch

        accelerator.log(eval_log_dict)


    # Evaluation
    # results = {}
    # if args.do_eval and args.local_rank in [-1, 0]:
    #     checkpoints = [args.output_dir]
    #     if args.eval_all_checkpoints:
    #         checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
    #         logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    #     logger.info("Evaluate the following checkpoints: %s", checkpoints)
    #     for checkpoint in checkpoints:
    #         global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
    #         model = model_class.from_pretrained(checkpoint)
    #         model.to(args.device)
    #         result = evaluate(args, model, reader, tokenizer, prefix=global_step)
    #         result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
    #         results.update(result)

    #     if args.remove_model:
    #         logger.info('REMOVING THE MODEL!')
    #         try:
    #             os.remove(os.path.join(args.output_dir,"pytorch_model.bin"))
    #             os.remove(os.path.join(args.output_dir,"vocab.txt"))
    #         except Exception as e:
    #             logger.error(e,exc_info=True)

    return results

if __name__ == "__main__":
    main()
