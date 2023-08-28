import math
from argparse import ArgumentParser
from functools import reduce
import os
from copy import deepcopy
from scipy import stats
import json
import higher
import csv
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, RobertaForMaskedLM, GPT2Model, RobertaForQuestionAnswering, \
    AutoModelForSequenceClassification, AutoModelForCausalLM, GPTJForCausalLM
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import load_from_disk

from configs.config import *
from eval_utils import compute_exact_match
from modules.buffer import RetrievalBuffer
from modules.model import MorphMemoryModel, MorphMemoryModelSQuAD, MorphMemoryModelSNLI, MorphMemoryModelGPT, \
    MorphMemoryModelGPTOnline, MorphMemoryModelGPTSubtoken, MorphMemoryModelMLMOnline, MorphMemoryModelGPTOnlineBinary, \
    MorphMemoryModelMLMOnlineBinary, MorphMemoryModelMLMOnlineFull
from data.data_utils import *
from train_utils import *
from data.few_shot_datasets import *

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-6)
    # parser.add_argument("--warmup", type=int, default=1e2)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--tgt_data_path", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load_checkpoint", type=bool, default=False)
    parser.add_argument("--mlm_prob", type=float, default=0.15)
    parser.add_argument("--taskName", type=str, default='mlm')
    parser.add_argument("--secondLM", type=str)
    parser.add_argument("--memory", type=str, default="mean")
    parser.add_argument("--num_examples", type=int, default=2)
    parser.add_argument("--intermediate_loss", type=bool, default=False)
    parser.add_argument("--trial", type=str, default='l2')
    parser.add_argument("--emb_gen", type=str, default='mlp')
    parser.add_argument("--strategy", type=str, default='mask')
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--maml", action="store_true")
    parser.add_argument("--word_path", type=str, default='')
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--random_ex", action="store_true")
    parser.add_argument("--cat", action="store_true")
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--mask_new", action="store_true")
    parser.add_argument("--resample", action="store_true")
    parser.add_argument("--prefill", action="store_true")
    return parser


def main():
    args = get_arguments().parse_args()

    print("Arguments: ", args)

    layers = [-1, -2, -3, -4] # add arg to pass this
    print("here")
    tokenizerMLM = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
    print("made first tokenizer")
    firstLM = RobertaForMaskedLM.from_pretrained('roberta-base')
    print("made firstLM, tokenizer")
    # change these to accept model names
    if args.taskName == "mlm":
        tokenizerTask = AutoTokenizer.from_pretrained('roberta-base', use_fast=True)
        secondLM = RobertaForMaskedLM.from_pretrained('roberta-base')

    elif args.taskName == "autoregressive":
        tokenizerTask = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
        secondLM = GPT2Model.from_pretrained("gpt2")

    elif args.taskName == "squad":
        tokenizerTask = AutoTokenizer.from_pretrained('deepset/tinyroberta-squad2', use_fast=True)
        secondLM = RobertaForQuestionAnswering.from_pretrained("deepset/tinyroberta-squad2")

    elif args.taskName == "snli":
        tokenizerTask = AutoTokenizer.from_pretrained('cross-encoder/nli-roberta-base', use_fast=True)
        secondLM = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-roberta-base')
    elif args.taskName == "addition":
        tokenizerTask = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
        tokenizerTask.pad_token = tokenizerTask.unk_token
        secondLM = AutoModelForCausalLM.from_pretrained("gpt2")
        nonces = ["<OP>"]
        dataset_name = "addition"
    elif args.taskName == "addition_subtok":
        tokenizerTask = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
        tokenizerTask.pad_token = tokenizerTask.unk_token
        secondLM = AutoModelForCausalLM.from_pretrained("gpt2")
        nonces = ["<NUM1>", "<NUM2>"]
        dataset_name = "addition_subtok"

    elif args.taskName == "online":
        if args.secondLM == "gpt2":
            tokenizerTask = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
            tokenizerTask.pad_token = tokenizerTask.unk_token
            print("making model")
            secondLM = AutoModelForCausalLM.from_pretrained("gpt2")
            print("made model")
        elif args.secondLM == "roberta":
            tokenizerTask = AutoTokenizer.from_pretrained('roberta-base', use_fast=True)
            secondLM = RobertaForMaskedLM.from_pretrained('tmp/test-mlm2/checkpoint-22000')
        elif args.secondLM == "gptj":
            tokenizerTask = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
            secondLM = GPTJForCausalLM.from_pretrained(
                                    "EleutherAI/gpt-j-6B",
                                    revision="float16",
                                    torch_dtype=torch.float16,
                                    low_cpu_mem_usage=True)

        else:
            raise NotImplementedError("{} Not Implemented for Task LM".format(args.secondLM))
        # with open(args.word_path, 'r') as f:
        #     reader = csv.reader(f)
        #     rows = [row for row in reader]
        # words = rows[0]
        word_dict = load_from_disk(args.word_path)
        words = word_dict['train']['words'] + word_dict['test']['words']
        nonces = list(map(lambda w: "<{}_new>".format(w), words))
    else:
        raise NotImplementedError("{} not implemented".format(args.taskName))

    if "addition" not in args.taskName:
        dataset = load_from_disk(args.data_path)
        # if args.taskName == "online":
        #     dataset = dataset.filter(lambda ex: len(ex['text']) > 10)

    print("here2")
    if "chimera" in args.data_path:
        nonces = list(map(make_nonce, list(set(dataset["train"]['id'] + dataset["test"]['id']))))
        dataset_name = "chimera{}".format(args.num_examples)

    elif "sanity" in args.data_path:
        nonces = list(map(make_sanity_nonce, list(set(dataset['word']))))
        dataset_name = "sanity"

    elif "squad" in args.data_path:
        dataset = dataset.filter(lambda ex: ex['replace'] != '')
        nonces = list(set(map(lambda e: "<{}_nonce>".format(e.lower()), dataset['replace'])))
        dataset_name = "squad"
    elif "snli" in args.data_path:
        nonces = list(map(snli_nonce, list(set(dataset['replace']))))
        dataset_name = "snli"
    elif "wikitext" in args.data_path:
        dataset_name = "online"
    else:
        if "addition" not in args.taskName:
            raise NotImplementedError("Not implemented for this dataset")

    # add support for other datasets

    # expand tokenizer
    tokenizerMLM.add_tokens(nonces)
    tokenizerTask.add_tokens(nonces)

    # resize models
    firstLM.resize_token_embeddings(len(tokenizerMLM))
    secondLM.resize_token_embeddings(len(tokenizerTask))

    # memory
    if args.memory == "mean":
        memory_config = AggregatorConfig()
        weight_decay = 0.02

    elif args.memory == "rnn":
        memory_config = RNNAggConfig()
        weight_decay = 0.015
    elif args.memory == "cls":
        memory_config = TransformerCLSConfig()
        weight_decay = 0.015
    else:
        raise NotImplementedError("This memory aggregation is not implemented")

    run_name = "redone_context_resample_{}__redo_full_gelu_{}_{}examples_{}_{}_{}_bs={}_modified_maml={}_random={}_finetune={}_cat_{}layers4_binary_{}_mask_new={}".format(args.resample, dataset_name,
                                                                                                 args.num_examples,
                                                                                                 args.lr,
                                                                                                 memory_config.agg_method,
                                                                                                 args.emb_gen,
                                                                                                 args.batch_size,
                                                                                                 args.maml,
                                                                                                 args.random_ex,
                                                                                                 args.finetune,
                                                                                                 args.cat,
                                                                                                 args.binary,
                                                                                                 args.mask_new)
    # proj_conf = ProjectConfiguration(automatic_checkpoint_naming=True)
    accelerator = Accelerator(log_with="wandb")


    device = accelerator.device

    mask_token_id = tokenizerMLM.mask_token_id

    if "chimera" in args.data_path:

        cos = nn.CosineSimilarity(dim=-1)

        split = dataset["train"].train_test_split(test_size=0.2)
        mlm_dataset = ChimeraMLMDataset(split["train"], tokenizerMLM, tokenizerTask, args.num_examples, args.trial)

        train_dl = DataLoader(mlm_dataset, batch_size=args.batch_size, shuffle=True)

        test_dataset = ChimeraTestDataset(split["test"], tokenizerMLM, tokenizerTask, args.num_examples, args.trial)

        collate = make_collate(test_dataset)

        test_dl = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    elif "sanity" in args.data_path:
        split = dataset.train_test_split(test_size=0.2)
        n = args.num_examples
        tokenizerMLM.add_tokens(nonces)
        tokenizerTask.add_tokens(nonces)

        mlm_dataset = SimpleMLMDataset(split["train"], tokenizerMLM, tokenizerTask, n)

        train_dl = DataLoader(mlm_dataset, batch_size=args.batch_size, shuffle=True)

        # train_eval = ChimeraTestDataset(chimera["train"], tokenizerMLM, tokenizerTask, n, trial)

        test_dataset = SimpleMLMDataset(split["test"], tokenizerMLM, tokenizerTask, n)

        # collate = make_collate(test_dataset)

        test_dl = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    elif "squad" in args.data_path:

        tokenizerMLM.add_tokens(nonces)
        tokenizerTask.add_tokens(nonces)

        split = dataset.train_test_split(0.2)

        train = SimpleSQuADDataset(split['train'], tokenizerMLM, tokenizerTask, args.num_examples)

        test = SimpleSQuADDataset(split['test'], tokenizerMLM, tokenizerTask, args.num_examples)

        train_dl = DataLoader(train, batch_size=args.batch_size, collate_fn=make_collate(train), shuffle=True)

        test_dl = DataLoader(test, batch_size=args.batch_size, collate_fn=make_collate(test))

    elif "snli" in args.data_path:
        n = args.num_examples
        split = dataset.train_test_split(0.2)
        train = SimpleSNLIDataset(split["train"], tokenizerMLM, tokenizerTask, n)
        test = SimpleSNLIDataset(split["test"], tokenizerMLM, tokenizerTask, n)

        train_dl = DataLoader(train, batch_size=args.batch_size, collate_fn=make_collate(train), shuffle=True)
        test_dl = DataLoader(test, batch_size=args.batch_size, collate_fn=make_collate(test))
    elif "wikitext" in args.data_path:
        # split = dataset.train_test_split(0.2)
        nt = tokenizerTask.convert_tokens_to_ids(nonces)
        train = SimpleOnlineDataset(dataset['train'], tokenizerMLM, tokenizerTask, new_tokens=nt, mask_new=args.mask_new)
        train_dl = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True)

        test = SimpleOnlineDataset(dataset['test'], tokenizerMLM, tokenizerTask, new_tokens=nt, mask_new=args.mask_new)
        test_dl = DataLoader(test, batch_size=args.batch_size, shuffle=True, drop_last=True)

    else:
        if args.taskName == "addition":
            train_size = 10000
            operation = "addition"
            orthography = "decimal"
            base_number = 10
            min_digits_train, max_digits_train = 2, 5
            train = MyDataset(n_examples=train_size, min_digits=min_digits_train,
                              max_digits=max_digits_train,
                              operation=operation, orthography=orthography,
                              base_number=base_number, invert_question=False,
                              invert_answer=False, balance=True)

            test = MyDataset(n_examples=1000, min_digits=min_digits_train,
                             max_digits=max_digits_train,
                             operation=operation, orthography=orthography,
                             base_number=base_number, invert_question=False,
                             invert_answer=False, balance=True)

            train_set = SimpleMathDataset(train, tokenizerMLM, tokenizerTask, 30)
            train_dl = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate_fn)

            test_set = SimpleMathDataset(test, tokenizerMLM, tokenizerTask, None)

            test_dl = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate_fn)
        elif args.taskName == "addition_subtok":
            train_size = 10000
            operation = "addition"
            orthography = "decimal"
            base_number = 10
            min_digits_train, max_digits_train = 2, 5
            train = MyDataset(n_examples=train_size, min_digits=min_digits_train,
                              max_digits=max_digits_train,
                              operation=operation, orthography=orthography,
                              base_number=base_number, invert_question=False,
                              invert_answer=False, balance=True)

            test = MyDataset(n_examples=1000, min_digits=min_digits_train,
                             max_digits=max_digits_train,
                             operation=operation, orthography=orthography,
                             base_number=base_number, invert_question=False,
                             invert_answer=False, balance=True)

            train_set = SimpleMathDatasetSubtok(train, tokenizerMLM, tokenizerTask, 30)
            train_dl = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate_fn)

            test_set = SimpleMathDatasetSubtok(test, tokenizerMLM, tokenizerTask, None)

            test_dl = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate_fn)

        else:
            raise NotImplementedError

    new_toks = tokenizerMLM.convert_tokens_to_ids(nonces)


    epochs = args.epochs
    lr = args.lr
    epsilon = 1e-8


    if "squad" in args.data_path:
        test_model = MorphMemoryModelSQuAD(firstLM, secondLM, new_toks,
                                      device, layers, mask_token_id, memory_config, args.emb_gen)
    elif "snli" in args.data_path:
        test_model = MorphMemoryModelSNLI(firstLM, secondLM, new_toks, device, [-1],
                                   tokenizerMLM.mask_token_id, memory_config, args.emb_gen)
    elif "sanity" in args.data_path:
        test_model = MorphMemoryModel(firstLM, secondLM, new_toks,
                                  device, layers, mask_token_id, memory_config, args.emb_gen)
    elif "wikitext" in args.data_path:
        if args.cat:
            assert args.memory == "mean"
        buffer = RetrievalBuffer(15, args.num_examples, new_toks, tokenizerMLM, args.random_ex, args.cat)
        if "gpt" in args.secondLM:
            if not args.binary:
                test_model = MorphMemoryModelGPTOnline(firstLM, secondLM, new_toks, device, [-1],
                                                       tokenizerMLM.mask_token_id, memory_config, emb_type='Transformer')
            else:
                test_model = MorphMemoryModelGPTOnlineBinary(firstLM, secondLM, new_toks, device, [-1],
                                                       tokenizerMLM.mask_token_id, memory_config,
                                                       emb_type='Transformer')
        elif args.secondLM == "roberta":
            if not args.binary:
                test_model = MorphMemoryModelMLMOnlineFull(firstLM, secondLM, new_toks, device, layers,
                                                       tokenizerMLM.mask_token_id, memory_config,
                                                       'Transformer')
            else:
                test_model = MorphMemoryModelMLMOnlineBinary(firstLM, secondLM, new_toks, device, [-1],
                                                       tokenizerMLM.mask_token_id, memory_config,
                                                       emb_type='Transformer')
    else:
        if args.taskName == "addition":
            test_model = MorphMemoryModelGPT(firstLM, secondLM, new_toks, device, [-1],
                                             tokenizerMLM.mask_token_id, memory_config, emb_type='Transformer')
        elif args.taskName == "addition_subtok":
            test_model = MorphMemoryModelGPTSubtoken(firstLM, secondLM, new_toks, device, [-1, -2],
                                             tokenizerMLM.mask_token_id, memory_config, emb_type='Transformer')
        else:
            raise NotImplementedError
    if args.finetune:
        test_model.secondLM.get_input_embeddings().weight.requires_grad = True

        param_list = [{"params": filter(lambda p: p.requires_grad, test_model.secondLM.parameters()), 'lr': 1e-5},
                      {'params': filter(lambda p: p.requires_grad, test_model.emb_gen.parameters()), 'lr': lr, 'weight_decay': weight_decay}]
    else:
        param_list = [{'params': filter(lambda p: p.requires_grad, test_model.emb_gen.parameters()), 'lr': lr,
                       'weight_decay': weight_decay},
                      {'params': filter(lambda p: p.requires_grad, test_model.memory.parameters()), 'lr': lr, 'weight_decay': weight_decay},
                      {'params': test_model.cls_token, 'lr': 3e-3, 'weight_decay': weight_decay}]

    opt = AdamW(param_list,
                eps=epsilon
                )

    warmup_steps = int(len(train_dl) * 0.3)
    eval_ind = 100
    if args.taskName == "addition":
        eval_ind=30



    scheduler = get_linear_schedule_with_warmup(opt, warmup_steps, epochs * len(train_dl))
    intermediate = args.intermediate_loss

    test_model, opt, train_dl, test_dl, scheduler = accelerator.prepare(
        test_model, opt, train_dl, test_dl, scheduler
    )
    accelerator.register_for_checkpointing(opt)
    accelerator.register_for_checkpointing(scheduler)

    project = "fewshot_model_{}".format(args.taskName)

    # run = wandb.init(project=project, reinit=True)

    accelerator.init_trackers(
        project_name=project,
        config={"num_examples": args.num_examples,
                "learning_rate": lr,
                "aggregation": memory_config.agg_method,
                "finetune_token": args.finetune,
                "batch_size": args.batch_size,
                "maml": args.maml,
                "model": test_model.module.model_name
                },
    init_kwargs = {"wandb": {"name": run_name}},
    )
    # wandb.run.name = "gelu_{}_{}examples_{}_{}_{}_bs={}_modified_maml={}_random={}_finetune={}".format(dataset_name,
    #                                                                         args.num_examples,
    #                                                                         lr,
    #                                                                         memory_config.agg_method,
    #                                                                         args.emb_gen,
    #                                                                         args.batch_size,
    #                                                                         args.maml,
    #                                                                         args.random_ex,
    #                                                                         args.finetune)

    if intermediate:
        run_name = run_name + "_intermediate"

    os.makedirs("/scratch/rst306/few_shot_word_learning/checkpoints/{}".format(dataset_name), exist_ok=True)
    os.makedirs("/scratch/rst306/few_shot_word_learning/checkpoints/{}/{}".format(dataset_name, run_name.replace("=", "")), exist_ok=True)

    save_folder = "{}/{}/".format(dataset_name, run_name)
    save_check_dir = "./model_checkpoints/{}/{}/checkpoints".format(run_name.replace("=", ""), test_model.module.model_name)
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_suffix = "/checkpoint_{}"
    save_ind = 0

    best_corr = 0
    best_acc = 0
    best_loss = 10000
    n_inner_iter = 3
    eval_num = 100
    initial_rows = dataset['train'].num_rows
    test_buffer = RetrievalBuffer(15, args.num_examples, new_toks, tokenizerMLM, args.random_ex, args.cat)
    print("Storing in test buffer")
    for b in test_dl:
        test_buffer.store(b['mlm_inputs'])

    buffer = RetrievalBuffer(15, args.num_examples, new_toks, tokenizerMLM, args.random_ex, args.cat)
    if args.prefill:
        for batch in train_dl:
            buffer.store(batch['mlm_inputs'])

    for epoch in range(epochs):
        train_corr = []
        train_losses = []
        train_correct = 0
        train_total = 0
        train_new_token_losses = []
        if "wikitext" in args.data_path:
            if args.resample and epoch > 0:
                with accelerator.main_process_first():
                    base_dir = "../wikitext_resamples/single_example/"
                    word_suffix = "replacements_{}".format(epoch)
                    data_suffix = "train_{}".format(epoch)

                    tokenizerMLM = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
                    tokenizerTask = AutoTokenizer.from_pretrained('roberta-base', use_fast=True)

                    epoch_word_dict = load_from_disk(base_dir + word_suffix)
                    words = epoch_word_dict['train']['words'] + word_dict['test']['words']

                    nonces = list(map(lambda w: "<{}_new>".format(w), words))
                    print(data_suffix)
                    print(nonces)
                    tokenizerMLM.add_tokens(nonces)
                    tokenizerTask.add_tokens(nonces)

                    new_toks = tokenizerTask.convert_tokens_to_ids(nonces)

                    epoch_train_set = load_from_disk(base_dir + data_suffix).select([i for i in range(initial_rows)]) #needs check if less than num initial

                train = SimpleOnlineDataset(epoch_train_set, tokenizerMLM, tokenizerTask, new_tokens=new_toks,
                                                mask_new=args.mask_new)

                train_dl = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True)
                train_dl, test_dl = accelerator.prepare(train_dl, test_dl)

                buffer = RetrievalBuffer(15, args.num_examples, new_toks, tokenizerMLM, args.random_ex, args.cat)
                if args.prefill:
                    if accelerator.is_local_main_process:
                        for batch in train_dl:
                            buffer.store(batch['mlm_inputs'])



        for i, batch in enumerate(train_dl):
            log_dict = {}

            test_model.train()
            test_model.module.firstLM.eval()
            test_model.module.secondLM.eval()
            if not args.maml:
                test_model.zero_grad()
                opt.zero_grad()

                if args.taskName == "online":

                    contexts = []
                    for j in range(batch['mlm_inputs']['input_ids'].shape[0]):

                        to_sample = [n for n in buffer.nonces if n in batch['mlm_inputs']['input_ids'][i]]
                        assert (len(to_sample) == 1)
                        for n in to_sample:
                            print(n in buffer.buffer, n)
                            sample = buffer.retrieve(n, batch['mlm_inputs'])
                            print("here", sample)
                            if sample is not None:
                                contexts.append(sample.to(device))

                    assert len(contexts) == batch['mlm_inputs']['input_ids'].shape[0], "Context has {} elements when it should have {}".format(len(contexts), batch['mlm_inputs']['input_ids'].shape[0])

                    batch['contexts'] = contexts

                out = test_model(batch)

                loss = out.loss
                print(loss)
                train_new_token = accelerator.gather(out.new_token_loss)

                log_dict['train loss'] = loss.item()
                log_dict['train new token loss'] = train_new_token.mean().item()
                train_losses.append(loss.item())
                train_new_token_losses.append(train_new_token.mean())

            elif args.maml:
                inner_opt = torch.optim.SGD(filter(lambda p: p.requires_grad, test_model.parameters()),
                                            lr=1e-5)

                test_model.store_mem(batch)
                with higher.innerloop_ctx(
                        test_model, inner_opt, copy_initial_weights=False
                ) as (fnet, diffopt):
                    inner_losses = []
                    for _ in range(n_inner_iter):
                        maml_out = test_model.forward_inner(batch)

                        inner_losses.append(maml_out.loss.item())
                        diffopt.step(maml_out.loss)

                    out = test_model(batch)
                    log_dict['average maml inner loss'] = sum(inner_losses) / len(inner_losses)
                    log_dict['maml outer loss'] =  out.loss.item()
                    #out.loss.backward()
                    loss = out.loss

            if "sanity" in args.data_path:
                nonce_loss = get_nonce_loss(batch, out, test_model.secondLM.vocab_size, device)
                if nonce_loss:
                    log_dict["new token loss"] = nonce_loss.item()
            elif "snli" in args.data_path:
                preds = out.logits
                preds = F.log_softmax(preds, dim=-1).argmax(dim=1)
                true_ans = batch['task_labels'].to(device).view(-1)
                num_correct = (preds == true_ans).sum()
                train_correct += num_correct
                train_total += batch['task_labels'].shape[0]


            # final_loss.backward()
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(test_model.parameters(), 1.0)
            # torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, test_model.parameters()), 1.0)
            for name, param in test_model.module.named_parameters():
                if param.grad is not None and param.requires_grad:
                    log_dict["gradients/post_{}_grad_norm".format(name)] = torch.norm(param.grad.view(-1)).item()
                    if torch.isnan(torch.norm(param.grad.view(-1))):
                        raise Exception("Nan Gradient")

            # if args.taskName == "addition":
                # for ind, val in enumerate(batch['generationTokens']):
                #     idx = deepcopy(val)
                #     gen_ans = test_model.generate(idx, 10)
                #     gen_ans = tokenizerTask.decode(gen_ans['input_ids'][0], skip_special_tokens=True,
                #                                    clean_up_tokenization_spaces=True)
                #     true_ans = tokenizerTask.decode(batch['task_inputs']['input_ids'][ind, 0, :], skip_special_tokens=True,
                #                                     clean_up_tokenization_spaces=True)
                #     train_total += 1
                #     train_correct += compute_exact_match(gen_ans, true_ans)

            opt.step()
            scheduler.step()
            log_dict['num_words_seen'] = len(buffer.buffer)
            norms = []
            for k in test_model.module.memory.memory:
                with torch.no_grad():
                     norms.append(test_model.module.memory.retrieve(k).norm())

            with torch.no_grad():
                log_dict["embed_norms/token embedding norm"] = torch.stack(norms).mean()

            with torch.no_grad():
                log_dict['embed_norms/cls_token_norm'] = test_model.module.cls_token.norm()

            test_model.module.memory.memory = {}
            if args.taskName == "online" and not args.prefill:
                buffer.store(batch['mlm_inputs'].to(device))
                buffer.cleanup()
            accelerator.log(log_dict)

            if i != 0 and (i % eval_ind == 0 or i % len(train_dl) == 0):
                opt.zero_grad(set_to_none=True)
                test_model.eval()
                with torch.no_grad():
                    if "chimera" in args.data_path:
                        corrs = []
                        for b in test_dl:
                            t_out= test_model.forward(b)
                            new_w = test_model.get_new_weights(batch, task="MLM").to(device)

                            indices = b['eval_indices']
                            sims = []
                            for probe in b['probe_inputs']:
                                p_sims = []
                                p_s = b['probe_inputs'][probe]['sentences']

                                for p_idx, s in enumerate(p_s):
                                    enc = tokenizerTask.encode_plus(s[0], return_tensors='pt').to(device)
                                    p_ind = indices[p_idx]
                                    locs = get_locs(s[0], p_ind.item(), tokenizerTask)

                                    p_emb = get_hidden_states(enc, locs, test_model.secondLM, [-1])
                                    n_emb = get_emb(t_out.hidden_states, locs, [-1], p_idx)
                                    p_sims.append(cos(n_emb, p_emb).item())

                                sim = sum(p_sims) / len(p_sims)
                                sims.append(sim)

                            ratings = [float(v) for v in b['ratings'][0].split(',')]
                            corr = stats.spearmanr(sims, ratings)
                            accelerator.log({'test_point_correlation': corr.correlation})
                            corrs.append(corr.correlation)

                        test_model.module.memory.memory = {}

                        avg_corr = sum(corrs) / len(corrs)
                        accelerator.log({'epoch': epoch, 'Correlation on Test': avg_corr})

                        if avg_corr > best_corr:
                            # chkpt_name = get_model_name_checkpoint(save_folder + test_model.model_name, epoch)
                            # print(chkpt_name)
                            # save(test_model, opt, chkpt_name, accelerator)
                            accelerator.wait_for_everyone()
                            save_dir = save_check_dir + save_suffix.format(save_ind)
                            save_ind += 1
                            accelerator.save_state(output_dir = save_dir)
                            best_corr = avg_corr

                    elif "sanity" in args.data_path:
                        test_model.eval()
                        test_losses = []
                        test_nonce_losses = []
                        for b in test_dl:
                            t_out = test_model.forward(b)
                            accelerator.log({'test point loss': t_out.loss.item()})
                            test_nonce_loss = get_nonce_loss(b, t_out, test_model.secondLM.vocab_size, device)
                            accelerator.log({"test loss on nonce tokens": test_nonce_loss.item()})
                            test_nonce_losses.append(test_nonce_loss.item())

                            test_losses.append(t_out.loss.item())

                        accelerator.log({'epoch': epoch, 'average test loss': sum(test_losses) / len(test_losses),
                                   'average test nonce loss': sum(test_nonce_losses) / len(test_nonce_losses)})
                        n_loss = sum(test_nonce_losses) / len(test_nonce_losses)

                        if n_loss < best_loss:
                            # chkpt_name = get_model_name_checkpoint(save_folder + test_model.model_name, epoch)
                            # save(test_model, opt, chkpt_name, accelerator)
                            accelerator.wait_for_everyone()
                            save_dir = save_check_dir + save_suffix.format(save_ind)
                            save_ind += 1
                            accelerator.save_state(output_dir=save_dir)
                            best_loss = n_loss

                    elif "squad" in args.data_path:
                        test_model.eval()
                        test_losses = []
                        for b in test_dl:
                            t_out = test_model.forward(b)

                            test_losses.append(t_out.loss.item())
                            test_model.module.memory.memory = {}

                        avg_test = sum(test_losses) / len(test_losses)
                        if avg_test < best_loss:
                            # chkpt_name = get_model_name_checkpoint(save_folder + test_model.module.model_name, epoch)
                            # save(test_model, opt, chkpt_name, accelerator)
                            accelerator.wait_for_everyone()
                            save_dir = save_check_dir + save_suffix.format(save_ind)
                            save_ind += 1
                            accelerator.save_state(output_dir=save_dir)
                            best_loss = avg_test

                        accelerator.log({'epoch': epoch, 'average test loss': avg_test})

                    elif "snli" in args.data_path:
                        test_model.eval()
                        total_correct = 0
                        total = 0
                        for b in test_dl:
                            t_out = test_model.forward(b)
                            preds = t_out.logits
                            preds = F.log_softmax(preds, dim=-1).argmax(dim=1)
                            true_ans = b['task_labels'].to(device).view(-1)
                            num_correct = (preds == true_ans).sum()
                            total_correct += num_correct
                            total += b['task_labels'].shape[0]
                            test_model.module.memory.memory = {}
                        acc = total_correct / total
                        accelerator.log({'epoch': epoch, 'average test accuracy': acc})
                        if best_acc < acc:
                            # chkpt_name = get_model_name_checkpoint(save_folder + test_model.module.model_name, epoch)
                            # save(test_model, opt, chkpt_name, accelerator)
                            # print("Saved {}".format(chkpt_name))
                            accelerator.wait_for_everyone()
                            save_dir = save_check_dir + save_suffix.format(save_ind)
                            save_ind += 1
                            accelerator.save_state(output_dir=save_dir)
                            best_acc = acc

                    elif "addition" in args.taskName:
                        test_matches = 0
                        test_total = 0
                        test_losses = []
                        for b in test_dl:
                            t_out = test_model.forward(b)

                            test_losses.append(t_out.loss.item())

                            for ind, val in enumerate(b['generationTokens']):
                                idx = deepcopy(val)
                                gen_ans = test_model.generate(idx, 10)
                                gen_ans = tokenizerTask.decode(gen_ans['input_ids'][0], skip_special_tokens=True,
                                                               clean_up_tokenization_spaces=True)
                                true_ans = tokenizerTask.decode(b['task_inputs']['input_ids'][ind, 0, :],
                                                                skip_special_tokens=True, clean_up_tokenization_spaces=True)
                                test_total += 1
                                test_matches += compute_exact_match(gen_ans, true_ans)

                            test_model.module.memory.memory = {}
                        avg_test = sum(test_losses) / len(test_losses)
                        avg_match = test_matches / test_total
                        accelerator.log({'epoch': epoch, 'average test loss': avg_test, "test exact match": avg_match})
                        if avg_test < best_loss:
                            # chkpt_name = get_model_name_checkpoint(save_folder + test_model.model_name, epoch)
                            # save(test_model, opt, chkpt_name, accelerator)
                            # print("Saved {}".format(chkpt_name))
                            accelerator.wait_for_everyone()
                            save_dir = save_check_dir + save_suffix.format(save_ind)
                            save_ind += 1
                            accelerator.save_state(output_dir=save_dir)
                            best_loss = avg_test

                    elif args.taskName == "online":
                        test_model.eval()
                        test_losses = []
                        # test_buffer = RetrievalBuffer(15, args.num_examples, new_toks, tokenizerMLM, args.random_ex,
                        #                          args.cat)
                        test_nonce_losses = []
                        # for b in test_dl:
                        #     if accelerator.is_local_main_process:
                        #         test_buffer.store(b['mlm_inputs'].to(device))
                        for b in test_dl:
                            contexts = []
                            for j in range(b['mlm_inputs']['input_ids'].shape[0]):
                                to_sample = [n for n in test_buffer.nonces if n in b['mlm_inputs']['input_ids']]
                                for n in to_sample:
                                    sample = test_buffer.retrieve(n, b['mlm_inputs'])
                                    if sample is not None:
                                        contexts.append(sample)
                            t_out = test_model(b)
                            all_losses = accelerator.gather(t_out.loss)
                            test_losses.append(all_losses)
                            test_model.module.memory.memory = {}
                            all_new_tokens = accelerator.gather(t_out.new_token_loss)
                            test_nonce_losses.append(all_new_tokens)

                        # test_losses = torch.cat(test_losses, dim=0)
                        avg_test = torch.stack(test_losses).mean()
                        avg_new_tok = torch.stack(test_nonce_losses).mean()
                        accelerator.log({'epoch': epoch, 'average test loss': avg_test, "average test loss on new tokens": avg_new_tok})

                        if avg_test < best_loss:
                            # chkpt_name = get_model_name_checkpoint(save_folder + test_model.module.model_name, eval_ind)
                            # print(chkpt_name)
                            # save(test_model, opt, chkpt_name, accelerator)
                            # print("Saved {}".format(chkpt_name))
                            accelerator.wait_for_everyone()
                            save_dir = save_check_dir + save_suffix.format(save_ind)
                            save_ind += 1
                            accelerator.save_state(output_dir=save_dir)
                            best_loss = avg_test

        accelerator.log({"epoch": epoch, 'average train loss': sum(train_losses) / len(train_losses),
                         "average train new token loss": sum(train_new_token_losses) / len(train_new_token_losses)})
        if "snli" in args.data_path:
            accelerator.log({"epoch": epoch, 'average train acc': train_correct / train_total})
        # if args.taskName == "addition":
        #     wandb.log({'epoch': epoch,
        #                'train exact match': train_correct / train_total})
    accelerator.end_training()

if __name__ == "__main__":
    main()
