from argparse import ArgumentParser

import torch
from datasets import load_from_disk
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.dataloader import default_collate

from transformers import RobertaForMaskedLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, \
    get_linear_schedule_with_warmup, AdamW, DataCollatorForLanguageModeling
import accelerate
from accelerate import init_empty_weights, Accelerator
from accelerate import load_checkpoint_and_dispatch
from transformers.modeling_outputs import CausalLMOutputWithPast

from modules.buffer import RetrievalBuffer
from modules.memory import OnlineProtoNet
from modules.model_outputs import CausalLMOutputWithNewToken
from modules.utils import combine_layers
from train_utils import get_new_token_loss_labels
import os
from configs.config import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from accelerate import DistributedDataParallelKwargs


class EmbeddingGenerator(nn.Module):

    def __init__(self, firstLM, secondLM):
        super().__init__()
        self.firstLM = firstLM
        self.secondLM = secondLM
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.firstLM.config.hidden_size,
                                                   nhead=self.firstLM.config.num_attention_heads,
                                                   activation='relu',
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        
        self.input_emb_head = nn.Linear(self.firstLM.config.hidden_size, self.secondLM.config.hidden_size)
        self.output_emb_head = nn.Linear(self.firstLM.config.hidden_size, self.secondLM.config.hidden_size)

    def forward(self, inputs, attn_mask):

        out = self.encoder(inputs, src_key_padding_mask=~attn_mask.bool())

        mean_pool = torch.sum(out * attn_mask.unsqueeze(-1), dim=1) / torch.sum(attn_mask, dim=-1, keepdim=True)

        inp_embeds = self.input_emb_head(mean_pool)
        out_embeds = self.output_emb_head(mean_pool)

        return inp_embeds, out_embeds


class MorphMemoryModelLLAMA(nn.Module):

    def __init__(self, firstLM, secondLM, num_new_tokens, layers, mask_token_id, memory_config):
        super().__init__()

        self.layers = layers
        self.mask_token_id = mask_token_id
        self.firstLM = firstLM
        self.secondLM = secondLM
        self.memory_config = memory_config
        self.memory = OnlineProtoNet(memory_config)
        self.num_new_tokens = num_new_tokens

        self.emb_gen = EmbeddingGenerator(self.firstLM, self.secondLM)

        initial_first_ind = int(self.firstLM.config.vocab_size - self.num_new_tokens)
        initial_second_ind = int(self.secondLM.config.vocab_size - self.num_new_tokens)

        self.first_list = list(range(initial_first_ind, self.firstLM.config.vocab_size))
        self.second_list = list(range(initial_second_ind, self.secondLM.config.vocab_size))

        self.model_name = "{}_{}".format(self.secondLM.config.model_type,memory_config.agg_method)

        self.emb_gen.apply(self._init_weights)
        self.emb_gen = self.emb_gen
        self.dropout = nn.Dropout(0.2)
        self.freeze()

        self.secondLM.config.vocab_size = self.secondLM.lm_head.weight.shape[0]

        #initialize new token embeddings
        with torch.no_grad():
            m_first = torch.mean(self.firstLM.get_input_embeddings().weight[:initial_first_ind, :], dim=0)
            m_second = torch.mean(self.secondLM.get_input_embeddings().weight[:initial_second_ind, :], dim=0)
            for n_first, n_second in zip(self.first_list, self.second_list):
                with torch.no_grad():
                    self.firstLM.get_input_embeddings().weight[n_first, :] = m_first
                    self.secondLM.get_input_embeddings().weight[n_second, :] = m_second

    def freeze(self):
        for parameter in self.firstLM.parameters():
            parameter.requires_grad = False

        for parameter in self.secondLM.parameters():
            parameter.requires_grad = False

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.secondLM.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.secondLM.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def swap_with_mask(self, inputs):
        inp = inputs.clone()
        for nonce in self.first_list:
            inp[inp == nonce] = self.mask_token_id
        return inp

    def get_new_output_weights(self, memory):
        w = self.secondLM.lm_head.weight.clone()
        n, hidden = w.shape
        w.requires_grad=True
        msk = torch.zeros_like(w).to(w.device)
        msk2 = torch.zeros_like(w).to(w.device)
        token_mapping = {k: v for k, v in zip(self.first_list, self.second_list)}
        for key in memory.memory:
            msk = msk.scatter(0, torch.tensor([token_mapping[key]]).to(w.device).expand(1, hidden),
                              memory.retrieve(key).to(w.dtype))
            msk2[token_mapping[key], :] = 1.

        return w * (~msk2.bool()) + msk

    def get_new_weights(self, task, memory):

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

        msk = torch.zeros_like(w).to(w.device)
        msk2 = torch.zeros_like(w).to(w.device)
        token_mapping = {k:v for k,v in zip(self.first_list, self.second_list)}
        for key in memory.memory:
            msk = msk.scatter(0, torch.tensor([token_mapping[key]]).to(w.device).expand(1, hidden), memory.retrieve(key).to(w.dtype))
            msk2[token_mapping[key], :] = 1.

        return w * (~msk2.bool()) + msk

    def llama_forward(self, labels, outputs, new_w):
        '''
        Copied from https://github.com/huggingface/transformers/blob/18ee1fe76295239335bf1528c744fe1cfba21cc8/src/transformers/models/llama/modeling_llama.py#L742C7-L742C7
        Note: Output layer weights are not tied to word embedding weights https://github.com/facebookresearch/llama/issues/138
        :param labels:
        :param outputs:
        :return:
        '''
        hidden_states = outputs[0]
        if self.secondLM.config.pretraining_tp > 1:
            lm_head_slices = new_w.split(self.secondLM.vocab_size // self.secondLM.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.secondLM.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = F.linear(hidden_states, new_w, bias=self.secondLM.lm_head.bias)
        logits = logits.float()
        print(logits.shape, "logits")
        print(self.secondLM.lm_head.weight.shape, "lm_logits")
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, new_w.shape[0])
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward(self, batch):
        # nonceMLM = batch["nonceMLM"]
        assert "labels" in batch, "You need labels"
        task_labels = batch["labels"]

        contexts = batch['contexts']

        b_task, l_task = batch["input_ids"].shape

        task_ids = batch['input_ids']
        task_attn = batch["attention_mask"]

        #if 'labels' in batch:
         #   task_labels = task_labels.reshape((b_task * k_task, l_task))

        task_labels = batch['labels']
        outs = []
        assert len(contexts) == b_task
        memories = []
        mem_embeds = []
        for i in range(b_task):
            c = contexts[i].to(self.firstLM.device)
            new_token = c['input_ids'][torch.isin(c['input_ids'], torch.tensor(self.first_list, device=c['input_ids'].device))].unique()[0].item()
            input_memory = OnlineProtoNet(self.memory_config, base_memory=self.memory)
            output_memory = OnlineProtoNet(self.memory_config, base_memory=self.memory)

            mlm_ids = self.swap_with_mask(c['input_ids'])

            with torch.no_grad():
                first_out = self.firstLM(input_ids=mlm_ids, attention_mask=c['attention_mask'],
                                         output_hidden_states=True)

            first_hidden = first_out.hidden_states
            combined = self.dropout(combine_layers(first_hidden, self.layers))

            if len(combined.shape) == 2:
                combined = combined.unsqueeze(0)

            attn = c['attention_mask']
            embed_inputs = combined
            inp_embs, out_embs = self.emb_gen(embed_inputs, attn)

            input_memory.store(new_token, inp_embs)
            output_memory.store(new_token, out_embs)

            new_w = self.get_new_weights(task="Task", memory=input_memory)
            input_embeds = F.embedding(task_ids[i], new_w)
            outputs = self.secondLM.model(
                inputs_embeds=input_embeds.unsqueeze(0),
                attention_mask=task_attn[i].unsqueeze(0),
                output_hidden_states=True
            )
            # print(task_labels[i].shape, "label_shape")
            # print(outputs[0].shape)
            output_weights = self.get_new_output_weights(output_memory)
            llama_outputs = self.llama_forward(task_labels[i], output_weights)
            new_tok_loss = get_new_token_loss_labels(task_labels[i].unsqueeze(0), llama_outputs.logits,
                                                     self.secondLM.lm_head.weight.shape[0],
                                                     torch.tensor(self.second_list, device=llama_outputs.logits.device).unique())
            print("before mem forward")
            mem_embeds.append(input_embeds)
            out_vals = CausalLMOutputWithNewToken(
                loss=llama_outputs.loss,
                logits=llama_outputs.logits,
                past_key_values=llama_outputs.past_key_values,
                hidden_states=llama_outputs.hidden_states,
                attentions=llama_outputs.attentions,
                new_token_loss=new_tok_loss,
            )
            # print("after mem forward")
            outs.append(out_vals)
            # memories.append(memory)

        #         print(outs, "output list")
        final_loss = torch.stack([o.loss for o in outs]).mean()
        final_logits = torch.cat([o.logits for o in outs], dim=0)
        final_hiddens = [o.hidden_states for o in outs]
        final_past_key_values = [o.past_key_values for o in outs]
        final_attentions = [o.attentions for o in outs]
        final_new_token_loss = torch.stack([o.new_token_loss for o in outs]).mean()
        # print("before return")
        return CausalLMOutputWithNewToken(
            loss=final_loss,
            logits=final_logits,
            hidden_states=final_hiddens,
            attentions=final_attentions,
            past_key_values=final_past_key_values,
            new_token_loss=final_new_token_loss
        )
        # task_embeds = torch.stack(mem_embeds)
        # outputs = self.secondLM.model(
        #     inputs_embeds=task_embeds,
        #     attention_mask=task_attn,
        #     output_hidden_states=True
        # )
        #
        # return self.secondLM(
        #     inputs_embeds=task_embeds,
        #     attention_mask=task_attn,
        #     labels=task_labels,
        #     output_hidden_states=True
        # )
        # return self.llama_forward(task_labels, outputs)

# class FewShotLlamaDataset(Dataset):
#     def __init__(self, tokenized_dataset, data_collator):
#         self.tokenized_dataset = tokenized_dataset
#         self.data_collator = data_collator
#
#
#     def __getitem__(self, item):
#         tokenized_input = self.tokenized_dataset[item]
#         return tokenized_input
#
#     def collate(self, batch):
#         out_batch = []
#         out_batch.append([b[0] for b in batch])
#
#         out_batch.append(self.data_collator([b[1] for b in batch]))
#
#         return out_batch

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-6)
    # parser.add_argument("--warmup", type=int, default=1e2)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--memory", type=str, default="mean")
    parser.add_argument("--num_examples", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--word_path", type=str, default='')
    parser.add_argument("--random_ex", action="store_true")
    parser.add_argument("--cat", action="store_true")
    parser.add_argument("--resample", action="store_true")
    parser.add_argument("--prefill", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    return parser


def create_checkpoint_directories(args):

    path = "model_checkpoints/layers/no_mp/llama/input_and_output/{}_batch_size/{}_agg/{}_examples/lr_{}/weight_decay_{}/checkpoints/"
    path = path.format(args.batch_size,args.memory, args.num_examples, args.lr, args.weight_decay)
    os.makedirs(path, exist_ok=True)

    return path


def main():
    def tokenize(ex):
        return tokenizerTask(ex['text'], truncation=True, padding=False, return_tensors=None)

    def tokenize_for_buffer(ex):
        return tokenizerMLM(ex['text'], truncation=True, return_tensors="pt")

    args = get_arguments().parse_args()
    checkpoint_path = create_checkpoint_directories(args)

    print("Arguments: ", args)

    tokenizerMLM = AutoTokenizer.from_pretrained("roberta-large", use_fast=False)
    tokenizerTask = LlamaTokenizer.from_pretrained("/vast/work/public/ml-datasets/llama/tokenizer", legacy=False, use_fast=False)
    tokenizerTask.add_bos_token = True
    tokenizerTask.add_eos_token = True

    tokenizerTask.pad_token = tokenizerTask.unk_token
    word_dict = load_from_disk(args.word_path)
    dataset = load_from_disk(args.data_path)

    words = word_dict['train']['words'] + word_dict['test']['words']
    nonces = list(map(lambda w: "<{}_new>".format(w), words))
    nonces = list(set(nonces))
    tokenizerMLM.add_tokens(nonces)
    tokenizerTask.add_tokens(nonces)
    mask_token_id = tokenizerMLM.mask_token_id

    token_mapping = {v: k for k, v in zip(tokenizerTask.convert_tokens_to_ids(nonces), tokenizerMLM.convert_tokens_to_ids(nonces))}

    #data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizerTask, return_tensors="pt", padding=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizerTask, mlm=False, return_tensors="pt")
    tokenized_train = dataset['train'].map(tokenize, remove_columns=dataset['train'].column_names)
    train_dl = DataLoader(tokenized_train, drop_last=True, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator)

    buffer = RetrievalBuffer(15, args.num_examples, tokenizerMLM.convert_tokens_to_ids(nonces), tokenizerMLM, args.random_ex, args.cat)
    test_buffer = RetrievalBuffer(15, args.num_examples, tokenizerMLM.convert_tokens_to_ids(nonces), tokenizerMLM, args.random_ex, args.cat)

    tokenized_test = dataset['test'].map(tokenize, remove_columns=dataset['train'].column_names)
    test_dl = DataLoader(tokenized_test, shuffle=True, drop_last=True,batch_size=args.batch_size, collate_fn=data_collator)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs])

    # with init_empty_weights():
    print("loading models")
    firstLM = RobertaForMaskedLM.from_pretrained("roberta-large")
    secondLM = LlamaForCausalLM.from_pretrained("/vast/work/public/ml-datasets/llama/hf/llama-7b")

    # firstLM = load_checkpoint_and_dispatch(firstLM, "roberta-large", device_map="auto")
    # secondLM = load_checkpoint_and_dispatch(secondLM, "/vast/work/public/ml-datasets/llama/hf/llama-7b", device_map="auto")

    firstLM.resize_token_embeddings(len(tokenizerMLM), pad_to_multiple_of=64)
    secondLM.resize_token_embeddings(len(tokenizerTask), pad_to_multiple_of=64) # pad for speed
    firstLM.eval()
    secondLM.eval()

    if args.memory == "mean":
        memory_config = AggregatorConfig()
        # weight_decay = 0.05

    # elif args.memory == "rnn":
    #     memory_config = RNNAggConfig()
    #     # weight_decay = 0.015
    # elif args.memory == "cls":
    #     memory_config = TransformerCLSConfig()
    else:
        raise NotImplementedError("This memory aggregation is not implemented")

    model = MorphMemoryModelLLAMA(firstLM, secondLM, len(nonces), [-1], mask_token_id, memory_config)

    ##pad to multiple of 64
    #for param in firstLM:
     #   param.requires_grad=False
    #for param in secondLM:
     #   param.requires_grad = False

    epochs = args.epochs
    lr = args.lr
    epsilon = 1e-8



    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]

    eval_ind = int(len(train_dl) // 3)



    opt = AdamW(optimizer_grouped_parameters,
                    eps=epsilon,
                    lr=lr
                    )
    warmup_steps = int(len(train_dl) * 0.03)
    scheduler = get_linear_schedule_with_warmup(opt, warmup_steps, epochs * len(train_dl))

    model, opt, train_dl, test_dl, scheduler = accelerator.prepare(
        model, opt, train_dl, test_dl, scheduler
    )

    print("loading buffer")
    tokenized_for_buffer = dataset['train'].map(tokenize_for_buffer, remove_columns=dataset['train'].column_names)
    buffer_dl = DataLoader(tokenized_for_buffer.with_format('torch'))
    for inp in buffer_dl:
        buffer.store(inp)

    print("Buffer has {} elements".format(len(buffer.buffer)))

    test_for_buffer = dataset['test'].map(tokenize_for_buffer, remove_columns=dataset['train'].column_names)
    buffer_test_dl = DataLoader(test_for_buffer.with_format('torch'))
    for inp in buffer_test_dl:
        test_buffer.store(inp)

    print("Test buffer has {} elements".format(len(test_buffer.buffer)))

    print("Total nonces = {}".format(len(nonces)))

    accelerator.register_for_checkpointing(opt)
    accelerator.register_for_checkpointing(scheduler)
    checkpoint_id = 0
    accelerator.init_trackers(
        project_name="fewshot_llama",
        config={"num_examples": args.num_examples,
                "learning_rate": lr,
                "aggregation": memory_config.agg_method,
                "batch_size": args.batch_size,
                },
    )

    for epoch in range(epochs):
        train_new_token_losses = []
        train_losses = []
        for i, batch in enumerate(train_dl):

            log_dict = {}

            model.train()
            model.module.firstLM.eval()
            model.module.secondLM.eval()
            model.zero_grad()
            opt.zero_grad()
            contexts = []
            for j in range(batch['input_ids'].shape[0]):
                to_sample = list(set([n for n in buffer.nonces if token_mapping[n] in batch['input_ids'][j]]))
                assert (len(to_sample) == 1)

                for n in to_sample:
                    sample = buffer.retrieve(n, batch)
                    if sample is not None:
                        contexts.append(sample)
                    else:
                        print("Null context for {}".format(n))

            assert len(contexts) == batch['input_ids'].shape[0], "Context has {} elements when it should have {}".format(len(contexts), batch['input_ids'].shape[0])
            batch['contexts'] = contexts
            # print(batch['input_ids'].shape[0])
            out = model(batch)
            loss = out.loss
            # print(loss)
            train_new_token = accelerator.gather(out.new_token_loss)
            log_dict['train loss'] = loss.item()
            log_dict['train new token loss'] = train_new_token.mean().item()
            train_losses.append(loss.item())
            train_new_token_losses.append(train_new_token.mean())
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
            for name, param in model.module.named_parameters():
                if param.grad is not None and param.requires_grad:
                    log_dict["gradients/post_{}_grad_norm".format(name)] = torch.norm(param.grad.view(-1)).item()
                    if torch.isnan(torch.norm(param.grad.view(-1))):
                        raise Exception("Nan Gradient for {}".format(name))

            opt.step()
            scheduler.step()
            log_dict['num_words_seen'] = len(buffer.buffer)

            # norms = []
            #for m in out.memories:
            #    new_ids = list(m.memory.keys())
            #    assert len(new_ids) == 1
            #    new_id = new_ids[0]
            #    with torch.no_grad():
            #        norms.append(m.retrieve(new_id).norm())

            #with torch.no_grad():

             #   log_dict["embed_norms/token embedding norm"] = torch.stack(norms).mean()

            accelerator.log(log_dict)

            if i != 0 and (i % eval_ind == 0 or i % len(train_dl) == 0):
                opt.zero_grad(set_to_none=True)
                model.eval()
                with torch.no_grad():
                    test_losses = []
                    test_nonce_losses = []

                    for b in test_dl:
                        contexts = []
                        for j in range(b['input_ids'].shape[0]):
                            to_sample = list(set([n for n in test_buffer.nonces if token_mapping[n] in b['input_ids'][j]]))
                            assert (len(to_sample) == 1)

                            for n in to_sample:
                                sample = test_buffer.retrieve(n, b)
                                if sample is not None:
                                    contexts.append(sample)
                        assert len(contexts) == b['input_ids'].shape[0], "Context has {} elements when it should have {}".format(len(contexts), b['input_ids'].shape[0])
                        b['contexts'] = contexts
                        t_out = model(b)
                        all_losses = accelerator.gather(t_out.loss)
                        test_losses.append(all_losses)
                        model.module.memory.memory = {}
                        all_new_tokens = accelerator.gather(t_out.new_token_loss)
                        test_nonce_losses.append(all_new_tokens)

                    avg_test = torch.stack(test_losses).mean()
                    avg_new_tok = torch.stack(test_nonce_losses).mean()
                    accelerator.log(
                        {'epoch': epoch, "eval_step": i // eval_ind, 'average test loss': avg_test, "average test loss on new tokens": avg_new_tok})

                    accelerator.wait_for_everyone()
                    save_dir = checkpoint_path + "checkpoint_{}".format(checkpoint_id)
                    os.makedirs(save_dir, exist_ok=True)
                    accelerator.save_state(save_dir)
                    checkpoint_id += 1

    accelerator.end_training()

if __name__ == "__main__":
    main()

