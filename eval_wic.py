from train_with_llama import *
from wic.wic_tsv.read_wic_tsv import *
import json
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import wandb
from torch.optim import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

class CoLLEGeEmbeddingModel(nn.Module):
    def __init__(self, firstLM, num_new_tokens, layers, mask_token_id, memory_config, num_layers,
                 distillation_temp, use_pos=False):
        super().__init__()
        
        self.layers = layers
        self.mask_token_id = mask_token_id
        self.firstLM = firstLM
        self.memory_config = memory_config

        self.num_new_tokens = num_new_tokens
        self.num_layers = num_layers
        self.distillation_temp = distillation_temp

        self.emb_gen = EmbeddingGenerator(self.firstLM, 4096, num_layers, config=self.memory_config, use_pos=use_pos)
        
        # with torch.no_grad():
        #     # firstLM_mean_embed = torch.mean(self.firstLM.get_output_embeddings().weight[:self.initial_first_ind, :], dim=0)
        #     output_mean_embed = torch.mean(
        #         self.secondLM.get_output_embeddings().weight.norm(dim=1))
        #     # firstLM_std = torch.std(self.firstLM.get_output_embeddings().weight[:self.initial_first_ind, :], dim=0)
        #     input_mean_embed = torch.mean(
        #         self.secondLM.get_input_embeddings().weight.norm(dim=1))

        #     self.emb_gen.init_weights(input_mean_embed, output_mean_embed)

        #     torch.register_buffer("firstLM_mean_embed", self.firstLM_mean_embed)
        #     torch.register_buffer("secondLM_mean_embed", self.secondLM_mean_embed)

        # with torch.no_grad():
        #     self.firstLM.get_input_embeddings().weight.data[self.first_list, :] = 0.
        #     self.secondLM.get_input_embeddings().weight[self.second_list, :] = 0.
        #     self.secondLM.get_output_embeddings().weight[self.second_list] = 0.

        self.freeze()

    @property
    def first_list(self):
        return list(range(self.firstLM.config.vocab_size, self.firstLM.config.vocab_size + self.num_new_tokens))
    
    @property
    def initial_first_ind(self):
        # vocab size + num new tokens - num new tokens
        return self.firstLM.config.vocab_size

    def freeze(self):
        for parameter in self.firstLM.parameters():
            parameter.requires_grad = False

    def swap_with_mask(self, inputs):
        inp = inputs.clone()
        for nonce in self.first_list:
            inp[inp == nonce] = self.mask_token_id
        return inp


    def get_college_embeddings(self, contexts):
        
        input_embeds, output_embeds, college_embeds = [], [], []
        for i, c in enumerate(contexts):
            new_token = c['input_ids'][
                                torch.isin(c['input_ids'], torch.tensor(self.first_list, device=c['input_ids'].device))].unique()[
                                0].item()
            mlm_ids = self.swap_with_mask(c['input_ids'])
            with torch.no_grad():
                first_out = self.firstLM(input_ids=mlm_ids, attention_mask=c['attention_mask'],
                                         output_hidden_states=True)

            first_hidden = first_out.hidden_states
            combined = combine_layers(first_hidden, self.layers)

            if len(combined.shape) == 2:
                combined = combined.unsqueeze(0)

            attn = c['attention_mask']
            embed_inputs = combined

            inp_embs, out_embs, college_embs = self.emb_gen.get_embeds(embed_inputs, attn)
            input_embeds.append(inp_embs)
            output_embeds.append(out_embs)
            college_embeds.append(college_embs)

        return input_embeds, output_embeds, college_embeds

class CoLLEGeWiCClassifier(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.classifier = nn.Linear(hidden_size * 2, 1)
    
    def forward(self, x, labels = None):
        logits = self.classifier(x)
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
        else:
            loss = None
        return logits, loss
# def predict_example(context, definition, model, tokenizer):
#     ctx_tok = tokenizer([context], return_tensors = 'pt').to("cuda")
#     def_tok = tokenizer([definition], return_tensors = 'pt').to("cuda")
#     input_embeds, output_embeds, college_embeds = model.get_college_embeddings([ctx_tok, def_tok])
#     cos = nn.CosineSimilarity()
#     input_cos = cos(input_embeds[0], input_embeds[1])
#     output_cos = cos(output_embeds[0], output_embeds[1])
#     college_cos = cos(college_embeds[0], college_embeds[1])

#     return input_cos, output_cos, college_cos

def predict_example(context, definition, model, tokenizerMLM, tokenizerTask, new_token_idx):
    ctx_tok = tokenizerMLM([context], return_tensors = 'pt').to("cuda")
    def_tok = tokenizerMLM([definition], return_tensors = 'pt').to("cuda")
    
    query_inputs = tokenizerTask([context, definition], return_tensors='pt', padding='longest').to("cuda")
    # input_embeds, output_embeds, college_embeds = model.get_college_embeddings([ctx_tok, def_tok])
    ctx_token_idx = torch.where(query_inputs['input_ids'][0] == new_token_idx)[0]
    def_token_idx = torch.where(query_inputs['input_ids'][1] == new_token_idx)[0]
    print(ctx_token_idx, def_token_idx)
    print(query_inputs['input_ids'])
    labels = query_inputs['input_ids'].clone()
    batch = {
        "contexts": [ctx_tok, def_tok],
        "input_ids": query_inputs['input_ids'],
        "attention_mask": query_inputs['attention_mask'],
        'labels': labels
    }
    outputs = model(batch, output_hidden_states=True)
    ctx_new_token_hidden = outputs.hidden_states[-1][0, ctx_token_idx, :]
    def_new_token_hidden = outputs.hidden_states[-1][1, def_token_idx, :]
    # cos = nn.CosineSimilarity()
    # input_cos = cos(input_embeds[0], input_embeds[1])
    # output_cos = cos(output_embeds[0], output_embeds[1])
    # college_cos = cos(college_embeds[0], college_embeds[1])

    return ctx_new_token_hidden, def_new_token_hidden


def compute_metrics(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    res = (preds == labels).astype(type(labels[0]))
    precision, r, f1, _ = precision_recall_fscore_support(y_true=labels, y_pred=preds, average='binary')
    acc = res.mean()
    return {
        "acc": acc,
        "F_1": f1,
        "P": precision,
        "R": r,
    }




if __name__ == "__main__":
    device = "cuda"

    path = "model_checkpoints/layers/no_mp/llama/input_and_output/filtered/redone_pile/layernorm/roberta-large/1_layers/last_1/32_batch_size/mean_agg/1_examples/lr_0.001/weight_decay_0.1/with_negatives_and_regression/distillation_weight_0.05_temp_3/output_embedding_cosine/checkpoints/checkpoint_7_28000"

    firstLM = RobertaForMaskedLM.from_pretrained("roberta-large", low_cpu_mem_usage=True).to(device)
    secondLM = LlamaForCausalLM.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf", low_cpu_mem_usage=True)
    tokenizerTask = LlamaTokenizer.from_pretrained(path + "tokenizerTask", use_fast=False, legacy=True)
    tokenizerMLM = AutoTokenizer.from_pretrained(path + "/tokenizerMLM", use_fast=False)
    memory_config = AggregatorConfig()
    nonces = list(tokenizerTask.get_added_vocab().keys())
    mask_token_id = tokenizerMLM.mask_token_id
    layers=[-1]
    epochs = 10
    lr = 1e-4
    weight_decay = 0.05
    batch_size = 10
    # model = CoLLEGeEmbeddingModel(
    #     firstLM=firstLM,
    #     num_new_tokens=1,
    #     layers=layers,
    #     mask_token_id=mask_token_id,
    #     memory_config=memory_config,
    #     num_layers=1,
    #     distillation_temp=0.3,
    #     use_pos=False,
    # )
    model = MorphMemoryModelLLAMA(firstLM, secondLM, len(nonces), layers, mask_token_id, memory_config, 1, None).to(device)
    model.emb_gen.load_state_dict(torch.load(path + "/pytorch_model.bin"))
    model = model.to(device)

    new_token_idx = len(tokenizerTask) - 1
    # tokenizerMLM.add_tokens(["<nonce>"])
    print(model.first_list)
    model.eval()
    thresholds = np.linspace(0, 1, num=100).tolist()

    classifier = CoLLEGeWiCClassifier(secondLM.config.hidden_size).to(device)
    for p in model.parameters():
        p.requires_grad = False

    run_name = "classifier_epochs={}_lr={}_weight_decay={}".format(epochs, lr, weight_decay)
    run = wandb.init(project="few_shot_wic",
                     name=run_name
                    )
    
    # results = {}
    # for e in ["input", "output", "college"]:
    #     results[e] = {"true pos": [0 for i in range(len(thresholds))],
    #                   "false pos": [0 for i in range(len(thresholds))],
    #                   "true neg": [0 for i in range(len(thresholds))],
    #                   "false neg": [0 for i in range(len(thresholds))]}
    # results = {"true pos": [0 for i in range(len(thresholds))],
    #                   "false pos": [0 for i in range(len(thresholds))],
    #                   "true neg": [0 for i in range(len(thresholds))],
    #                   "false neg": [0 for i in range(len(thresholds))]}
    

    train_folder = Path('wic/wic_tsv/data/en/Training')
    dev_folder = Path('wic/wic_tsv/data/en/Training')
    contexts, target_inds, hypernyms, definitions, labels = dp.read_wic_tsv(wic_tsv_folder=train_folder)
    print("Ratio of positives:", sum(labels) / len(labels))
    dev_contexts, dev_target_inds, dev_hypernyms, dev_definitions, dev_labels = dp.read_wic_tsv(wic_tsv_folder=dev_folder)

    opt = AdamW(params=classifier.parameters(),
                lr=lr,
                weight_decay=weight_decay)
    
    scheduler = get_cosine_schedule_with_warmup(opt, 
                                                num_warmup_steps = 300,
                                                num_training_steps = len(contexts) * epochs)

    for i, (context, target_id, definition) in enumerate(zip(contexts, target_inds, definitions)):
        word = context.split()[target_id]
        
        contexts[i] = re.sub(r"\b({})\b".format(word), "<nonce>", context, flags=re.I)
        definitions[i] = re.sub(r"\b({})\b".format(word), "<nonce>", definition, flags=re.I)
    
    for i, (context, target_id, definition) in enumerate(zip(dev_contexts, dev_target_inds, dev_definitions)):
        word = context.split()[target_id]
        
        dev_contexts[i] = re.sub(r"\b({})\b".format(word), "<nonce>", context, flags=re.I)
        dev_definitions[i] = re.sub(r"\b({})\b".format(word), "<nonce>", definition, flags=re.I)
    
    global_step = 0
    for epoch in range(epochs):
        train_predictions = []
        train_labels = []
        train_loss = 0
        classifier.train()

        curr_train_batch_size = 0
        for context, definition, label in tqdm(zip(contexts, definitions, labels), total=len(contexts)):
            print(context, definition)
            def_str = "The word <nonce> is defined as {}".format(definition)
            with torch.no_grad():
                ctx_hidden, def_hidden = predict_example(context, def_str, model, tokenizerMLM, tokenizerTask, new_token_idx)
                print("def", def_hidden.shape)
                print("ctx", ctx_hidden.shape)
                if def_hidden.shape[0] > 1:
                    def_hidden = torch.mean(def_hidden, dim=0, keepdim=True)
                if ctx_hidden.shape[0] > 1:
                    ctx_hidden = torch.mean(ctx_hidden, dim=0, keepdim=True)
                cat_embeds = torch.cat([ctx_hidden, def_hidden], dim=1)
                if curr_train_batch_size == 0:
                    batch_inputs = cat_embeds
                    batch_labels = torch.tensor([label], device=device).unsqueeze(0)
                else:
                    batch_inputs = torch.cat([batch_inputs, cat_embeds], dim=0)

                    ex_labels = torch.tensor([label], device=device).unsqueeze(0)
                    batch_labels = torch.cat([batch_labels, ex_labels], dim=0)
                    curr_train_batch_size += 1
            if curr_train_batch_size == batch_size:

                logits, loss = classifier(batch_inputs, labels=batch_labels)
                wandb.log({"train loss": loss.item(),
                        "global step": global_step})
                loss.backward()
                opt.step()
                scheduler.step()
                opt.zero_grad()
                model.zero_grad()

                preds = torch.flatten((logits >= 0.5).int()).detach().tolist()
                train_predictions += preds
                train_loss += loss.detach().float()
                train_labels += torch.flatten(ex_labels).detach().tolist()

                global_step += 1
        
        test_predictions = []
        test_labels = []
        test_loss = 0
        classifier.eval()
        for dev_context, dev_definition, dev_label in tqdm(zip(dev_contexts, dev_definitions, dev_labels), total=len(contexts)):
            def_str = "The word <nonce> is defined as {}".format(dev_definition)
            with torch.no_grad():
                ctx_hidden, def_hidden = predict_example(dev_context, def_str, dev_definition, model,
                                                         tokenizerMLM, tokenizerTask, new_token_idx)
                cat_embeds = torch.cat([ctx_hidden, def_hidden], dim=1)
                logits, loss = classifier(cat_embeds, labels=torch.tensor([label], device=device))
                test_preds = torch.flatten((logits >= 0.5).int()).detach().tolist()
                test_loss += loss.detach().float()
                test_predictions += test_preds
                test_labels.append(dev_label)
        

        avg_test_loss = test_loss / len(dev_contexts)
        avg_train_loss = train_loss / len(contexts)
        train_metrics = compute_metrics(train_predictions, train_labels)
        test_metrics = compute_metrics(test_predictions, test_labels)

        epoch_log_dict = {
            "average train loss": avg_train_loss,
            "average test loss": avg_test_loss,
            "epoch": epoch
        }

        for key in train_metrics:
            epoch_log_dict["train_{}".format(key)] = train_metrics[key]
        
        for key in test_metrics:
            epoch_log_dict["test_{}".format(key)] = train_metrics[key]
        
        wandb.log(epoch_log_dict)

            # ex_results = []
            # for j, threshold in enumerate(thresholds):
            #     cos = nn.CosineSimilarity()
            #     sim = cos(ctx_hidden, def_hidden)
            #     if len(sim) > 1:
            #         pred = int(sim.max().item() >= threshold)
            #     else:
            #         pred = int(sim.item() >= threshold)
            #     # int(sims[i].item() >= threshold)
            #     print(sim, threshold, pred, label)
            #     if pred == label:
            #         if pred == 1:
            #             results["true pos"][j] += 1
            #         elif pred == 0:
            #             results["true neg"][j] += 1
                
            #     else:
            #         if pred == 1:
            #             results["false pos"][j] += 1
            #         elif pred == 0:
            #             results["false neg"][j] += 1
    
    # for key in results:
    #     print("Results for {} Embeddings".format(key.upper()))
    #     emb_results = results[key]
    # precisions = []
    # accs = []
    # f1s = []
    # recalls = []

    # for i in range(len(thresholds)):
    #     true_pos = results['true pos'][i]
    #     true_neg = results['true neg'][i]
    #     false_pos = results['false pos'][i]
    #     false_neg = results['false neg'][i]
    #     try:
    #         precision = true_pos / (true_pos + false_pos)
    #     except ZeroDivisionError:
    #         precision = 0
        
    #     try:
    #         recall = true_pos / (true_pos + false_neg)
    #     except ZeroDivisionError:
    #         recall = 0
    #     try:   
    #         f1 = 2 * (precision * recall) / (precision + recall)
    #     except ZeroDivisionError:
    #         f1 = 0
    #     acc = (true_pos + true_neg) / len(contexts)

    #     precisions.append(precision)
    #     accs.append(acc)
    #     f1s.append(f1)
    #     recalls.append(recall)

    # print("Thresholds: ", thresholds)
    # print("Precisions: ", precisions)
    # print("Recalls: ", recalls)
    # print("F1 Scores: ", f1s)
    # print("Accuracies: ", accs)

    # with open("college_wic_results.json", 'w') as fp:
    #     json.dump(results, fp)
    
