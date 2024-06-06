from train_with_llama import *
from wic.wic_tsv.read_wic_tsv import *
import json
from tqdm import tqdm
import torch.nn as nn
import numpy as np

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
    
    query_inputs = tokenizerTask([context, definition], return_tensors='pt', padding='longest')
    # input_embeds, output_embeds, college_embeds = model.get_college_embeddings([ctx_tok, def_tok])
    ctx_token_idx = torch.where(query_inputs['input_ids'][0] == new_token_idx)
    def_token_idx = torch.where(query_inputs['input_ids'][0] == new_token_idx)
    labels = query_inputs['input_ids'].clone()
    batch = {
        "contexts": [ctx_tok, def_tok],
        "input_ids": query_inputs['input_ids'],
        "attention_mask": query_inputs['attention_mask'],
        'labels': labels
    }
    outputs = model(batch, output_hidden_states=True)
    ctx_new_token_hidden = outputs.hidden_states[-1][0, ctx_token_idx, :]
    def_new_token_hidden = outputs.hidden_states[-1][1, def_token_idx :]
    cos = nn.CosineSimilarity()
    # input_cos = cos(input_embeds[0], input_embeds[1])
    # output_cos = cos(output_embeds[0], output_embeds[1])
    # college_cos = cos(college_embeds[0], college_embeds[1])

    return cos(ctx_new_token_hidden, def_new_token_hidden)




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
    # tokenizerMLM.add_tokens(["<nonce>"])
    print(model.first_list)
    model.eval()
    thresholds = np.linspace(0, 1, num=100).tolist()
    results = {}
    for e in ["input", "output", "college"]:
        results[e] = {"true pos": [0 for i in range(len(thresholds))],
                      "false pos": [0 for i in range(len(thresholds))],
                      "true neg": [0 for i in range(len(thresholds))],
                      "false neg": [0 for i in range(len(thresholds))]}
    
    

    train_folder = Path('wic/wic_tsv/data/en/Training')
    contexts, target_inds, hypernyms, definitions, labels = dp.read_wic_tsv(wic_tsv_folder=train_folder)
    print("Ratio of positives:", sum(labels) / len(labels))

    for i, (context, target_id, definition) in enumerate(zip(contexts, target_inds, definitions)):
        word = context.split()[target_id]
        
        contexts[i] = re.sub(r"\b({})\b".format(word), "<nonce>", context, flags=re.I)
        definitions[i] = re.sub(r"\b({})\b".format(word), "<nonce>", definition, flags=re.I)

    for context, definition, label in tqdm(zip(contexts, definitions, labels), total=len(contexts)):
        print(context, definition)
        def_str = "The word <nonce> is defined as {}".format(definition)
        sims = predict_example(context, def_str, model, tokenizerMLM)

        ex_results = []
        for j, threshold in enumerate(thresholds):
            for i, e in enumerate(["input", "output", "college"]):
                pred = int(sims[i].item() >= threshold)
                print(sims[i], sims[i].item(), threshold, pred, label)
                if pred == label:
                    if pred == 1:
                        results[e]["true pos"][j] += 1
                    elif pred == 0:
                        results[e]["true neg"][j] += 1
                
                else:
                    if pred == 1:
                        results[e]["false pos"][j] += 1
                    elif pred == 0:
                        results[e]["false neg"][j] += 1
    
    for key in results:
        print("Results for {} Embeddings".format(key.upper()))
        emb_results = results[key]
        precisions = []
        accs = []
        f1s = []
        recalls = []

        for i in range(len(thresholds)):
            true_pos = emb_results['true pos'][i]
            true_neg = emb_results['true neg'][i]
            false_pos = emb_results['false pos'][i]
            false_neg = emb_results['false neg'][i]
            try:
                precision = true_pos / (true_pos + false_pos)
            except ZeroDivisionError:
                precision = 0
            
            try:
                recall = true_pos / (true_pos + false_neg)
            except ZeroDivisionError:
                recall = 0
            try:   
                f1 = 2 * (precision * recall) / (precision + recall)
            except ZeroDivisionError:
                f1 = 0
            acc = (true_pos + true_neg) / len(contexts)

            precisions.append(precision)
            accs.append(acc)
            f1s.append(f1)
            recalls.append(recall)

        print("Thresholds: ", thresholds)
        print("Precisions: ", precisions)
        print("Recalls: ", recalls)
        print("F1 Scores: ", f1s)
        print("Accuracies: ", accs)

        with open("college_wic_results.json", 'w') as fp:
            json.dump(results, fp)
    
