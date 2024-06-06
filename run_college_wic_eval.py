from eval_wic import *

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str)
    return parser

if __name__ == "__main__":
    args = get_arguments().parse_args()
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

    model = MorphMemoryModelLLAMA(firstLM, secondLM, len(nonces), layers, mask_token_id, memory_config, 1, None).to(device)
    model.emb_gen.load_state_dict(torch.load(path + "/pytorch_model.bin"))
    model = model.to(device)

    classifier_path = args.model_path

    classifier = CoLLEGeWiCClassifier(secondLM.config.hidden_size).to(device)
    classifier.load_state_dict(torch.load(classifier_path))
    test_folder = Path('wic/wic_tsv/data/en/Test')

    contexts, target_inds, hypernyms, definitions, labels = dp.read_wic_tsv(wic_tsv_folder=test_folder)
    print(contexts[0])
    print(labels[0])
    print(definitions[0])

    for i, (context, target_id, definition) in enumerate(zip(contexts, target_inds, definitions)):
        word = context.split()[target_id]
        
        contexts[i] = re.sub(r"\b({})\b".format(word), "<nonce>", context, flags=re.I)
        definitions[i] = re.sub(r"\b({})\b".format(word), "<nonce>", definition, flags=re.I)
    print("post replace")
    print(contexts[0])
    print(labels[0])
    print(definitions[0])

    test_dataset = SimpleWiCDataset(contexts = contexts, 
                                     definitions=definitions,
                                     labels = labels)
    
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    test_predictions = []
    test_labels = []
    classifier.eval()
    model.eval()
    torch.set_grad_enabled(False)
    for (tc, td, tl) in tqdm(test_dl, total=len(test_dl)):
        test_context = tc[0]
        test_definition = td[0]
        test_label = tl[0]

        def_str = "The word <nonce> is defined as {}".format(test_definition)

        with torch.no_grad():
            ctx_hidden, def_hidden = predict_example(test_context, def_str, model,
                                                        tokenizerMLM, tokenizerTask, new_token_idx)
            
            if def_hidden.shape[0] > 1:
                def_hidden = torch.mean(def_hidden, dim=0, keepdim=True)
            if ctx_hidden.shape[0] > 1:
                ctx_hidden = torch.mean(ctx_hidden, dim=0, keepdim=True)

            cat_embeds = torch.cat([ctx_hidden, def_hidden], dim=1)
            logits, loss = classifier(cat_embeds, labels=torch.tensor([label], device=device).unsqueeze(0))
            test_preds = torch.flatten((logits >= 0.0).int()).detach().tolist()
            # test_loss += loss.detach().float()
            test_predictions += test_preds
            test_labels.append(test_label)
    
    test_metrics = compute_metrics(test_predictions, test_labels)
    print(test_metrics)