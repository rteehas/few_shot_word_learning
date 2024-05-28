from definition_eval import *
import pandas as pd
from tqdm import tqdm

def generate_oxford_def_emb_gen(model, ex, tokenizerMLM, tokenizerTask, with_prompt):
    examples = [ex['replaced_examples']]
    context = tokenizerMLM(examples, truncation=True, padding='longest', return_tensors='pt')
    # nonce = "<{}_new>".format(ex['word'].lower())
    nonce = "<nonce>"
    if with_prompt:
        prompt = definition_prompt.format("\n".join(examples), nonce)
    else:
        prompt = "The word \"{}\" is defined as".format(nonce)

    inputs = tokenizerTask(prompt, truncation=True, return_tensors='pt', max_length=256).to(device)

    outputs = generate(model, context, inputs['input_ids'], inputs['attention_mask'], 30, mask_new_tokens=True)
    # print(outputs)
    generated_def = tokenizerTask.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    # print(ex['word'], generated_def)
    new_ex = {'definition': ex['definition'],
           'word': ex['word'].split(),
           'generated definition': generated_def,
            'examples': examples,
            'prompt': prompt}
    return new_ex

def run_emb_gen(def_task, path):

    id = uuid.uuid4()
    fname_format = "oxford_task_outputs/emb_gen_generations_masked_new_token_new_data_new_model_{}".format(id)
    tokenizerMLM = AutoTokenizer.from_pretrained(path + "/tokenizerMLM", use_fast=False)
    tokenizerTask = LlamaTokenizer.from_pretrained(path + "tokenizerTask", use_fast=False, legacy=True)
    nonces = list(tokenizerTask.get_added_vocab().keys())

    firstLM = RobertaForMaskedLM.from_pretrained("roberta-large", low_cpu_mem_usage=True)
    secondLM = LlamaForCausalLM.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf", low_cpu_mem_usage=True)

    memory_config = AggregatorConfig()

    mask_token_id = tokenizerMLM.mask_token_id

    layers=[-1]
    model = MorphMemoryModelLLAMA(firstLM, secondLM, len(nonces), layers, mask_token_id, memory_config, 1, None).to(device)
    model.emb_gen.load_state_dict(torch.load(path + "/pytorch_model.bin"))
    model.device = device
    model.firstLM.eval()
    model.secondLM.eval()


    model.eval()
    all_outputs = []
    for ex in tqdm(def_task):
        # print(ex)
        step_output_with_prompt = generate_oxford_def_emb_gen(model, ex, tokenizerMLM, tokenizerTask, with_prompt=True)
        step_output_without_prompt = generate_definitions_emb_gen(model, ex, tokenizerMLM, tokenizerTask, with_prompt=False)
        all_outputs.append(step_output_with_prompt)
        all_outputs.append(step_output_without_prompt)

    # save_dir = fname_format.format(lr)
    keys = all_outputs[0].keys()
    data_dict = {}
    for key in keys:
        data_dict[key] = [output_ex[key] for output_ex in all_outputs]
    print("Saving...")
    Dataset.from_dict(data_dict).save_to_disk(fname_format)
    return all_outputs



if __name__ == "__main__":
    args = get_arguments().parse_args()
    # def_task = pd.read_csv("merged_oxford_test_set.csv")
    def_task = load_dataset("csv", data_files="merged_oxford_test_set.csv")
    def_task = def_task['train']
    # if args.model == "hice":
    #     run_hice(def_task)
    # elif args.model == "additive":
    #     run_additive(def_task)
    # elif args.model == "baseline_gd":
    #     run_baseline(def_task, args.lr)
    # elif args.model == "baseline_no_gd":
    #     run_baseline_no_gd(def_task)
    if args.model == "emb_gen":
        path = "model_checkpoints/layers/no_mp/llama/input_and_output/filtered/redone_pile/layernorm/roberta-large/1_layers/last_1/32_batch_size/mean_agg/1_examples/lr_0.001/weight_decay_0.1/with_negatives_and_regression/distillation_weight_0.05_temp_3/output_embedding_cosine/checkpoints/checkpoint_7_28000"
        run_emb_gen(def_task, path)
    else:
        raise NotImplementedError
