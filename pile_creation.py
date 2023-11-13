from datasets import load_from_disk, load_dataset
from datasets import DatasetDict, Dataset
import re
import os
from torch.utils.data import DataLoader
from datasets.distributed import split_dataset_by_node


word_dict = load_from_disk("pile_word_replacements")
words = word_dict['train']['words']


def check_example(ex):
    found = False

    for w in words:
        if re.search(r"\b({})\b".format(w), ex['text'], flags=re.I) is not None:
            found=True
            break

    return found

def check(ex):
    pile_set = ex['meta']['pile_set_name'].lower()
    if "github" not in pile_set and "europarl" not in pile_set:
        return True
    else:
        return False

if __name__ == "__main__":
    # dist.init_process_group(backend="gloo")

    data = load_dataset("stanford-crfm/DSIR-filtered-pile-50M", streaming=True)
    data = data.shuffle(buffer_size=1)
    data = data.filter(check)
    filtered = data.filter(check_example)
    # print("Rank:", os.environ["RANK"])
    # print("world size", os.environ["WORLD_SIZE"])
    sample_train = filtered['train']
    # sample_test = filtered['test'].shuffle(buffer_size=1)

    sample_train_dl = DataLoader(sample_train, num_workers=30)
    total_samples = 150000
    l = []

    for i, ex in enumerate(sample_train_dl):
        if i < total_samples:
            l.append(ex)
        else:
            break


    sample_train = Dataset.from_dict(
        {'text': [v['text'] for v in l], 'meta': [v['meta'] for v in l]})
    sample_train.save_to_disk("filtered_pile_500k_train")
    # sample_test = Dataset.from_dict(
    #     {'text': [v['text'] for v in sample_test], 'meta': [v['meta'] for v in sample_test]})

    # DatasetDict({'train': sample_train, 'test': sample_test}).save_to_disk("pile_processing/pile_samples_large_{}".format(int(os.environ["RANK"])))
