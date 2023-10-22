from datasets import load_from_disk, load_dataset
from datasets import DatasetDict, Dataset
import re
import os
from datasets.distributed import split_dataset_by_node


word_dict = load_from_disk("pile_word_replacements")
words = word_dict['train']['words'] + word_dict['test']['words']


def check_example(ex):
    found = False
    if re.search(r"\b({})\b".format("|".join(words)), ex['text'], flags=re.I):
        found = True

    return found

def check(ex):
    pile_set = ex['meta']['pile_set_name'].lower()
    if "github" not in pile_set and "europarl" not in pile_set:
        return True
    else:
        return False

if __name__ == "__main__":
    data = load_dataset("/scratch/work/public/ml-datasets/pile", streaming=True)
    data = data.filter(check)
    filtered = data.filter(check_example)
    train = split_dataset_by_node(filtered['train'], rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
    test = split_dataset_by_node(filtered['test'], rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
    sample_train = filtered['train'].shuffle(buffer_size=1).take(300000)
    sample_test = filtered['test'].shuffle(buffer_size=1).take(2000)
    sample_train = list(sample_train)
    sample_test = list(sample_test)

    sample_train = Dataset.from_dict(
        {'text': [v['text'] for v in sample_train], 'meta': [v['meta'] for v in sample_train]})
    sample_test = Dataset.from_dict(
        {'text': [v['text'] for v in sample_test], 'meta': [v['meta'] for v in sample_test]})

    DatasetDict({'train': sample_train, 'test': sample_test}).save_to_disk("pile_processing/pile_samples_large_{}".format(int(os.environ["RANK"])))