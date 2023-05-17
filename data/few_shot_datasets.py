import numpy as np
from copy import deepcopy
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset
from modules.utils import get_word_idx

from data.data_utils import *

class ChimerasDataset(Dataset):
    def __init__(self, data, tokenizerMLM, tokenizerTask, n_samples, trial):
        self.ids = data["id"]
        self.probes = data["probes"]
        self.ratings = data["ratings"]
        self.texts = data["text"]
        self.n_samples = n_samples
        self.trial = trial

        self.tokenizerMLM = tokenizerMLM
        self.tokenizerTask = tokenizerTask

        if not self.tokenizerTask.pad_token:
            self.tokenizerTask.pad_token = self.tokenizerTask.eos_token

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        probes = self.probes[idx]
        nonce = make_nonce(self.ids[idx])
        ratings = self.ratings[idx]

        nonceMLM = self.tokenizerMLM.convert_tokens_to_ids(nonce)
        nonceTask = self.tokenizerTask.convert_tokens_to_ids(nonce)

        n_sentences = int(self.trial[1])
        do_sample = False
        if self.n_samples < n_sentences:  # sample if we want to choose less example sentences than there are in the trial
            do_sample = True

        if self.tokenizerMLM.model_max_length:
            mlm_length = self.tokenizerMLM.model_max_length
        else:
            raise Exception("Model Max Length does not exist for MLM")

        if self.tokenizerTask.model_max_length:
            task_length = self.tokenizerTask.model_max_length
        else:
            raise Exception("Model Max Length does not exist for TaskLM")

        #         mlm_examples = []
        #         task_examples = []

        sentences = text.split("[SEN]")
        sentences = [s.strip() for s in sentences]
        if do_sample:
            sentences = np.random.choice(sentences, size=self.n_samples)

        tokensMLM = self.tokenizerMLM(sentences,
                                      max_length=mlm_length,
                                      padding='max_length',
                                      truncation=True,
                                      return_tensors='pt')

        tokensTask = self.tokenizerTask(sentences,
                                        max_length=task_length,
                                        padding='max_length',
                                        truncation=True,
                                        return_tensors='pt',
                                        return_special_tokens_mask=True)

        #         mlm_examples.append(tokensMLM)
        #         task_examples.append(tokensTask)

        return {
            'mlm_inputs': tokensMLM,  # shape for output is batch (per nonce) x k (examples) x 512 (tokens)
            'task_inputs': tokensTask,
            'probes': probes,
            'ratings': ratings,
            'nonceMLM': nonceMLM,
            'nonceTask': nonceTask
        }


class ChimeraMLMDataset(ChimerasDataset):
    def __init__(self, data, tokenizerMLM, tokenizerTask, n_samples, trial):
        super(ChimeraMLMDataset, self).__init__(data, tokenizerMLM, tokenizerTask, n_samples, trial)
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizerTask, mlm=True, mlm_probability=0.15)
        self.eval_set = ChimeraTestDataset(data, tokenizerMLM, tokenizerTask, n_samples, trial)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)

        # apply data collator to the task inputs
        task_ids, task_labels = self.data_collator.torch_mask_tokens(inputs=item['task_inputs']['input_ids'],
                                                                     special_tokens_mask=item['task_inputs'][
                                                                         "special_tokens_mask"])

        masked_task = deepcopy(item['task_inputs'])

        masked_task['input_ids'] = task_ids

        eval_items = self.eval_set.__getitem__(idx)
        return {
            'mlm_inputs': item['mlm_inputs'],  # shape for output is batch (per nonce) x k (examples) x 512 (tokens)
            'task_inputs': masked_task,
            'probes': item['probes'],
            'ratings': item['ratings'],
            'nonceMLM': item['nonceMLM'],
            'nonceTask': item['nonceTask'],
            'task_labels': task_labels,
            #             'probe_inputs': eval_items['probe_inputs'],
            #             'eval_nonce': eval_items['eval_nonce'],
            #             'eval_nonce_sentence': eval_items['eval_nonce_sentence']

        }


class ChimeraTestDataset(Dataset):
    def __init__(self, data, tokenizerMLM, tokenizerTask, n_samples, trial):
        self.ids = data["id"]
        self.probes = data["probes"]
        self.ratings = data["ratings"]
        self.texts = data["text"]
        self.n_samples = n_samples
        self.trial = trial

        self.tokenizerMLM = tokenizerMLM
        self.tokenizerTask = tokenizerTask

        if not self.tokenizerTask.pad_token:
            self.tokenizerTask.pad_token = self.tokenizerTask.eos_token

        self.data_collator = DataCollatorForLanguageModeling(self.tokenizerTask, mlm=True, mlm_probability=0.15)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        probes = self.probes[idx]
        nonce = make_nonce(self.ids[idx])
        ratings = self.ratings[idx]

        nonceMLM = self.tokenizerMLM.convert_tokens_to_ids(nonce)
        nonceTask = self.tokenizerTask.convert_tokens_to_ids(nonce)

        n_sentences = int(self.trial[1])
        do_sample = False
        if self.n_samples < n_sentences:  # sample if we want to choose less example sentences than there are in the trial
            do_sample = True

        if self.tokenizerMLM.model_max_length:
            mlm_length = self.tokenizerMLM.model_max_length
        else:
            raise Exception("Model Max Length does not exist for MLM")

        if self.tokenizerTask.model_max_length:
            task_length = self.tokenizerTask.model_max_length
        else:
            raise Exception("Model Max Length does not exist for TaskLM")

        #         mlm_examples = []
        #         task_examples = []
        sentences = text.split("[SEN]")
        sentences = [s.strip() for s in sentences]
        indices = []
        try:
            for sentence in sentences:
                indices.append(get_word_idx(sentence, nonce))
        except:
            return None

        if do_sample:
            sentences = np.random.choice(sentences, size=self.n_samples)

        tokensMLM = self.tokenizerMLM(sentences,
                                      max_length=mlm_length,
                                      padding='max_length',
                                      truncation=True,
                                      return_tensors='pt')

        tokensTask = self.tokenizerTask(sentences,
                                        max_length=task_length,
                                        padding='max_length',
                                        truncation=True,
                                        return_tensors='pt',
                                        return_special_tokens_mask=True)

        eval_nonce = self.tokenizerTask(nonce,
                                        max_length=task_length,
                                        truncation=True,
                                        return_tensors='pt',
                                        return_special_tokens_mask=True)

        eval_nonce_sentence = self.tokenizerTask.encode_plus(sentences[-1],
                                                             max_length=task_length,
                                                             truncation=True,
                                                             return_tensors='pt',
                                                             return_special_tokens_mask=True)

        task_ids, task_labels = self.data_collator.torch_mask_tokens(inputs=tokensTask['input_ids'],
                                                                     special_tokens_mask=tokensTask[
                                                                         "special_tokens_mask"])

        #         word_idx = get_word_idx(sentences[-1], nonce)
        #         nonce_locs = get_locs(sentences[-1], word_idx, self.tokenizerTask)
        probe_words = probes.split(',')
        probe_inputs = {}
        for probe in probe_words:
            probe_sentences = []
            eval_input = {}
            for sentence in sentences:
                p_s = sentence.replace(nonce, probe)
                probe_sentences.append(p_s)

            #             probesMLM = self.tokenizerMLM(probe_sentences,
            #                                       max_length=mlm_length,
            #                                       padding='max_length',
            #                                       truncation=True,
            #                                       return_tensors='pt')

            #             probesTask = self.tokenizerTask(probe_sentences,
            #                                         max_length=task_length,
            #                                         padding='max_length',
            #                                         truncation=True,
            #                                         return_tensors='pt')
            #             print(probesTask.)
            probesMLM = self.tokenizerMLM(probe,
                                          max_length=mlm_length,
                                          truncation=True,
                                          return_tensors='pt')
            probesTask = self.tokenizerTask(probe,
                                            max_length=task_length,
                                            truncation=True,
                                            return_tensors='pt')
            sentence = sentences[-1]
            probe_sentence = sentence.replace(nonce, probe)

            probe_sent_enc = self.tokenizerTask.encode_plus(probe_sentence,
                                                            max_length=task_length,
                                                            truncation=True,
                                                            return_tensors='pt')
            #             probe_locs = get_locs(probe_sentence, word_idx, self.tokenizerTask)
            eval_input['mlm'] = probesMLM
            eval_input['task'] = probesTask
            eval_input['sentences'] = probe_sentences
            #             eval_input['probe_locs'] = probe_locs
            probe_inputs[probe] = eval_input

        #         mlm_examples.append(tokensMLM)
        #         task_examples.append(tokensTask)

        return {
            'mlm_inputs': tokensMLM,  # shape for output is batch (per nonce) x k (examples) x 512 (tokens)
            'task_inputs': tokensTask,
            'probes': probes,
            'ratings': ratings,
            'nonceMLM': nonceMLM,
            'nonceTask': nonceTask,
            'probe_inputs': probe_inputs,
            'eval_nonce': eval_nonce,
            'eval_nonce_sentence': eval_nonce_sentence,
            'eval_sentences': sentences,
            'eval_indices': indices,
            'gd_ids': task_ids,
            'gd_labels': task_labels
            #             'nonce_locs': nonce_locs
        }


class SimpleBaselineDataset(Dataset):
    def __init__(self, data, tokenizerMLM, tokenizerTask, n_samples):
        super(SimpleBaselineDataset, self).__init__()
        self.words = data["word"]
        self.texts = data["sentences"]
        self.n_samples = n_samples
        self.device = device

        self.tokenizerMLM = tokenizerMLM
        self.tokenizerTask = tokenizerTask

        if not self.tokenizerTask.pad_token:
            self.tokenizerTask.pad_token = self.tokenizerTask.eos_token

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        word = self.words[idx]
        nonce = "<{}>".format(word)

        text = text.replace(word, nonce)

        nonceMLM = self.tokenizerMLM.convert_tokens_to_ids(nonce)
        nonceTask = self.tokenizerTask.convert_tokens_to_ids(nonce)

        do_sample = True

        if self.tokenizerMLM.model_max_length:
            mlm_length = self.tokenizerMLM.model_max_length
        else:
            raise Exception("Model Max Length does not exist for MLM")

        if self.tokenizerTask.model_max_length:
            task_length = self.tokenizerTask.model_max_length
        else:
            raise Exception("Model Max Length does not exist for TaskLM")

        sentences = text.split("[SEN]")
        sentences = [s.strip() for s in sentences]
        if do_sample:
            sentences = np.random.choice(sentences, size=self.n_samples).tolist()

        tokensMLM = self.tokenizerMLM(sentences,
                                      max_length=mlm_length,
                                      padding='max_length',
                                      truncation=True,
                                      return_tensors='pt')

        tokensTask = self.tokenizerTask(sentences,
                                        max_length=task_length,
                                        padding='max_length',
                                        truncation=True,
                                        return_tensors='pt',
                                        return_special_tokens_mask=True)

        return {
            'mlm_inputs': tokensMLM,  # shape for output is batch (per nonce) x k (examples) x 512 (tokens)
            'task_inputs': tokensTask,
            'nonceMLM': nonceMLM,
            'nonceTask': nonceTask
        }


class SimpleMLMDataset(SimpleBaselineDataset):
    def __init__(self, data, tokenizerMLM, tokenizerTask, n_samples):
        super(SimpleMLMDataset, self).__init__(data, tokenizerMLM, tokenizerTask, n_samples)
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizerTask, mlm=True, mlm_probability=0.15)

    #         self.eval_set = ChimeraTestDataset(data, tokenizerMLM, tokenizerTask, n_samples)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)

        # apply data collator to the task inputs
        #         task_ids, task_labels = self.data_collator.torch_mask_tokens(inputs = item['task_inputs']['input_ids'],
        #                                                                     special_tokens_mask=item['task_inputs']["special_tokens_mask"])
        task_labels = item['task_inputs']['input_ids'].detach().clone()
        task_ids = item['task_inputs']['input_ids'].detach().clone()
        # make sure new token is masked so we get loss on it
        task_ids[item['task_inputs']['input_ids'] == item['nonceTask']] = self.tokenizerTask.mask_token_id
        task_labels[task_ids != self.tokenizerTask.mask_token_id] = -100

        #         masked_task = deepcopy(item['task_inputs'])

        masked_task = {"input_ids": task_ids, "attention_mask": item["task_inputs"]['attention_mask']}

        return {
            'mlm_inputs': item['mlm_inputs'],  # shape for output is batch (per nonce) x k (examples) x 512 (tokens)
            'original_task_inputs': item['task_inputs'],
            'task_inputs': masked_task,
            'nonceMLM': item['nonceMLM'],
            'nonceTask': item['nonceTask'],
            'task_labels': task_labels,
        }
