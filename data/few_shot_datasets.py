import numpy as np
from copy import deepcopy
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset
import re

from modules.utils import get_word_idx

from data.data_utils import *
from train_utils import snli_nonce

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
        #super(SimpleBaselineDataset, self).__init__()
        self.words = data["word"]
        self.texts = data["sentences"]
        self.n_samples = n_samples

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
        task_ids, task_labels = self.data_collator.torch_mask_tokens(inputs=item['task_inputs']['input_ids'],
                                                                     special_tokens_mask=item['task_inputs'][
                                                                         "special_tokens_mask"])
        inputs = item['task_inputs']['input_ids'].detach().clone()
        task_ids = item['task_inputs']['input_ids'].detach().clone()
        # make sure new token is masked so we get loss on it
        task_ids[inputs == item['nonceTask']] = self.tokenizerTask.mask_token_id
        task_labels[task_ids != self.tokenizerTask.mask_token_id] = -100
        task_labels[inputs == item['nonceTask']] = item['nonceTask']

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


class SimpleSNLIDataset(Dataset):
    def __init__(self, data, tokenizerMLM, tokenizerTask, n_samples):
        #         super(SimpleSNLIDataset, self).__init__()
        self.premises = data["premise"]
        self.hypotheses = data["hypothesis"]
        self.sentences = data['sentences']
        self.replacements = data['replace']
        self.labels = data['label']
        self.n_samples = n_samples

        self.tokenizerMLM = tokenizerMLM
        self.tokenizerTask = tokenizerTask

        if not self.tokenizerTask.pad_token:
            self.tokenizerTask.pad_token = self.tokenizerTask.eos_token

    def __len__(self):
        return len(self.premises)

    def __getitem__(self, idx):
        premise = self.premises[idx]
        hypothesis = self.hypotheses[idx]
        replaced = self.replacements[idx]
        label = self.labels[idx]
        sentences = self.sentences[idx]

        if label == -1:
            return None

        nonce = snli_nonce(replaced.lower())

        premise = re.sub(r"\b({})\b".format(replaced), nonce, premise, flags=re.I)
        hypothesis = re.sub(r"\b({})\b".format(replaced), nonce, hypothesis, flags=re.I)

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

        #         sentences = [s.strip() for s in sentences]
        #         sentences = [s.replace('\n', ' ') for s in sentences]
        # #         print(replaced)
        # #         print(sentences)
        #         for s in sentences:
        #             if not re.search(r"\b({})\b".format(replaced), s):
        #                 return None

        sentences = [s.strip() for s in sentences]
        sentences = [s.replace('\n', " ") for s in sentences]
        sents = []

        if len(sentences) < self.n_samples:
            return None

        for s in sentences:
            if not re.search(r"\b({})\b".format(replaced), s, flags=re.I):
                continue
            else:
                sents.append(re.sub(r"\b({})\b".format(replaced), nonce, s, flags=re.I, count=1))

        sentences = sents

        try:
            if do_sample:
                sentences = np.random.choice(sentences, size=self.n_samples).tolist()
        except:
            return None

        tokensMLM = self.tokenizerMLM(sentences,
                                      max_length=mlm_length,
                                      padding='max_length',
                                      truncation=True,
                                      return_tensors='pt')

        tokensTask = self.tokenizerTask(premise,
                                        hypothesis,
                                        max_length=task_length,
                                        padding='max_length',
                                        truncation=True,
                                        return_tensors='pt',
                                        return_special_tokens_mask=True)

        return {
            'mlm_inputs': tokensMLM,  # shape for output is batch (per nonce) x k (examples) x 512 (tokens)
            'task_inputs': tokensTask,
            'nonceMLM': nonceMLM,
            'nonceTask': nonceTask,
            'task_labels': torch.LongTensor([label])
        }


class SimpleSQuADDataset(Dataset):
    def __init__(self, data, tokenizerMLM, tokenizerTask, n_samples):
        #         super(SimpleSQuADDataset, self).__init__()
        self.ids = data['id']
        self.contexts = data['context']
        self.questions = data['question']
        self.answers = data['answers']
        self.replace = data['replace']
        self.sentences = data['sentences']

        self.tokenizerMLM = tokenizerMLM
        self.tokenizerTask = tokenizerTask

        self.n_samples = n_samples

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        replace = self.replace[idx].lower()
        sentences = self.sentences[idx]
        context = self.contexts[idx]

        nonce = "<{}_nonce>".format(replace.lower())

        # do mlm stuff
        sentences = [s.strip() for s in sentences]
        sentences = [s.replace('\n', " ") for s in sentences]
        sents = []
        for s in sentences:
            if not re.search(r"\b({})\b".format(replace), s, flags=re.I):
                continue
            else:
                sents.append(re.sub(r"\b({})\b".format(replace), nonce, s, flags=re.I, count=1))

        sentences = sents

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

        if len(sentences) < self.n_samples:
            return None

        if do_sample:
            sentences = np.random.choice(sentences, size=self.n_samples).tolist()

        tokensMLM = self.tokenizerMLM(sentences,
                                      max_length=mlm_length,
                                      padding='max_length',
                                      truncation=True,
                                      return_tensors='pt')

        new_question = re.sub(r"\b({})\b".format(replace), nonce, question, flags=re.I)
        new_context = re.sub(r"\b({})\b".format(replace), nonce, context, flags=re.I)

        new_answer_text = [re.sub(r"\b({})\b".format(replace), nonce, a, flags=re.I) for a in answer['text']]
        new_answer_start = [new_context.find(a) for a in new_answer_text]

        new_answer = {'text': new_answer_text, 'answer_start': new_answer_start}

        # do task stuff
        task_inputs = self.tokenizerTask(
            new_question,
            new_context,
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length"
        )
        #         print(task_inputs.sequence_ids(0))

        offset = task_inputs.pop("offset_mapping")
        start_positions = []
        end_positions = []

        start_char = new_answer["answer_start"][0]
        end_char = new_answer["answer_start"][0] + len(new_answer["text"][0])

        # Find the start and end of the context
        sequence_ids = task_inputs.sequence_ids(0)
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

        task_inputs = self.tokenizerTask(
            new_question,
            new_context,
            max_length=384,
            truncation="only_second",
            padding="max_length",
            return_tensors='pt'
        )
        return {
            'mlm_inputs': tokensMLM,  # shape for output is batch (per nonce) x k (examples) x 512 (tokens)
            'task_inputs': task_inputs,
            'nonceMLM': nonceMLM,
            'nonceTask': nonceTask,
            'task_start': start_positions,
            'task_end': end_positions
        }

class MyDataset(Dataset):
    def __init__(self, n_examples: int, min_digits: int, max_digits: int, operation: str,
                 orthography: str, base_number: int, invert_question: bool, invert_answer: bool,
                 balance: bool):

        self.operation = operation
        self.orthography = orthography
        self.invert_answer = invert_answer
        self.invert_question = invert_question
        self.base_number = base_number
        self.max_digits = max_digits
        if self.base_number != 10:
            assert self.orthography != 'words', 'Cannot convert to words when base is different than 10.'
            assert self.operation == 'addition', 'Cannot perform {self.operation} when base is different than 10.'

        if self.invert_question or self.invert_answer:
            assert self.orthography != 'words', 'Cannot invert number when ortography = "words".'

        if balance:
            self.examples = []
            for _ in range(n_examples):
                example = []
                for _ in range(2):
                    max_digits_i = random.randint(min_digits, max_digits)
                    min_number = int((max_digits_i - 1) * '9') + 1
                    max_number = int(max_digits_i * '9')
                    example.append(random.randint(min_number, max_number))
                self.examples.append(example)
        else:
            self.examples = [
                (random.randint(0, int(max_digits * '9')), random.randint(0, int(max_digits * '9')))
                for _ in range(n_examples)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        first_term, second_term = self.examples[idx]

        if self.operation == 'addition':
            operation_term = 'plus'
            result = first_term + second_term
        elif self.operation == 'subtraction':
            operation_term = 'minus'
            result = first_term - second_term
        else:
            raise Exception(f'Invalid operation: {self.operation}')

        first_term = self.convert_number(first_term, invert_number=self.invert_question)
        second_term = self.convert_number(second_term, invert_number=self.invert_question)
        answer = self.convert_number(result, invert_number=self.invert_answer)

        return first_term, second_term, operation_term, answer

    def convert_number(self, number: int, invert_number: bool) -> str:
        number = str(number)
        if self.base_number != 10:
            number = convert_to_base(num=int(number), base=self.base_number)

        if self.orthography == 'decimal':
            return convert_to_character(
                number=number, separator='', invert_number=invert_number,
                max_digits=-1)
        elif self.orthography == 'character':
            return convert_to_character(
                number=number, separator=' ', invert_number=invert_number,
                max_digits=-1)
        elif self.orthography == 'character_fixed':
            return convert_to_character(
                number=number, separator=' ', invert_number=invert_number,
                max_digits=self.max_digits)
        elif self.orthography == 'underscore':
            return convert_to_character(
                number=number, separator='_', invert_number=invert_number,
                max_digits=-1)
        elif self.orthography == 'words':
            raise NotImplementedError
        elif self.orthography == '10based':
            raise NotImplementedError
        elif self.orthography == '10ebased':
            raise NotImplementedError
        else:
            raise Exception(f'Wrong orthography: {self.orthography}')

class CalcDataset(Dataset):

    def __init__(self, n_examples: int, min_digits: int, max_digits: int, operation: str,
                 orthography: str, base_number: int, invert_question: bool, invert_answer: bool,
                 balance: bool):

        self.math_dataset = MyDataset(n_examples, min_digits,
                                      max_digits, operation, orthography,
                                      base_number, invert_question, invert_answer, balance)
    def __len__(self):
        return len(self.math_dataset)

    def __getitem__(self, idx):
        first_term, second_term, operation_term, answer = self.math_dataset.__getitem__(idx)

        random_idx = np.random.choice(len(self.math_dataset))

        random_pair = self.math_dataset.examples[random_idx]

        simple_statement = "{0} plus {1} = [CALC] {0} plus {1} -> {2} [END_CALC]".format(first_term, second_term, answer)
        complex_statement = "[CALC] {0} plus {1} -> {2} [END_CALC] plus {3} = ".format(first_term, second_term, answer, random_pair[0])

        complex_ans = int(answer) + random_pair[0]
        complex_api = "[CALC] {} plus {} -> ".format(answer, random_pair[0]) + "{} [END_CALC]"
        return simple_statement, complex_statement, complex_api, complex_ans


class SimpleMathDataset(Dataset):

    def __init__(self, dataset, tokenizerMLM, tokenizerTask, max_length):
        self.max_length = max_length
        self.base_dataset = dataset
        self.tokenizerMLM = tokenizerMLM
        self.tokenizerTask = tokenizerTask

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        #         print(self.base_dataset.__getitem__(idx))
        first, second, op, answer = self.base_dataset.__getitem__(idx)

        new_tok = "<OP>"

        sentence = 'Q: What is {} {} {}? A: {}'.format(first, new_tok, second, answer)
        sentence_minus_answer = 'Q: What is {} {} {}? A: '.format(first, new_tok, second)

        nonceMLM = self.tokenizerMLM.convert_tokens_to_ids(new_tok)
        nonceTask = self.tokenizerTask.convert_tokens_to_ids(new_tok)

        if self.tokenizerMLM.model_max_length:
            mlm_length = self.tokenizerMLM.model_max_length
        else:
            raise Exception("Model Max Length does not exist for MLM")

        if self.tokenizerTask.model_max_length:
            task_length = self.tokenizerTask.model_max_length
        else:
            raise Exception("Model Max Length does not exist for TaskLM")

        tokensMLM = self.tokenizerMLM(sentence,
                                      max_length=self.max_length,
                                      truncation=True,
                                      padding='max_length',
                                      return_tensors='pt')

        tokensTask = self.tokenizerTask(sentence,
                                        max_length=self.max_length,
                                        truncation=True,
                                        padding='max_length',
                                        return_tensors='pt')
        task_labels = tokensTask['input_ids'].clone()
        task_labels[task_labels == self.tokenizerTask.unk_token_id] = -100

        generationTokens = self.tokenizerTask(sentence_minus_answer,
                                              truncation=True,
                                              return_tensors='pt')
        return {
            'mlm_inputs': tokensMLM,  # shape for output is batch (per nonce) x k (examples) x 512 (tokens)
            'task_inputs': tokensTask,
            'task_labels': task_labels,
            'nonceMLM': nonceMLM,
            'nonceTask': nonceTask,
            'generationTokens': generationTokens
        }


class SimpleOnlineDataset(Dataset):
    def __init__(self, data, tokenizerMLM, tokenizerTask):
        self.text = data['text']
        self.tokenizerMLM = tokenizerMLM
        self.tokenizerTask = tokenizerTask

        if "roberta" in self.tokenizerTask.name_or_path:
            self.data_collator = DataCollatorForLanguageModeling(self.tokenizerTask, mlm=True, mlm_probability=0.15)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx]

        if self.tokenizerMLM.model_max_length:
            mlm_length = self.tokenizerMLM.model_max_length
        else:
            raise Exception("Model Max Length does not exist for MLM")

        if self.tokenizerTask.model_max_length:
            task_length = self.tokenizerTask.model_max_length
        else:
            raise Exception("Model Max Length does not exist for TaskLM")

        tokensMLM = self.tokenizerMLM(text,
                                      max_length=mlm_length,
                                      truncation=True,
                                      padding='max_length',
                                      return_tensors='pt')

        if "gpt2" in self.tokenizerTask.name_or_path:
            tokensTask = self.tokenizerTask(self.tokenizerTask.bos_token + text + self.tokenizerTask.eos_token,
                                            max_length=task_length,
                                            truncation=True,
                                            padding='max_length',
                                            return_tensors='pt')

            task_labels = tokensTask['input_ids'].clone()
            task_labels[task_labels == self.tokenizerTask.unk_token_id] = -100
            task_inputs = tokensTask

        elif "roberta" in self.tokenizerTask.name_or_path:
            tokensTask = self.tokenizerTask(text,
                                            max_length=task_length,
                                            padding='max_length',
                                            truncation=True,
                                            return_tensors='pt',
                                            return_special_tokens_mask=True)

            task_ids, task_labels = self.data_collator.torch_mask_tokens(inputs=tokensTask['input_ids'],
                                                                         special_tokens_mask=tokensTask[
                                                                             "special_tokens_mask"])

            masked_task = deepcopy(tokensTask)

            masked_task['input_ids'] = task_ids
            task_inputs = masked_task

        return {
            'mlm_inputs': tokensMLM,  # shape for output is batch (per nonce) x k (examples) x 512 (tokens)
            'task_inputs': task_inputs,
            'task_labels': task_labels
        }


class SimpleMathDatasetSubtok(Dataset):

    def __init__(self, dataset, tokenizerMLM, tokenizerTask, max_length):
        self.max_length = max_length
        self.base_dataset = dataset
        self.tokenizerMLM = tokenizerMLM
        self.tokenizerTask = tokenizerTask

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        #         print(self.base_dataset.__getitem__(idx))
        first, second, op, answer = self.base_dataset.__getitem__(idx)

        tok1 = "<NUM1>"
        tok2 = "<NUM2>"
        sentence = 'Q: What is {} plus {}? A: '.format(first, second)

        sentence_task = 'Q: What is {} plus {}? A: {}'.format(tok1, tok2, answer)

        sentence_gen = 'Q: What is {} plus {}? A: '.format(tok1, tok2)

        if self.tokenizerMLM.model_max_length:
            mlm_length = self.tokenizerMLM.model_max_length
        else:
            raise Exception("Model Max Length does not exist for MLM")

        if self.tokenizerTask.model_max_length:
            task_length = self.tokenizerTask.model_max_length
        else:
            raise Exception("Model Max Length does not exist for TaskLM")

        tokensMLM = self.tokenizerMLM(sentence,
                                      max_length=self.max_length,
                                      truncation=True,
                                      padding='max_length',
                                      return_tensors='pt')

        tokensTask = self.tokenizerTask(self.tokenizerTask.bos_token + sentence_task + self.tokenizerTask.eos_token,
                                        max_length=self.max_length,
                                        truncation=True,
                                        padding='max_length',
                                        return_tensors='pt')

        task_labels = tokensTask['input_ids'].clone()
        #task_labels[task_labels == self.tokenizerTask.unk_token_id] = -100

        firstSpan = get_span(sentence, first, self.tokenizerMLM)
        secondSpan = get_span(sentence, second, self.tokenizerMLM)

        generationTokens = self.tokenizerTask(self.tokenizerTask.bos_token + sentence_gen,
                                              truncation=True,
                                              return_tensors='pt')
        return {
            'mlm_inputs': tokensMLM,  # shape for output is batch (per nonce) x k (examples) x 512 (tokens)
            'task_inputs': tokensTask,
            'task_labels': task_labels,
            'firstSpan': firstSpan,
            'secondSpan': secondSpan,
            'generationTokens': generationTokens
        }


class SimpleCalcDataset(Dataset):

    def __init__(self, dataset, tokenizerMLM, tokenizerTask, max_length, n, complex_first=0.0):
        self.max_length = max_length
        self.base_dataset = dataset
        self.tokenizerMLM = tokenizerMLM
        self.tokenizerTask = tokenizerTask
        self.n = n
        self.complex_first = complex_first

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        #         print(self.base_dataset.__getitem__(idx))
        choice_idxs = np.random.choice(len(self.base_dataset), size=self.n - 1)

        context = []

        for i in choice_idxs:
            simple, cplx, cplx_api, answer = self.base_dataset.__getitem__(i)
            if np.random.binomial(n=1, p=self.complex_first) == 1:
                context.append(cplx + cplx_api.format(answer))
            else:
                context.append(simple)

        simple, cplx, cplx_api, answer = self.base_dataset.__getitem__(idx)

        new_toks = ["[CALC]", "[END_CALC]"]

        cplx_with_ans = cplx + cplx_api.format(answer)
        print(answer)
        print(cplx_with_ans)
        nonceMLM = self.tokenizerMLM.convert_tokens_to_ids(new_toks)
        nonceTask = self.tokenizerTask.convert_tokens_to_ids(new_toks)

        if self.tokenizerMLM.model_max_length:
            mlm_length = self.tokenizerMLM.model_max_length
        else:
            raise Exception("Model Max Length does not exist for MLM")

        if self.tokenizerTask.model_max_length:
            task_length = self.tokenizerTask.model_max_length
        else:
            raise Exception("Model Max Length does not exist for TaskLM")

        context.append(simple)

        tokensMLM = self.tokenizerMLM(context,
                                      max_length=self.max_length,
                                      truncation=True,
                                      padding='max_length',
                                      return_tensors='pt')

        tokensTask = self.tokenizerTask(self.tokenizerTask.bos_token + cplx_with_ans + self.tokenizerTask.eos_token,
                                        max_length=self.max_length,
                                        truncation=True,
                                        padding='max_length',
                                        return_tensors='pt')

        task_labels = tokensTask['input_ids'].clone()
        task_labels[task_labels == self.tokenizerTask.unk_token_id] = -100

        generationTokens = self.tokenizerTask(self.tokenizerTask.bos_token + cplx,
                                              truncation=True,
                                              return_tensors='pt')
        return {
            'mlm_inputs': tokensMLM,  # shape for output is batch (per nonce) x k (examples) x 512 (tokens)
            'task_inputs': tokensTask,
            'task_labels': task_labels,
            'nonceMLM': nonceMLM,
            'nonceTask': nonceTask,
            'generationTokens': generationTokens
        }
