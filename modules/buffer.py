import numpy as np


class RetrievalBuffer():

    def __init__(self, max_num, k, nonce_list, tokenizer, random, cat):
        self.k = k
        self.max_num = max_num
        self.buffer = {}
        self.nonces = nonce_list
        self.tokenizer = tokenizer
        self.random = random
        self.cat = cat

    def store(self, mlm_inputs):

        for n in self.nonces:
            if n not in mlm_inputs['input_ids']:
                continue
            else:
                locs = (mlm_inputs['input_ids'] == n)
                tups = locs.nonzero(as_tuple=True)
                b, k, l = tups
                uq = list(set([(i, j) for i, j in zip(b, k)]))
                for (i, j) in uq:
                    mem = self.tokenizer.decode(mlm_inputs['input_ids'][i, j, :], skip_special_tokens=True,
                                                clean_up_tokenization_spaces=True)

                    if n in self.buffer:
                        self.buffer[n] = [mem] + self.buffer[n]
                    else:
                        self.buffer[n] = [mem]

    def cleanup(self):
        for key in self.buffer:
            while len(self.buffer[key]) > self.max_num:
                self.buffer[key].pop()

    def retrieve(self, nonce, mlm_inputs):
        curr_sentences = locs = (mlm_inputs['input_ids'] == nonce)
        tups = locs.nonzero(as_tuple=True)
        b, k, l = tups
        uq = list(set([(i, j) for i, j in zip(b, k)]))
        curr_examples = []
        for (i, j) in uq:
            mem = self.tokenizer.decode(mlm_inputs['input_ids'][i, j, :], skip_special_tokens=True,
                                        clean_up_tokenization_spaces=True)
            curr_examples.append(mem)

        if nonce not in self.buffer:
            return None
        else:
            if self.random:
                k = np.random.choice(list(range(1, self.k + 1)))
            else:
                k = self.k

            memories = [s for s in self.buffer[nonce] if s not in curr_examples]

            if len(memories) > k:
                samples = np.random.choice(memories, size=k).tolist()
            else:
                samples = memories
            if not self.cat:
                tokens = self.tokenizer(samples,
                                        max_length=self.tokenizer.model_max_length,
                                        truncation=True,
                                        padding='max_length',
                                        return_tensors='pt')
            else:
                sample = " ".join(samples)
                tokens = self.tokenizer(sample,
                                        max_length=self.tokenizer.model_max_length,
                                        truncation=True,
                                        padding='max_length',
                                        return_tensors='pt')
            return tokens
