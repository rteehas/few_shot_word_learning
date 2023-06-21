import numpy as np

class RetrievalBuffer():

    def __init__(self, max_num, k, nonce_list):
        self.k = k
        self.max_num = max_num
        self.buffer = {}
        self.nonces = nonce_list


    def store(self, mlm_inputs):

        for n in self.nonces:
            if n not in mlm_inputs['input_ids']:
                continue
            else:
                locs = (mlm_inputs['input_ids'] == n)
                tups = locs.nonzero(as_tuple=True)
                b, k, l = tups
                uq = list(set([(i ,j) for i ,j in zip(b ,k)]))
                for (i ,j) in uq:
                    mem = {'input_ids': mlm_inputs['input_ids'][i ,j ,:].detach().unsqueeze(0),
                           'attention_mask': mlm_inputs['attention_mask'][i ,j ,:].detach().unsqueeze(0)}

                    if n in self.buffer:
                        self.buffer[n] = [mem] + self.buffer[n]
                    else:
                        self.buffer[n] = [mem]

    def cleanup(self):
        for key in self.buffer:
            while len(self.buffer[key]) > self.max_num:
                self.buffer[key].pop()

    def retrieve(self, nonce):
        if len(self.buffer[nonce]) > self.k:
            return np.random.choice(self.buffer[nonce], size=self.k).tolist()
        else:
            return self.buffer[nonce]


