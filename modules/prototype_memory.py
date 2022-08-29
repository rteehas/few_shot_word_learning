import torch
from torch import nn
import torch.functional as F

class OnlineProtoNet(nn.Module):

    def __init__(self):
        super(OnlineProtoNet, self).__init__()
        self.memory = {}

    def store(self, word, embed):
        if word not in self.memory:
            self.memory[word] = (embed, 1)
        else:
            self.memory[word] = (torch.div(embed + self.memory[word][0], self.memory[word][1]), self.memory[word][1] + 1)
