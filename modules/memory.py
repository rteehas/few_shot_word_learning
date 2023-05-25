import torch
from torch import nn
from modules.aggregators import *

class OnlineProtoNet(nn.Module):

    def __init__(self, config, device):
        super(OnlineProtoNet, self).__init__()

        self.config = config
        self.agg_method = self.config.agg_method

        self.device = device
        self.memory = {}

        if self.agg_method == "CLS":

            input_size = self.config.input_size
            nhead = self.config.nhead
            num_layers = self.config.num_layers
            hidden_size = self.config.hidden_size
            num_positions = self.config.num_positions

            self.agg = TransformerSummarizerWithCLS(input_size, nhead, num_layers, hidden_size, num_positions,
                                                    self.device).to(self.device)

        elif self.agg_method == "RNN":

            input_size = self.config.input_size
            output_size = self.config.output_size

            self.agg = BiGRUSummarizer(input_size, output_size).to(self.device)

    def store(self, word, embed):

        if word not in self.memory:
            self.memory[word] = [embed]
        else:

            self.memory[word].append(embed)

    def retrieve(self, word):  # retrieves embed after linear layer to fix dims if needed

        ctx = torch.cat(self.memory[word], dim=0)
        if self.agg_method == "mean":
            return torch.mean(ctx, dim=0).unsqueeze(0)
        else:
            return self.agg(ctx.unsqueeze(0))

    def detach_past(self):
        for word in self.memory:
            self.memory[word] = [v.detach() for v in self.memory[word]]

    def clear_memory(self):
        self.memory = {}
