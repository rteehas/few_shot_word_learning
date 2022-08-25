import torch
from torch import nn
import torch.functional as F

class OnlineProtoNet(nn.Module):

    def __init__(self, betas_r, gammas_r, betas_w, gammas_w, distance):

        self.memory = {}
        self.betas_r = betas_r
        self.gammas_r = gammas_r
        self.betas_w = betas_w
        self.gammas_w = gammas_w
        self.dist = distance


    def forward(self):
        pass

    def avg(self, p_last, c_last, embed):
        c = c_last + 1
        next_val = (1 / c) @ (p_last @ c_last + embed)
        return c, next_val

    def get_novelty(self, t, input, embed):
        dists = self.dist(embed, self.query(input))
        return torch.sigmoid(torch.min(dists) - self.betas_w[t]/self.gammas_w[t])

    # def storestore_example_example(self, x, embed, c_last):

    def query(self, input):
        pass

