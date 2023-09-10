from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dim1, output_dim):
                    super(MLP, self).__init__()
                            self.dense1 = nn.Linear(input_dim, hidden_dim1)
                                    self.dense2 = nn.Linear(hidden_dim1, hidden_dim1)
                                            self.dense3 = nn.Linear(hidden_dim1, output_dim)
                                                def forward(self, x):
                                                            x = F.relu(self.dense1(x))
                                                                    x = F.relu(self.dense2(x))
                                                                            x = self.dense3(x)
                                                                                    return x
