
import torch.nn as nn

class Simple_MLP(nn.Module):


    def __init__(self, **kwargs):

        super(Simple_MLP, self).__init__()
        self.expected_input_size = (28, 28)

        self.main = nn.Sequential(nn.Linear(28*28*3, 20, bias=False),
                                  nn.ReLU(),
                                  nn.Linear(20, 10, bias=False))

    def forward(self, input):
        out = input.view(input.size(0), -1)
        out = self.main(out)
        return out
