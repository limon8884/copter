import torch.nn as nn

class Network(nn.Module):
    '''
    This network will be uploaded to arduino, so it should not be large.
    Will be traided via RL approach.
    '''
    def __init__(self, in_channels, out_channels, n_hidden=10, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels # state tensor
        self.out_channels = out_channels # out probability distributions
        self.n_hidden = n_hidden # num of hidden states

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            # nn.Dropout(dropout),
            # nn.BatchNorm1d(n_hidden),
            # nn.ReLU(),
            nn.Linear(n_hidden, out_channels),
            # nn.Softmax(),
        ) 
        

    def forward(self, x):
        x = self.mlp(x)
        return x
    

