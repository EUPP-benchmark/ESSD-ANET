import torch
from torch import nn
import numpy as np

class Model(nn.Module):

    EPS = 1e-12

    def __init__(self, number_of_predictors, lead_time, output_features, hidden_features = 64):

        super().__init__()

        self.lead_time = lead_time
        self.output_features = output_features

        # Little helper functions

        def dense(in_f, out_f, bnorm = True):

            if bnorm:
                return nn.Sequential(nn.Linear(in_f, out_f, dtype = torch.float32), nn.ReLU(), nn.BatchNorm1d(out_f))
            else:
                return nn.Sequential(nn.Linear(in_f, out_f, dtype = torch.float32), nn.ReLU())

        # Define encoder block

        self.E = nn.Sequential(\
                dense(number_of_predictors + lead_time, hidden_features, bnorm = False),\
                dense(hidden_features, hidden_features, bnorm = False))

        # Define attention block

        self.A = nn.Sequential(
                dense(hidden_features, hidden_features, bnorm = False),\
                nn.Linear(hidden_features, 1, dtype = torch.float32))

        # Define regression block

        self.R = nn.Sequential(\
                dense(hidden_features, hidden_features, bnorm = False),\
                dense(hidden_features, hidden_features, bnorm = False),\
                nn.Linear(hidden_features, output_features*lead_time, dtype = torch.float32))

    def forward(self, x, p):

        # x: [batch, lead, members]
        # p: [batch, predictors]

        m = x.mean(dim = -1)
        s = x.std(dim = -1)

        n_batch_members = x.shape[2]
        batch_size = x.shape[0]
       
        p = torch.unsqueeze(p, dim = -1).expand(-1, -1, x.shape[-1])
        p = torch.swapaxes(p, 1, -1)
        x = torch.swapaxes(x, 1, -1)
        x = torch.cat([x, p], dim = -1)

        D = self.E(x)
        A = self.A(D)

        D = D*A

        Y = self.R(torch.squeeze(D.mean(dim = 1), dim = 1))
        Y = torch.reshape(Y, (Y.shape[0], self.lead_time, self.output_features))

        a = Y[:, :, 0]
        b = Y[:, :, 1]

        return torch.stack([m + a, nn.Softplus()(s + b)], dim = -1)

    def loss(self, x, p, y):

        d = self(x, p)

        m = d[:, :, 0]
        s = d[:, :, 1]

        idx = ~torch.isnan(y)

        m = m[idx]
        s = torch.clamp(s[idx], min = self.EPS)
        y = y[idx]

        T0 = torch.log(s)
        T1 = 0.5*torch.pow((m - y)/s, 2)

        L = T0 + T1
        return L.mean()

