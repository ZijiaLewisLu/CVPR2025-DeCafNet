from torch import nn
import torch.nn.functional as F

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, nchannels, dropout=0.5, layernorm=True, layernorm_eps=1e-5, ngroup=1):
        super(DilatedResidualLayer, self).__init__()
        self.dilation = dilation
        self.nchannels = nchannels
        self.dropout_rate = dropout

        self.conv_dilated = nn.Conv1d(nchannels, nchannels, 3, padding=dilation, dilation=dilation, groups=ngroup)
        self.conv_1x1 = nn.Conv1d(nchannels, nchannels, 1)
        self.dropout = nn.Dropout(dropout)

        self.use_layernorm=layernorm
        if layernorm:
            self.norm = nn.LayerNorm(nchannels, eps=layernorm_eps)
        else:
            self.norm = None

    def forward(self, x, mask=None):
        """
        x: B, D, T
        """
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        if mask is not None:
            x = (x + out) * mask[:, 0:1, :]
        else:
            x = x + out

        if self.norm:
            x = x.permute(0, 2, 1) # B, T, D
            x = self.norm(x)
            x = x.permute(0, 2, 1) # B, D, T

        return x

class TCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, dropout=0.5, dilation_factor=2, ln=True, ngroup=1, in_map=False):
        super(TCN, self).__init__()
        if in_map:
            self.conv_1x1 = nn.Conv1d(in_dim, hid_dim, 1)
        else:
            assert in_dim == hid_dim

        self.layers = nn.ModuleList([DilatedResidualLayer(dilation_factor ** i, hid_dim, dropout, layernorm=ln, ngroup=ngroup) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(hid_dim, out_dim, 1)

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.in_map = in_map
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.dilation_factor = dilation_factor


    def __str__(self):
        return self.string 

    def __repr__(self):
        return str(self)

    def forward(self, x, mask=None):
        # assert mask is None

        if self.in_map:
            out = self.conv_1x1(x)
        else:
            out = x

        for layer in self.layers:
            out = layer(out, mask)

        out = self.conv_out(out)  # B, H, T
        # out = out.permute([2, 0, 1]) # T, 1, H 

        if mask is not None:
            out = out * mask[:, 0:1, :]

        self.output = out
        return self.output