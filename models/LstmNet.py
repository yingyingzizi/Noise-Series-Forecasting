import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs, device):
        super().__init__()
        self.pred_len = configs.pred_len
        self.enc_in   = configs.enc_in
        self.hidden   = configs.d_model or 32
        n_layers      = configs.lstm_layers or 1  # LSTM 层数，默认为 1

        self.encoder = nn.LSTM(
            input_size  = self.enc_in,
            hidden_size = self.hidden,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = configs.dropout if n_layers > 1 else 0.0)

        self.proj = nn.Linear(self.hidden, self.pred_len * configs.c_out)

    def forward(self, x_enc, *_):
        _, (h_n, _) = self.encoder(x_enc)   # [n_layers,B,hidden]
        last_h = h_n[-1]
        out = self.proj(last_h)             # [B, pred_len*c_out]
        return out.view(out.size(0), self.pred_len, -1)