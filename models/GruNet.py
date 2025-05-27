import torch
import torch.nn as nn

class Model(nn.Module):
    """
    简易 Seq2Seq GRU：
        Encoder: GRU  ≤3层
        Decoder: 取最后隐状态 → 线性映射到 pred_len 步
    只用到 batch_x，无需时间戳；可适配 Exp 框架的 4 个输入，
    其余参数仅占位，不会用到。
    """
    def __init__(self, configs, device):
        super().__init__()
        self.pred_len   = configs.pred_len
        self.enc_in     = configs.enc_in          #  = 特征数
        self.d_model    = configs.d_model or 32    # 隐状态维度
        n_layers        = configs.gru_layers or 1  # GRU 层数，默认为 1

        self.encoder = nn.GRU(
            input_size = self.enc_in,
            hidden_size= self.d_model,
            num_layers = n_layers,
            batch_first= True,
            dropout    = configs.dropout if n_layers > 1 else 0.0
        )
        # 将最后一个 hidden_state 线性映射成 pred_len × c_out
        self.proj = nn.Linear(self.d_model, self.pred_len * configs.c_out)

    def forward(self, x_enc, *_):
        """
        兼容 Exp 的 forward(api)，后 3 个参数忽略
        x_enc: [B, seq_len, enc_in]
        return : [B, pred_len, c_out]
        """
        _, h_n = self.encoder(x_enc)             # h_n: [n_layers, B, d_model]
        last_hidden = h_n[-1]                    # [B, d_model]
        out = self.proj(last_hidden)             # [B, pred_len*c_out]
        return out.view(out.size(0), self.pred_len, -1)