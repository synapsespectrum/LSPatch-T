import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        """
        patch_len: the length of the patch for patch embedding
        stride: the stride for patch embedding
        """
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = configs.stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(configs.d_model, configs.patch_len, configs.stride, padding, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            encoder_layers_1=[
                EncoderLayer(
                    attention=AttentionLayer(
                        attention_mechanism=FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                          output_attention=configs.output_attention),
                        d_model=configs.d_model, n_heads=configs.n_heads),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ], norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Prediction Head
        self.head_nf = configs.d_model * int((configs.seq_len - configs.patch_len) / configs.stride + 2)  # number of patches
        self.head = FlattenHead(configs.enc_in, self.head_nf, self.pred_len, configs.dropout)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()  # [Batch, 1, seq_len]
        x_enc = x_enc - means  # [Batch, seq_len, n_vars]
        # Calculate the standard deviation
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev  # Change the scale to stabilize the training
        # Patching and embedding
        x_enc = x_enc.permute(0, 2, 1)  # [Batch, n_vars, seq_len]
        enc_out, n_vars = self.patch_embedding(x_enc)  # [Batch*n_vars, patch_num, d_model]
        # Encoder
        enc_out, attns = self.encoder(enc_out)  # [Batch*n_vars, patch_num, d_model]
        # [Batch*n_vars, patch_num, d_model ]->        [Batch, n_vars, patch_num, d_model]
        enc_out = torch.reshape(enc_out, shape=(-1, n_vars, enc_out.shape[1], enc_out.shape[2]))
        enc_out = enc_out.permute(0, 1, 3, 2)  # permute to [Batch, n_vars, d_model, patch_num]

        # Prediction Head
        x = self.head(enc_out)  # [Batch, n_vars, pred_len]
        x = x.permute(0, 2, 1)  # [Batch, pred_len, n_vars]
        # De-normalization
        x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)) + \
            means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return x[:, -self.pred_len:, :]  # [Batch, pred_len, n_vars]
