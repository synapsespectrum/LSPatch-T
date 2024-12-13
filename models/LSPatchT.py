import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding, PatchMaskedEmbedding

from utils.RevIN import RevIN


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, d_model, head_dropout=0):
        super().__init__()
        head_dim = d_model * nf
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(head_dim, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)  # [bs x nvars x (d_model x patch_num)]
        x = self.linear(x) # [bs x nvars x target_window]
        x = self.dropout(x)
        return x


class PretrainHead(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        """
        x: tensor [bs x nvars x d_model x num_patch]
        output: tensor [bs x nvars x num_patch x patch_len]
        """

        # x = x.transpose(2, 3)  # [bs x nvars x num_patch x d_model]
        x = self.linear(self.dropout(x))  # [bs x nvars x num_patch x patch_len]
        x = x.permute(0, 2, 1, 3)  # [bs x num_patch x nvars x patch_len]
        return x


class PretrainModel(nn.Module):
    def __init__(self, configs):
        super(PretrainModel, self).__init__()
        padding = 0

        self.revin_layer = RevIN(configs.enc_in)  # enc_in is the number of input variables

        # patching and embedding
        self.patch_embedding = PatchMaskedEmbedding(configs.d_model, configs.patch_len, configs.stride,
                                                    padding, configs.dropout,
                                                    embed_type='learned',
                                                    seq_len=configs.seq_len,
                                                    mask_ratio=configs.mask_ratio)

        # Encoder
        self.encoder = Encoder(
            encoder_layers_1=[
                EncoderLayer(
                    attention=AttentionLayer(
                        attention_mechanism=FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                          output_attention=configs.output_attention),
                        d_model=configs.d_model, n_heads=configs.n_heads,
                        relative_position_embedding=False),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ], norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Pretrain Head
        self.head = PretrainHead(configs.d_model, configs.patch_len, configs.dropout)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Normalization from Non-stationary Transformer
        n_vars = x_enc.shape[2]  # number of variables
        x_enc = self.revin_layer(x_enc, 'norm')  # Calculate mean and std for [Batch, 1, seq_len]
        # Patching and embedding
        x_enc = x_enc.permute(0, 2, 1)  # [Batch, n_vars, seq_len]
        x_masked, x_patch, mask = self.patch_embedding(x_enc)  # [Batch*n_vars, patch_num, d_model]
        # Encoder
        enc_out, attns = self.encoder(x_masked)  # [Batch*n_vars, patch_num, d_model]
        # [Batch*n_vars, patch_num, d_model ]->        [Batch, n_vars, patch_num, d_model]
        enc_out = torch.reshape(enc_out, shape=(-1, n_vars, enc_out.shape[1], enc_out.shape[2]))
        # enc_out = enc_out.permute(0, 1, 3, 2)  # permute to [Batch, n_vars, d_model, patch_num]

        # Pretrain Head
        x = self.head(enc_out)  # [bs x num_patch x nvars x patch_len]

        # # De-normalization
        # x = self.revin_layer(x, 'denorm')

        return x, x_patch, mask  # x is patches the prediction, x_patch is the un-masked input


class DownStreamingModel(nn.Module):
    def __init__(self, configs):
        """
        patch_len: the length of the patch for patch embedding
        stride: the stride for patch embedding
        """
        super(DownStreamingModel, self).__init__()
        padding = configs.stride

        self.revin_layer = RevIN(configs.enc_in)  # enc_in is the number of input variables

        # patching and embedding
        self.patch_embedding = PatchEmbedding(configs.d_model, configs.patch_len, configs.stride,
                                              padding, configs.dropout)

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
        num_patch = (max(configs.seq_len, configs.patch_len) - configs.patch_len + padding) // configs.stride + 1
        self.head = FlattenHead(configs.enc_in, num_patch, configs.pred_len, configs.d_model, configs.dropout)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Normalization from Non-stationary Transformer
        x_enc = self.revin_layer(x_enc, 'norm')
        # Patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        # Encoder
        enc_out, attns = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, shape=(-1, n_vars, enc_out.shape[1], enc_out.shape[2]))
        enc_out = enc_out.permute(0, 1, 3, 2)  # permute to [Batch, n_vars, d_model, patch_num]

        # Prediction Head
        x = self.head(enc_out)  # [Batch, n_vars, pred_len]
        x = x.permute(0, 2, 1)  # [Batch, pred_len, n_vars]
        # De-normalization
        x = self.revin_layer(x, 'denorm')  # [Batch, pred_len, n_vars]

        return x


def transfer_weights(weights_path, model, exclude_head=True, device='cpu'):
    # state_dict = model.state_dict()
    new_state_dict = torch.load(weights_path, map_location=device)
    # print('new_state_dict',new_state_dict)
    new_state_dict = {k.replace('model.', ''): v for k, v in new_state_dict.items()}
    matched_layers = 0
    unmatched_layers = []
    for name, param in model.state_dict().items():
        if exclude_head and 'head' in name: continue
        if name in new_state_dict:
            matched_layers += 1
            input_param = new_state_dict[name]
            if input_param.shape == param.shape:
                param.copy_(input_param)
            elif 'value_embedding' in name:
                # try to duplicate the value embedding from Pretrain model to Downstream model: from shape [12, 512] to [96, 512] by repeating
                param_repeat = input_param.repeat(1, 8)
                param.copy_(param_repeat)
            else:
                unmatched_layers.append(name)
        else:
            unmatched_layers.append(name)
            pass  # these are weights that weren't in the original model, such as a new head
    if matched_layers == 0:
        raise Exception("No shared weight names were found between the models")
    else:
        if len(unmatched_layers) > 0:
            print(f'check unmatched_layers: {unmatched_layers}')
        else:
            print(f"weights from {weights_path} successfully transferred!\n")
    model = model.to(device)
    return model


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        if configs.is_pretrain:
            self.model = PretrainModel(configs)
        else:
            self.model = DownStreamingModel(configs)
            if configs.pretrained_model is not None:
                self.model = transfer_weights(configs.pretrained_model, self.model)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
