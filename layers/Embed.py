import torch
import torch.nn as nn
import math


def compared_version(ver1, ver2):
    """
    :param ver1
    :param ver2
    :return: ver1< = >ver2 False/True
    """
    list1 = str(ver1).split(".")
    list2 = str(ver2).split(".")

    for i in range(len(list1)) if len(list1) < len(list2) else range(len(list2)):
        if int(list1[i]) == int(list2[i]):
            pass
        elif int(list1[i]) < int(list2[i]):
            return -1
        else:
            return 1

    if len(list1) == len(list2):
        return True
    elif len(list1) < len(list2):
        return False
    else:
        return True


# class PositionalEmbedding(nn.Module):
#     def __init__(self, d_model, max_len=5000, normalize=False):
#         super(PositionalEmbedding, self).__init__()
#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len, d_model).float()
#         pe.require_grad = False
#
#         position = torch.arange(0, max_len).float().unsqueeze(1)
#         div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
#
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#
#         if normalize:
#             pe = pe - pe.mean()
#             pe = pe / (pe.std() * 10)
#
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         return self.pe[:, :x.size(1)]

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if compared_version(torch.__version__, '1.5.0') else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):  # Transformers, Informer
    """
    Embedding the data to 3 components: value, positional, and temporal embeddings.

    Attributes:
        value_embedding (TokenEmbedding): Embeds the input values.
        position_embedding (PositionalEmbedding): Adds positional information to the embeddings.
        temporal_embedding (TemporalEmbedding or TimeFeatureEmbedding): Adds temporal information to the embeddings.
        dropout (nn.Dropout): Dropout layer to prevent overfitting.

    Methods:
        forward(x, x_mark):
            Combines the value, positional, and temporal embeddings and applies dropout.
    """

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        # c_in: the number of features
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' \
            else (TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):  # Autoformer
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


# Trainable positional embedding
class PositionalEmbeddingLearned(nn.Module):
    def __init__(self, d_model, q_len):
        super(PositionalEmbeddingLearned, self).__init__()
        self.W_pe = torch.empty((q_len, d_model))
        nn.init.uniform_(self.W_pe, -0.02, 0.02)
        self.W_pe = nn.Parameter(self.W_pe, requires_grad=True)
        # nn.Parameter is a Matrix wrapper that makes a tensor trainable.
        # it's different from nn.Embedding,
        # which is a layer that takes an index and returns the corresponding row in the weight matrix;
        # nn.Embedding is kind of like a dictionary lookup (a neural network layer that maps integers to vectors).
        # Example:
        # class LinearRegressionModel(nn.Module):
        #     def __init__(self):
        #         super().__init__()
        #         self.weights = nn.Parameter(torch.randn(5, 3)) # Here
        #         self.bias = nn.Parameter(torch.randn(5))  # Here
        #
        #     def forward(self, x):
        #         return self.weights * x + self.bias  # is equivalent to nn.Linear(3, 5, bias=True)

    def forward(self, x):
        return self.W_pe  # [Batch*n_vars, patch_num, d_model]


class Patching(nn.Module):  # PatchTST and PatchTST-Self-Supervised
    def __init__(self, d_model, patch_len, stride, padding, dropout, embed_type='fixed', seq_len=None):
        """
        embed_type: the type of embedding: fixed or learned
        """
        if type(self) is Patching:
            raise TypeError("Patching is an abstract class and cannot be instantiated directly.")
        super(Patching, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))  # padding to the end of this sequence

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding: Transform patch index to a d-dim vector space (dim: d_model)
        if embed_type == 'fixed':
            self.position_embedding = PositionalEmbedding(d_model=d_model)#, max_len=patch_len, normalize=True)
        else:  # learned
            num_patch = (max(seq_len, patch_len) - patch_len + padding) // stride + 1
            self.position_embedding = PositionalEmbeddingLearned(d_model=d_model, q_len=num_patch)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark=None):
        # Patching
        n_vars = x.shape[1]  # [Batch, n_vars, seq_len]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [Batch, n_vars, patch_num, patch_len]
        return x, n_vars


class PatchEmbedding(Patching):  # PatchTST
    def __init__(self, d_model, patch_len, stride, padding, dropout, embed_type='fixed'):
        super(PatchEmbedding, self).__init__(d_model, patch_len, stride, padding, dropout, embed_type)

    def forward(self, x, x_mark=None):
        # Patching
        x_patch, n_vars = super().forward(x, x_mark)
        x_patch = torch.reshape(x_patch,
                                shape=(x_patch.shape[0] * x_patch.shape[1],
                                       x_patch.shape[2], x_patch.shape[3]))  # [Batch*n_vars, patch_num, patch_len]
        # input encoding
        x = self.value_embedding(x_patch) + self.position_embedding(x_patch)  # [Batch*n_vars, patch_num, d_model]
        return self.dropout(x), n_vars


class RandomMasking(nn.Module):
    def __init__(self, mask_ratio=0.15):
        super(RandomMasking, self).__init__()
        self.mask_ratio = mask_ratio

    def forward(self, xb):
        xb = xb.permute(0, 2, 1, 3)  # [bs x num_patch x n_vars x patch_len]
        bs, L, nvars, D = xb.shape
        x = xb.clone()

        len_keep = int(L * (1 - self.mask_ratio))

        noise = torch.rand(bs, L, nvars, device=xb.device)  # noise in [0, 1], bs x L x nvars

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # ids_restore: [bs x L x nvars]

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep, :]  # ids_keep: [bs x len_keep x nvars]
        x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1,
                                                                            D))  # x_kept: [bs x len_keep x nvars  x patch_len]

        # removed x
        x_removed = torch.zeros(bs, L - len_keep, nvars, D,
                                device=xb.device)  # x_removed: [bs x (L-len_keep) x nvars x patch_len]
        x_ = torch.cat([x_kept, x_removed], dim=1)  # x_: [bs x L x nvars x patch_len]

        # combine the kept part and the removed one
        x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, 1,
                                                                                  D))  # x_masked: [bs x num_patch x nvars x patch_len]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([bs, L, nvars], device=x.device)  # mask: [bs x num_patch x nvars]
        mask[:, :len_keep, :] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)  # [bs x num_patch x nvars]
        return x_masked, mask


class PatchMaskedEmbedding(Patching):  # PatchTST-Self-Supervised
    def __init__(self, d_model, patch_len, stride, padding, dropout, embed_type='learned', seq_len=96, mask_ratio=0.4):
        super(PatchMaskedEmbedding, self).__init__(d_model, patch_len, stride, padding, dropout, embed_type, seq_len)
        self.random_masking = RandomMasking(mask_ratio)

    def forward(self, x, x_mark=None):
        # Patching
        x_patch, n_vars = super().forward(x, x_mark)  # [Batch, n_vars, patch_num, patch_len]
        # random masking
        x_masked, mask = self.random_masking(x_patch)  # [bs x num_patch x nvars x patch_len]
        x_patch = x_patch.permute(0, 2, 1, 3)  # [bs x num_patch x nvars x patch_len]
        x_masked = torch.reshape(x_masked,
                                 shape=(x_masked.shape[0] * x_masked.shape[2],
                                        x_masked.shape[1], x_masked.shape[3]))  # [Batch*n_vars, patch_num, patch_len]
        # input encoding
        x = self.value_embedding(x_masked) + self.position_embedding(x_masked)  # [Batch*n_vars, patch_num, d_model]
        return self.dropout(x), x_patch, mask  # [Batch*n_vars, patch_num, d_model], mask: [Batch, patch_num, n_vars]


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)  # [bs, n_vars + 4(temporal features), d_model]
