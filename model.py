import torch
from torch import nn
from torchtuples import tuplefy
import pdb

def init_embedding(emb):
    """Weight initialization of embeddings (in place).
    Best practise from fastai
    
    Arguments:
        emb {torch.nn.Embedding} -- Embedding
    """
    w = emb.weight.data
    sc = 2 / (w.shape[1]+1)
    w.uniform_(-sc, sc)

def _accuracy(input, target):
    """Accuracy, i.e. mean(input == target)"""
    return input.eq(target.view_as(input)).float().mean()

def accuracy_binary(input, target):
    """Accuracy for binary models on input for logit models in (-inf, inf).
    Do not used for models with sigmoid output activation.
    """
    if len(input.shape) == 1:
        raise NotImplementedError(f"`accuracy_argmax` not implemented for shape {input.shape}")
    assert (target.min() == 0) and (target.max() == 1), 'We have binary classfication so we need 0/1'
    pred = torch.zeros_like(input).to(target.dtype)
    pred[input >= 0.] = 1
    return _accuracy(pred, target)

def accuracy_argmax(input, target):
    """Accuracy after argmax on input for logit models in (-inf, inf).
    Do not used for models with sigmoid/softmax output activation.

    Tupycally used as a metric passed to Model.fit()
    If input is one dimensional, we assume we have binary classification.
    """
    if len(input.shape) == 1:
        raise NotImplementedError(f"`accuracy_argmax` not implemented for shape {input.shape}")
    if input.shape[1] == 1:
        raise NotImplementedError("`accuracy_argmax` not for binary data. See `accuracy_binary` instead.")
    else:
        pred = input.argmax(dim=1, keepdim=True)
    return _accuracy(pred, target)


class DenseVanillaBlock(nn.Module):
    def __init__(self, in_features, out_features, bias=True, batch_norm=True, dropout=0., activation=nn.ReLU,
                 w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        if w_init_:
            w_init_(self.linear.weight.data)
        self.activation = activation()
        self.batch_norm = nn.BatchNorm1d(out_features) if batch_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, input):
        input = self.activation(self.linear(input))
        if self.batch_norm:
            input = self.batch_norm(input)
        if self.dropout:
            input = self.dropout(input)
        return input


class MLPVanilla(nn.Module):
    def __init__(self, in_features, num_nodes, out_features, batch_norm=True, dropout=None, activation=nn.ReLU,
                 output_activation=None, output_bias=True,
                 w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()
        num_nodes = tuplefy(in_features, num_nodes).flatten()
        if not hasattr(dropout, '__iter__'):
            dropout = [dropout for _ in range(len(num_nodes)-1)]
        net = []
        for n_in, n_out, p in zip(num_nodes[:-1], num_nodes[1:], dropout):
            net.append(DenseVanillaBlock(n_in, n_out, True, batch_norm, p, activation, w_init_))
        net.append(nn.Linear(num_nodes[-1], out_features, output_bias))
        if output_activation:
            net.append(output_activation)
        self.net = nn.Sequential(*net)
        self.bn = torch.nn.BatchNorm1d(in_features)
    def forward(self, input):
        return self.net(self.bn(input))


class EntityEmbeddings(nn.Module):
    def __init__(self, num_embeddings, embedding_dims, dropout=0.):
        super().__init__()
        if not hasattr(num_embeddings, '__iter__'):
            num_embeddings = [num_embeddings]
        if not hasattr(embedding_dims, '__iter__'):
            embedding_dims = [embedding_dims]
        if len(num_embeddings) != len(embedding_dims):
            raise ValueError("Need 'num_embeddings' and 'embedding_dims' to have the same length")
        self.embeddings = nn.ModuleList()
        for n_emb, emb_dim in zip(num_embeddings, embedding_dims):
            emb = nn.Embedding(n_emb, emb_dim)
            init_embedding(emb)
            self.embeddings.append(emb)
        self.dropout = nn.Dropout(dropout) if dropout else None
    
    def forward(self, input):
        if input.shape[1] != len(self.embeddings):
            raise RuntimeError(f"Got input of shape '{input.shape}', but need dim 1 to be {len(self.embeddings)}.")
        
        input = [emb(input[:, i]) for i, emb in enumerate(self.embeddings)]
        input = torch.cat(input, 1)
        if self.dropout:
            input = self.dropout(input)
        return input


class MixedInputMLP(nn.Module):
    def __init__(self, in_features, num_embeddings, embedding_dims, num_nodes, out_features,
                 batch_norm=True, dropout=None, activation=nn.ReLU, dropout_embedding=0.,
                 output_activation=None, output_bias=True,
                 w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()
        self.embeddings = EntityEmbeddings(num_embeddings, embedding_dims, dropout_embedding)
        input_mlp = in_features + sum(embedding_dims)
        self.mlp = MLPVanilla(input_mlp, num_nodes, out_features, batch_norm, dropout, activation,
                              output_activation, output_bias, w_init_)
        self.bn = torch.nn.BatchNorm1d(input_mlp)
    def forward(self, input_numeric, input_categoric):
        input = torch.cat([input_numeric, self.embeddings(input_categoric)], 1)
        return self.mlp(self.bn(input))


class Transformer(nn.Module):
    def __init__(self,in_features, num_embeddings, num_nodes, out_features,
                 batch_norm=True, dropout=None, activation=nn.ReLU, dropout_embedding=0.,
                 output_activation=None, output_bias=True,
                 w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()
        if not hasattr(num_embeddings, '__iter__'):
            num_embeddings = [num_embeddings]
            
        emb_dim = num_nodes[0]


        self.emb_numeric = nn.Embedding(in_features, emb_dim)
        
        self.embeddings = nn.ModuleList()
        for n_emb in num_embeddings:
            emb = nn.Embedding(n_emb, emb_dim)
            init_embedding(emb)
            self.embeddings.append(emb)
            
        
        self.encoder_layer1 = torch.nn.TransformerEncoderLayer(d_model=emb_dim, nhead=4)
        self.encoder_layer2 = torch.nn.TransformerEncoderLayer(d_model=emb_dim, nhead=4)
        
        self.mlp = MLPVanilla(emb_dim, num_nodes, out_features, batch_norm, dropout, activation,
                              output_activation, output_bias, w_init_)
    
        self.mask_pad = torch.zeros(1,len(num_embeddings)).bool()
    def forward(self,input_numeric,input_categoric):
        emb_numeric = self.emb_numeric.weight.unsqueeze(0).repeat(input_numeric.shape[0],1,1)
        emb_numeric = (input_numeric.unsqueeze(2) *emb_numeric).permute(1,0,2)
        emb_categoric = torch.cat([emb(input_categoric[:, i]).unsqueeze(1) for i, emb in enumerate(self.embeddings)],dim=1).permute(1,0,2)
        emb = torch.cat([emb_numeric,emb_categoric],dim=0)
        mask_pad = torch.cat([input_numeric==0, self.mask_pad.to(input_numeric.device).repeat(input_numeric.shape[0],1)],dim=1)
        
        emb = self.encoder_layer1(src=emb, src_key_padding_mask=mask_pad)
        emb = self.encoder_layer2(src=emb, src_key_padding_mask=mask_pad)
        emb = emb.mean(dim=0)
        output = self.mlp(emb)
        return output

'''
class TransformerModule(nn.Module):
    def __init__(self, hidden_channels, num_layers=1,num_heads=4, dropout=0.1):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(TransformerLayer( hidden_channels,num_heads, dropout))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, src, key, query, attn_mask, key_padding_mask, mask):
        output = src
        for layer in self.layers:
            output = layer(output, key, query, attn_mask, key_padding_mask, mask)
        return output

class TransformerLayer(nn.Module):
    def __init__(self, hidden_channels,num_heads, dropout):
        super(TransformerLayer, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=hidden_channels, num_heads=num_heads, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_channels, hidden_channels)

        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu
        self.reset_parameters()

    def reset_parameters(self):
        self.attn._reset_parameters()
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()

    def forward(self, src, key, query, attn_mask, key_padding_mask):
        src2 = self.attn(query, key, value=src, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]
        src = src + self.dropout1(src2)           # Add
        src = self.norm1(src)                     # & Norm
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))  # FFN
        src = src + self.dropout2(src2)           # Add
        src = self.norm2(src)                     # & Norm
        return src
'''
