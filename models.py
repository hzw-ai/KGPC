import os.path

import torch
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from torch import nn
import math
from torch.nn import functional as F
from torch.autograd import Variable
import torch.optim as optim


# GCN
class GraphConvolution(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, num_relations, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.num_relations = num_relations
        self.alpha = torch.nn.Embedding(num_relations + 1, 1, padding_idx=0)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        alp = self.alpha(adj[1]).t()[0]
        A = torch.sparse_coo_tensor(adj[0], alp, torch.Size([adj[2], adj[2]]), requires_grad=True)
        A = A + A.transpose(0, 1)
        support = torch.mm(input, self.weight)
        output = torch.sparse.mm(A, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output


class SACN(torch.nn.Module):
    def __init__(self, num_entities, num_relations, init_emb_size, gc1_emb_size, embedding_dim, input_dropout,
                 dropout_rate, channels, kernel_size, batch_size, **kwargs):
        super(SACN, self).__init__()

        self.is_subtract_sacn = False
        self.is_subtract_ail = False

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.init_emb_size = init_emb_size
        self.gc1_emb_size = gc1_emb_size
        self.embedding_dim = embedding_dim
        self.input_dropout = input_dropout
        self.dropout_rate = dropout_rate
        self.channels = channels
        self.kernel_size = kernel_size
        self.batch_size = batch_size

        if self.is_subtract_sacn:
            self.emb_e = torch.nn.Embedding(num_entities, embedding_dim, padding_idx=0)
        else:
            self.emb_e = torch.nn.Embedding(num_entities, init_emb_size, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, embedding_dim, padding_idx=0)
        self.gc1 = GraphConvolution(init_emb_size, gc1_emb_size, num_relations)
        self.gc2 = GraphConvolution(gc1_emb_size, embedding_dim, num_relations)
        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(dropout_rate)
        self.feature_map_drop = torch.nn.Dropout(dropout_rate)
        self.conv1 = nn.Conv1d(2, channels, kernel_size, stride=1, padding=int(
            math.floor(kernel_size / 2)))  # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)
        self.bn3 = torch.nn.BatchNorm1d(gc1_emb_size)
        self.bn4 = torch.nn.BatchNorm1d(embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(init_emb_size)

        # FIEGNN
        self.dense_user_onehop_biinter = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.dense_user_onehop_siinter = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.dense_user_cate_self = nn.Linear(2 * self.embedding_dim, self.embedding_dim)
        self.leakyrelu = nn.LeakyReLU()

        # FC
        self.fc2classes = torch.nn.Linear(embedding_dim, kwargs['n_clusters'])

        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.gc1.weight.data)
        xavier_normal_(self.gc2.weight.data)
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)
		
    def set_emb(self, index, emb):
        self.emb_e.weight.data[index] = emb

    def kl_loss(self, q):
        weight = q ** 2 / q.sum(0)
        p = (weight.t() / weight.sum(1)).t()
        return F.kl_div(q.log(), p, reduction='batchmean')

    def feat_interaction(self, feature_embedding, f1, f2, dimension):
        summed_features_emb_square = (torch.sum(feature_embedding, dim=dimension)) ** 2
        squared_sum_features_emb = torch.sum(feature_embedding ** 2, dim=dimension)
        deep_fm = 0.5 * (summed_features_emb_square - squared_sum_features_emb)
        deep_fm = F.leaky_relu(f1(deep_fm))
        bias_fm = F.leaky_relu(f2(feature_embedding.sum(dim=dimension)))
        nfm = deep_fm + bias_fm
        return nfm

    def forward(self, e1, rel, attr, X, A):
        """

        :param e1:
        :param rel:
        :param attr:
        :param X:
        :param A: 邻接矩阵
        :return:
        """
        batch_size = e1.shape[0]


        # SACN
        emb_initial = self.emb_e(X)
        if not self.is_subtract_sacn:
            x = self.gc1(emb_initial, A)
            x = self.bn3(x)
            x = torch.tanh(x)
            x = F.dropout(x, self.dropout_rate, training=self.training)

            x = self.bn4(self.gc2(x, A))
            e1_embedded_all = torch.tanh(x)
            e1_embedded_all = F.dropout(e1_embedded_all, self.dropout_rate, training=self.training)
            e1_embedded = e1_embedded_all[e1]
            rel_embedded = self.emb_rel(rel)
            stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
            stacked_inputs = self.bn0(stacked_inputs)
            x = self.inp_drop(stacked_inputs)
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.feature_map_drop(x)
            x = x.view(batch_size, -1)
            x = self.fc(x)
            x = self.hidden_drop(x)
            x = self.bn2(x)
            x = F.relu(x)

        # 属性交互
        if not self.is_subtract_ail:
            if self.is_subtract_sacn:
                e1_embedded_all = emb_initial
                user_embedding = e1_embedded_all[e1].view(batch_size, -1)
                attr_emb = e1_embedded_all[attr]
            else:
                user_embedding = x
                attr_emb = e1_embedded_all[attr]

            user_self_feat = F.dropout(
                torch.cat([attr_emb[:, 0, i, :].unsqueeze(1) for i in range(attr.shape[-1])], dim=1),
                self.dropout_rate, training=self.training)
            user_self_feat = self.feat_interaction(user_self_feat, self.dense_user_onehop_biinter,
                                                   self.dense_user_onehop_siinter, dimension=1)
            user_self_embed = self.dense_user_cate_self(torch.cat([user_self_feat, user_embedding], dim=-1))
            user_gcn_embed = self.leakyrelu(user_self_embed)
        else:
            user_gcn_embed = x

        x = torch.mm(user_gcn_embed, e1_embedded_all.transpose(1, 0))
        pred = torch.sigmoid(x)

        user_gcn_embed = self.fc2classes(user_gcn_embed)
        user_gcn_embed = torch.sigmoid(user_gcn_embed)

        return pred, user_gcn_embed
