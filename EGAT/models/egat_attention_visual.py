
import os
import time
import sys
import numpy as np

import torch as t
from torch import nn
from torch.autograd import Variable


sys.path.append("../")
from config import DefaultConfig

configs = DefaultConfig()

class egat_ppi(nn.Module):
    def __init__(self, ratio=None):
        super(egat_ppi, self).__init__()
        global configs

        ####
        self.outLayer = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid())
        ####
        self.conv_encoder = nn.Sequential(
            nn.Conv1d(in_channels=1024,
                      out_channels=32,
                      kernel_size=7, stride=1,
                      padding=7 // 2, dilation=1, groups=1,
                      bias=True, padding_mode='zeros'),
            nn.LeakyReLU(negative_slope=.01),
            nn.BatchNorm1d(num_features=32,
                           eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(.5)  # it was .2
        )

        from models import EdgeAggregatedGAT_attention_visual as egat
        config_dict = egat.config_dict
        config_dict['feat_drop'] = 0.2
        config_dict['edge_feat_drop'] = 0.1
        config_dict['attn_drop'] = 0.2

        self.gat_layer = egat.MultiHeadEGATLayer(
                                    in_dim=32,
                                    out_dim=32,
                                    edge_dim=2,
                                    num_heads=1,
                                    use_bias=False,
                                    merge='cat',
                                    config_dict=config_dict)

    def forward(self, protbert_feature, graph_batch):
        shapes = protbert_feature.data.shape
        features = protbert_feature.squeeze(1).permute(0, 2, 1)
        features = self.conv_encoder(features)
        features = features.permute(0, 2, 1).contiguous()
        # print('1', features.shape, shapes, features.is_contiguous())
        # print(features.view([-1, 32]).shape)
        # features2 = self.multi_CNN(features)
        features2, head_attn_scores = self.gat_layer(graph_batch, features.view([shapes[0]*500, 32]))
        features2 = features2.view([shapes[0], 500, 32])
        # print('features2.shape, features.shape:', features2.shape, features.shape)
        #z.repeat(1,2).view(shapes[0],2,shapes[1])
        features = t.cat((features2, features), 2)
        # print(features2.shape, features.shape)
        # features = features.view([shapes[0], 500, 32*2])[t.nonzero(label_idx_onehot.view([shapes[0], 500])==1, as_tuple=True)]

        features = self.outLayer(features)
        # print('output size', features.shape)
        return features, head_attn_scores