
import os
import time
import pickle
import torch as t
import numpy as np
from torch.utils import data
import gzip
from time import time
from config import DefaultConfig
import torch
import dgl
import threading

class dataSet(data.Dataset):
    def __init__(self, root_dir, protein_list_file):
        super(dataSet, self).__init__()

        self.edge_feat_mean = [31.83509173, 1.56021911] #calculated from trainset only
        self.edge_feat_std = [16.79204272, 0.69076342] #calculated from trainset only

        self.all_protBert_feature = pickle.load(gzip.open(root_dir+'/inputs/ProtBert_features.pkl.gz', "rb"))['ProtBert_features']
        self.all_dist_matrix = pickle.load(gzip.open(root_dir+'/inputs/ppisp_dist_matrix_map.pkl.gz', 'rb'))
        self.all_angle_matrix = pickle.load(gzip.open(root_dir+'/inputs/ppisp_angle_matrix_map.pkl.gz', 'rb'))

        print('protein_list_file:', protein_list_file)
        with open(protein_list_file, "r") as f:
            protein_list = f.readlines()
            self.protein_list = [x.strip() for x in protein_list]

        self.config = DefaultConfig()
        self.max_seq_len = self.config.max_sequence_length
        self.neighbourhood_size = 21

        self.protein_list_len = len(self.protein_list)

        self.all_graphs = self.generate_all_graphs()
        print('All graphs generated.')

    def __getitem__(self, index):
        t0=time()
        protein_name = self.protein_list[index]
        id_idx = index

        _all_protBert_feature_ = self.all_protBert_feature[id_idx][:self.max_seq_len]
        seq_len = _all_protBert_feature_.shape[0]
        protein_info = {
            'protein_name': protein_name,
            'protein_idx': id_idx,
            'seq_length': seq_len
        }
        if seq_len < self.max_seq_len:
            temp = np.zeros([self.max_seq_len, _all_protBert_feature_.shape[1]])
            temp[:seq_len, :] = _all_protBert_feature_
            _all_protBert_feature_ = temp

        _all_protBert_feature_ = _all_protBert_feature_[np.newaxis, :, :]
        G = self.all_graphs[id_idx]
        return torch.from_numpy(_all_protBert_feature_).type(torch.FloatTensor), \
               G, \
               protein_info

    def __len__(self):
        return self.protein_list_len

    def generate_all_graphs(self):
        graph_list = {}
        for id_idx in self.all_dist_matrix:
            G = dgl.DGLGraph()
            G.add_nodes(self.max_seq_len)
            neighborhood_indices = self.all_dist_matrix[id_idx]['dist_matrix'][:self.max_seq_len, :self.max_seq_len, 0] \
                                       .argsort()[:, 1:self.neighbourhood_size]
            if neighborhood_indices.max() > self.max_seq_len-1 or neighborhood_indices.min() < 0:
                print(neighborhood_indices.max(), neighborhood_indices.min())
                raise
            edge_feat = np.array([
                self.all_dist_matrix[id_idx]['dist_matrix'][:self.max_seq_len, :self.max_seq_len, 0],
                self.all_angle_matrix[id_idx]['angle_matrix'][:self.max_seq_len, :self.max_seq_len]
            ])
            edge_feat = np.transpose(edge_feat, (1, 2, 0))
            edge_feat = (edge_feat - self.edge_feat_mean) / self.edge_feat_std  # standardize features

            self.add_edges_custom(G,
                                  neighborhood_indices,
                                  edge_feat
                                  )
            graph_list[id_idx]= G
        return  graph_list

    def add_edges_custom(self, G, neighborhood_indices, edge_features):
        t1 = time()
        size = neighborhood_indices.shape[0]
        neighborhood_indices = neighborhood_indices.tolist()
        src = []
        dst = []
        temp_edge_features = []
        for center in range(size):
            src += neighborhood_indices[center]
            dst += [center] * (self.neighbourhood_size - 1)
            for nbr in neighborhood_indices[center]:
                temp_edge_features += [np.abs(edge_features[center, nbr])]
        if len(src) != len(dst):
            prit('source and destination array should have been of the same length: src and dst:', len(src), len(dst))
            raise Exception
        G.add_edges(src, dst)
        G.edata['ex'] = np.array(temp_edge_features)



def graph_collate(samples):
    protbert_data, graph_batch, protein_info = map(list, zip(*samples))
    graph_batch = dgl.batch(graph_batch)
    protbert_data = torch.cat(protbert_data)
    return protbert_data, graph_batch, protein_info

