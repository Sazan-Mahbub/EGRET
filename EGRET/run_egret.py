from time import time
t000 = time()

import os


import pickle
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.nn.init import xavier_normal,xavier_normal_
from torch import nn
import torch.utils.data.sampler as sampler

import dgl

from config import DefaultConfig
from models.egret_attention_visual import egret_ppi
import data_generator_attention_visual as data_generator
from data_generator_attention_visual import graph_collate
from feature_generator import ProtBERT_feature_generator, distance_and_angle_generator

configs = DefaultConfig()
THREADHOLD = 0.2

def test(model, loader, root_dir):
    global configs
    model.eval()
    result = []
    attention_scores = []
    edges = []
    protein_infos_ = []

    for batch_idx, (protbert_data, graph_batch, protein_info) in enumerate(loader):
        with torch.no_grad():
            if torch.cuda.is_available():
                protbert_var = torch.autograd.Variable(protbert_data.cuda().float())
                graph_batch = graph_batch.to('cuda')
                graph_batch.edata['ex'] = torch.autograd.Variable(graph_batch.edata['ex'].cuda().float())
            else:
                protbert_var = torch.autograd.Variable(protbert_data.float())
                graph_batch.edata['ex'] = torch.autograd.Variable(graph_batch.edata['ex'].float())

            # compute output
            # t0 = time.time()
            output, head_attn_scores = model(protbert_var, graph_batch)
            # print(output.__len__(), output.shape)
            shapes = output.data.shape
            output = output.view(shapes[0], configs.max_sequence_length)
            output = output.data.cpu().numpy()
            head_attn_scores = head_attn_scores[0].view(shapes[0], configs.max_sequence_length, 20).numpy()
            graph_list = dgl.unbatch(graph_batch)
            for i, graph in enumerate(graph_list):
                __len_limit__ = min(configs.max_sequence_length, protein_info[i]['seq_length'])
                protein_infos_.append(protein_info[i])
                result.append(output[i][:__len_limit__])
                attention_scores.append(head_attn_scores[i, :__len_limit__, :])
                edges.append(graph.edges()[0].view(__len_limit__, 20).numpy())

    predict_result = {}

    predict_result["pred"] = result
    predict_result["protein_info"] = protein_infos_
    predict_result["edges"] = edges
    predict_result["attention_scores"] = attention_scores
    result_file = "{}/outputs/prediction_and_attention_scores.pkl".format(root_dir)
    with open(result_file,"wb") as fp:
        pickle.dump(predict_result,fp)

def predict(model_file, root_dir):
    # test_protBERT_file = [root_dir+'/inputs/ProtBert_features.pkl.gz']
    protein_list_file = root_dir+'/inputs/protein_list.txt'
    test_dataSet = data_generator.dataSet(root_dir, protein_list_file)
    # print(protein_list.__len__(), protein_list)
    test_loader = torch.utils.data.DataLoader(test_dataSet,
                                              batch_size=configs.batch_size,
                                              shuffle=False,
                                              pin_memory=(torch.cuda.is_available()),
                                              num_workers=configs.num_workers, drop_last=False, collate_fn=graph_collate)

    model = egret_ppi()
    pretrained_dict = torch.load(model_file)
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()
    test(model, test_loader, root_dir)


if __name__ == '__main__':
    root_dir = '.'
    # generate_distance_and_angle_matrix_and_fasta
    distance_and_angle_generator.generate_distance_and_angle_matrix(root_dir)
    # generate_protbert_features
    ProtBERT_feature_generator.generate_protbert_features(root_dir)
    t111 = time()

    model_dir = "{}/models/egret_model_weight.dat".format(root_dir)
    predict(model_dir, root_dir)
    print('Prediction completed. Results saved at:', "{}/outputs/prediction_and_attention_scores.pkl".format(root_dir))

    t222 = time()
    print('\nOnly Feature Generation Time:', t111-t000)
    print('Only Inference Time:', t222-t111)
    print('Total Time:', t222-t000)

