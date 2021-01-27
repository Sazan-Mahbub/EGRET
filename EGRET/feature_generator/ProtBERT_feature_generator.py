from time import time
    
import torch
from transformers import BertModel, BertTokenizer
import re
import os
import requests
from tqdm.auto import tqdm
import numpy as np
import gzip
import pickle

def generate_protbert_features(root_dir):
    t0=time()
    modelUrl = 'https://www.dropbox.com/s/dm3m1o0tsv9terq/pytorch_model.bin?dl=1'
    configUrl = 'https://www.dropbox.com/s/d3yw7v4tvi5f4sk/bert_config.json?dl=1'
    vocabUrl = 'https://www.dropbox.com/s/jvrleji50ql5m5i/vocab.txt?dl=1'

    downloadFolderPath = root_dir+'/inputs/ProtBert_model/'

    modelFolderPath = downloadFolderPath

    modelFilePath = os.path.join(modelFolderPath, 'pytorch_model.bin')

    configFilePath = os.path.join(modelFolderPath, 'config.json')

    vocabFilePath = os.path.join(modelFolderPath, 'vocab.txt')

    if not os.path.exists(modelFolderPath):
        os.makedirs(modelFolderPath)
        
    def download_file(url, filename):
      response = requests.get(url, stream=True)
      with tqdm.wrapattr(open(filename, "wb"), "write", miniters=1,
                        total=int(response.headers.get('content-length', 0)),
                        desc=filename) as fout:
          for chunk in response.iter_content(chunk_size=4096):
              fout.write(chunk)
                

    if not os.path.exists(modelFilePath):
        download_file(modelUrl, modelFilePath)

    if not os.path.exists(configFilePath):
        download_file(configUrl, configFilePath)

    if not os.path.exists(vocabFilePath):
        download_file(vocabUrl, vocabFilePath)
        
    tokenizer = BertTokenizer(vocabFilePath, do_lower_case=False )
    model = BertModel.from_pretrained(modelFolderPath)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model = model.eval()

    def make_aseq(seq):
        protAlphabet = 'ACDEFGHIKLMNPQRSTVWYX'
        return ' '.join([protAlphabet[x] for x in seq])


    # data = ['MSREEVESLIQEVLEVYPEKARKDRNKHLAVNDPAVTQSKKCIISNKKSQPGLMTIRGCAYAGSKGVVWGPIKDMIHISHGPVGCGQYSRAGRRNYYIGTTGVNAFVTMNFTSDFQEKDIVFGGDKKLAKLIDEVETLFPLNKGISVQSECPIGLIGDDIESVSKVKGAELSKTIVPVRCEGFRGVSQSLGHHIANDAVRDWVLGKRDEDTTFASTPYDVAIIGDYNIGGDAWSSRILLEEMGLRCVAQWSGDGSISEIELTPKVKLNLVHCYRSMNYISRHMEEKYGIPWMEYNFFGPTKTIESLRAIAAKFDESIQKKCEEVIAKYKPEWEAVVAKYRPRLEGKRVMLYIGGLRPRHVIGAYEDLGMEVVGTGYEFAHNDDYDRTMKEMGDSTLLYDDVTGYEFEEFVKRIKPDLIGSGIKEKFIFQKMGIPFREMHSWDYSGPYHGFDGFAIFARDMDMTLNNPCWKKLQAPWEASQQVDKIKASYPLFLDQDYKDM',
    #         'HLQSTPQNLVSNAPIAETAGIAEPPDDDLQARLNTLKKQ']

    sequences = []

    with open(root_dir+'/inputs/protein_list.txt', 'r') as f:
        protein_list = f.readlines()
        for protein in protein_list:
            seq = open(root_dir+'/inputs/fasta_files/{}.fasta'.format(protein.strip()), 'r').readlines()
            sequences += [seq[1].strip()]
        
    sequences_Example = [' '.join(list(seq)) for seq in sequences]
    sequences_Example = [re.sub(r"[-UZOB]", "X", sequence) for sequence in sequences_Example]


    all_protein_features = []

    for i, seq in enumerate(sequences_Example):
        ids = tokenizer.batch_encode_plus([seq], add_special_tokens=True, pad_to_max_length=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids,attention_mask=attention_mask)[0]
        embedding = embedding.cpu().numpy()
        features = [] 
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][1:seq_len-1]
            features.append(seq_emd)
    #     print(features.__len__())
    #     print(features[0].shape)
        # print(all_protein_sequences['all_protein_complex_pdb_ids'][i])
    #     print(features)
        all_protein_features += features
        
    pickle.dump({'ProtBert_features':all_protein_features}, 
                gzip.open(root_dir+'/inputs/ProtBert_features.pkl.gz',
                          'wb')
                )

    print('Total time spent for ProtBERT:',time()-t0)
