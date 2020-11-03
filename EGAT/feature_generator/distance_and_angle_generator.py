# coding: utf-8

# In[1]:

import numpy as np
from time import time
import gzip
import warnings
import pickle

warnings.filterwarnings("ignore")

from Bio.PDB import *
import Bio
from config import DefaultConfig
configs = DefaultConfig()



parser = PDBParser()

THIRD_ATOM = 'N'  # 'O'


# In[2]:


def residue_distance(res1, res2):
    distance = []
    cnt = 0
    for atom1 in res1:
        for atom2 in res2:
            distance += [abs(atom1 - atom2)]
            cnt += 1
    distance = np.array(distance)
    dist_mean = distance.mean()
    dist_std = distance.std()
    if 'CA' in res1 and 'CA' in res2:
        dist_ca = abs(res1['CA'] - res2['CA'])
    else:
        dist_ca = dist_mean
    return dist_mean, dist_std, dist_ca


def residue_relative_angle(res1, res2):
    if 'CA' in res1 and THIRD_ATOM in res1 and 'C' in res1:
        v1 = res1['CA'].get_vector().get_array()
        v2 = res1[THIRD_ATOM].get_vector().get_array()
        v3 = res1['C'].get_vector().get_array()
        normal1 = np.cross((v2 - v1), (v3 - v1))
    else:
        k = list(res1)
        if len(k) == 1:
            normal1 = k[0].get_vector().get_array()
        else:
            raise
    normal1 = normal1 / np.linalg.norm(normal1)

    if 'CA' in res2 and THIRD_ATOM in res2 and 'C' in res2:
        v1 = res2['CA'].get_vector().get_array()
        v2 = res2[THIRD_ATOM].get_vector().get_array()
        v3 = res2['C'].get_vector().get_array()
        normal2 = np.cross((v2 - v1), (v3 - v1))
    else:
        k = list(res2)
        if len(k) == 1:
            normal2 = k[0].get_vector().get_array()
        else:
            raise
    normal2 = normal2 / np.linalg.norm(normal2)

    return np.arccos(np.clip(np.dot(normal1, normal2), -1.0, 1.0))


def get_dist_and_angle_matrix(residues):
    size = len(residues)
    dist_mat = np.zeros([size, size, 3])
    angle_mat = np.zeros([size, size])
    for i in range(size):
        for j in range(i + 1, size):
            dist_mean, dist_std, dist_ca = residue_distance(residues[i], residues[j])
            angle = residue_relative_angle(residues[i], residues[j])

            dist_mat[i, j, 0] = dist_mean
            dist_mat[i, j, 1] = dist_std
            dist_mat[i, j, 2] = dist_ca

            dist_mat[j, i, 0] = dist_mean
            dist_mat[j, i, 1] = dist_std
            dist_mat[j, i, 2] = dist_ca

            angle_mat[i, j] = angle
            angle_mat[j, i] = angle

    return dist_mat, angle_mat


# In[3]:


from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from Bio import SeqUtils

protein_letters_3to1 = SeqUtils.IUPACData.protein_letters_3to1

ppb = PPBuilder()


def generate_distance_and_angle_matrix(root_dir):
    t0 = time()
    dist_matrix_map = {}  # key: idx, value: {'protein_id':<complexName>_<chains>, 'dist_matrix':dist_matrix}
    angle_matrix_map = {}  # key: idx, value: {'protein_id':<complexName>_<chains>, 'angle_matrix':angle_matrix}

    with open(root_dir + '/inputs/protein_list.txt', 'r') as f:
        protein_list = f.readlines()
        protein_list = [x.strip() for x in protein_list]

    for idx, protein_name in enumerate(protein_list):

        complex_name, chains = protein_name.upper().split('_')
        try:
            path = root_dir + '/inputs/pdb_files/' + protein_name + '.pdb'
            structure = parser.get_structure(complex_name, path)
        except:
            path = root_dir + '/inputs/pdb_files/' + protein_name.lower() + '.pdb'
            structure = parser.get_structure(complex_name, path)

        model = structure[0]  # every structure object has only one model object in DeepPPISP's dataset
        pep_pdb = ''
        residues = []
        for chain_id in chains:
            for residue in model[chain_id].get_residues():
                residues += [residue]
            peptides = ppb.build_peptides(model[chain_id])
            pep_pdb += ''.join([str(pep.get_sequence()) for pep in peptides])

        pep_seq_from_res_list = ''
        i = 0
        total_res = 0
        temp = 0
        residues2 = []
        original_total_res = len(residues)
        while i < original_total_res:
            res = residues[i]
            res_name = res.get_resname()
            if res_name[0] + res_name[1:].lower() not in protein_letters_3to1:
                temp += 1
            else:
                pep_seq_from_res_list += protein_letters_3to1[res_name[0] + res_name[1:].lower()]
                residues2 += [residues[i]]
                total_res += 1
                if total_res == configs.max_sequence_length:
                    break
            i += 1

        if pep_pdb[:configs.max_sequence_length] != pep_seq_from_res_list and False: # for debug only
            print('Extraction of residue-information from PDB file filed for protein:', protein_name)
            # print('Pairwise Alignment:')
            alignments = pairwise2.align.globalxx(pep_pdb[:configs.max_sequence_length], pep_seq_from_res_list)
            print(format_alignment(*alignments[0]))
            print('len(residues2):', len(residues2),
                  ', len(pep_pdb):', len(pep_pdb),
                  ', len(pep_seq_from_res_list):', len(pep_seq_from_res_list),
                  '\n')
            # print(format_alignment(*alignments[0]))
            # print('...')
            raise Exception
        else:
            t1 = time()
            print(idx, ':', protein_name, len(pep_seq_from_res_list))
            fasta_string = '>{}\n'.format(protein_name) + pep_seq_from_res_list

            with open(root_dir + '/inputs/fasta_files/{}.fasta'.format(protein_name), 'w') as f:
                f.write(fasta_string)

            dist_mat, angle_mat = get_dist_and_angle_matrix(residues2[:configs.max_sequence_length])
            dist_matrix_map[idx] = {'protein_id': protein_name, 'dist_matrix': dist_mat}
            angle_matrix_map[idx] = {'protein_id': protein_name, 'angle_matrix': angle_mat}
            print(idx, 'done.', time()-t1)

    pickle.dump(dist_matrix_map, gzip.open(root_dir + '/inputs/ppisp_dist_matrix_map.pkl.gz', 'wb'))
    pickle.dump(angle_matrix_map, gzip.open(root_dir + '/inputs/ppisp_angle_matrix_map.pkl.gz', 'wb'))
    print('Total time for Distance and Angle matrices generation:', time() - t0)

#     dist_matrix_map, angle_matrix_map

