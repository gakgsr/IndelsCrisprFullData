#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 13:12:09 2018

@author: zqwu
"""

import numpy as np
from sklearn import linear_model, metrics
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import time
import pickle
import networkx as nx
from sequence_logos import plot_seq_logo
from preprocess_indel_files import preprocess_indel_files
from compute_summary_statistic import compute_summary_statistics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from simple_linear_model import fraction_of_deletion_insertion, \
  expected_deletion_insertion_length, coding_region_finder, flatness_finder, \
  length_of_repeat_finder
import glob
import copy

def one_hot_index(nucleotide):
  nucleotide_array = ['A', 'C', 'G', 'T']
  return nucleotide_array.index(nucleotide)

def eff_vec_finder(indel_count_matrix,name_genes_grna_unique):
    num_indel,num_site = np.shape(indel_count_matrix)
    dict_eff = {}
    # Redefined 
    for filename in glob.glob('Other_Data/muteff/*.txt'):
        file = open(filename)
        for line in file:
            if 'RL384' in line:
                line = line.replace('_','-')
                line = line.replace('"', '')
                if 'N' not in line.split(',')[1]:
                    eff = float(line.split(',')[1])
                    line_list = (line.split(',')[0]).split('-')
                    dict_eff[line_list[1]+'-'+line_list[2]] = eff


    eff_vec = np.zeros(num_site)
    site = 0
    for site_name in name_genes_grna_unique:
        site_name_list = site_name.split('-')
        eff_vec[site] = dict_eff[site_name_list[1] + '-' + site_name_list[2]]
        site += 1

    return eff_vec

def load_gene_sequence(sequence_file_name, 
                       name_genes_grna_unique,
                       homopolymer_matrix,
                       chrom_mat):
  """ Michael: I redefined this function for simplicity """
  # Create numpy matrix of size len(name_genes_grna_unique) * 23, to store the sequence as one-hot encoded
  sequence_pam_per_gene_grna = np.zeros((len(name_genes_grna_unique), 23, 4), dtype = bool)
  homop_per_gene_grna = np.zeros((len(name_genes_grna_unique), 4))
  chromatin_per_gene_grna = np.zeros((len(name_genes_grna_unique), 33))

  chromatin_filled_values = np.nanmedian(np.concatenate(chrom_mat.values(), axis=0), axis=0)
  # Obtain the grna and PAM sequence corresponding to name_genes_grna_unique
  with open(sequence_file_name) as f:
    for line in f:
      line = line.replace('"', '')
      line = line.replace(' ', '')
      line = line.replace('\n', '')
      l = line.split(',')
      if l[1] + '-' + l[0] in name_genes_grna_unique:
        index_in_name_genes_grna_unique = name_genes_grna_unique.index(l[1] + '-' + l[0])
        for i in range(20):
          sequence_pam_per_gene_grna[index_in_name_genes_grna_unique, i, one_hot_index(l[2][i])] = 1
        for i in range(3):
          sequence_pam_per_gene_grna[index_in_name_genes_grna_unique, 20 + i, one_hot_index(l[3][i])] = 1

        
        
        chromatin_values = np.nanmean(chrom_mat[l[1] + '-' + l[0]],axis=0)
        nan_exceptions = np.where(chromatin_values != chromatin_values)
        if len(nan_exceptions) > 0:
          for nan_pos in nan_exceptions[0]:
            chromatin_values[nan_pos] = chromatin_filled_values[nan_pos]
        chromatin_per_gene_grna[index_in_name_genes_grna_unique, :] = chromatin_values

        homop_per_gene_grna[index_in_name_genes_grna_unique, :] = homopolymer_matrix[:,index_in_name_genes_grna_unique]

  #plot_seq_logo(np.mean(sequence_pam_per_gene_grna, axis=0), "input_spacer")
  # Scikit needs only a 2-d matrix as input, so reshape and return
  return chromatin_per_gene_grna,\
         homop_per_gene_grna,\
         sequence_pam_per_gene_grna

def Kfold_Accuracy(features, labels, model_ins, K=3):
  n_samples = len(labels)
  total_scores = []
  groups = np.linspace(0, n_samples, K+1).astype(int)
  ids = np.arange(n_samples)
  for repeat in range(10):
    print("On repeat %d" % repeat)
    np.random.shuffle(ids)
    scores = []
    for i in range(K):
      valid_inds = ids[groups[i]:groups[i+1]]
      train_inds = np.delete(ids, valid_inds)
      
      train_X = features[train_inds, :]
      train_y = labels[train_inds]
      valid_X = features[valid_inds, :]
      valid_y = labels[valid_inds]

      model = copy.deepcopy(model_ins)
      model.fit(train_X, train_y)
      valid_y_pred = model.predict(valid_X)
      scores.append(r2_score(valid_y, valid_y_pred))
    print(np.mean(scores, axis=0))
    total_scores.append(np.mean(scores, axis=0))
  print("Mean: %f, STD: %f" % (np.mean(total_scores, axis=0), np.std(total_scores, axis=0)))
  return np.mean(total_scores, axis=0), np.std(total_scores, axis=0)

if __name__ == "__main__":
  sequence_file_name = "sequence_pam_gene_grna_big_file_donor.csv"
  
  
  print "loading name_genes_grna_unique ..."
  name_genes_grna_unique = pickle.load(open('storage/name_genes_grna_unique_one_patient_per_site.p', 'rb'))
  #name_genes_grna_unique = pickle.load(open('storage/name_genes_grna_unique.p', 'rb'))
  print "loading name_indel_type_unique ..."
  name_indel_type_unique = pickle.load(open('storage/name_indel_type_unique.p', 'rb'))
  print "loading indel_count_matrix ..."
  indel_count_matrix = pickle.load(open('storage/indel_count_matrix_one_patient_per_site.p', 'rb'))
  #indel_count_matrix = pickle.load(open('storage/indel_count_matrix.p', 'rb'))
  print "loading indel_prop_matrix ..."
  indel_prop_matrix = pickle.load(open('storage/indel_prop_matrix_one_patient_per_site.p', 'rb'))
  #indel_prop_matrix = pickle.load(open('storage/indel_prop_matrix.p', 'rb'))
  print "loading length_indel ..."
  length_indel_insertion = pickle.load(open('storage/length_indel_insertion.p', 'rb'))
  length_indel_deletion = pickle.load(open('storage/length_indel_deletion.p', 'rb'))
  print "loading homopolymer matrix"
  homopolymer_matrix = pickle.load(open('storage/homopolymer_matrix_w-3:3.p', 'rb'))
  
  chrom_mat = pickle.load(open('storage/chrom_label_dic.p', 'rb'))
  chrom_mat_name = pickle.load(open('storage/chrom_label_dic_name.p', 'rb'))
  
  # BUILD LABELS
  fraction_insertions, fraction_deletions = fraction_of_deletion_insertion(indel_count_matrix,length_indel_insertion,length_indel_deletion)
  exp_insertion_length, exp_deletion_length = expected_deletion_insertion_length(indel_count_matrix,length_indel_insertion,length_indel_deletion)
  eff_vec = eff_vec_finder(indel_count_matrix,name_genes_grna_unique)

  # BUILD FEATURES
  chromatin_f, homop_f, sequence_pam_f = load_gene_sequence(sequence_file_name, 
                                                            name_genes_grna_unique,
                                                            homopolymer_matrix,
                                                            chrom_mat)
  sequence_pam_f = sequence_pam_f.reshape((1674, 23*4))

  feat = np.concatenate([sequence_pam_f, chromatin_f, homop_f], axis=1)
  feature_names = [str(i) + str(j) for i in range(23) for j in ['A', 'C', 'G', 'T']] + \
                  chrom_mat_name + ['homo' + str(j) for j in ['A', 'C', 'G', 'T']]
  feature_names = np.array(feature_names)
  
  
  # DEFINE MODEL
  #model_ins = XGBRegressor(n_estimators=500)
  model_ins = RandomForestRegressor(n_estimators=1000)
  
  # RUN MODEL
  Kfold_Accuracy(feat, eff_vec, model_ins)