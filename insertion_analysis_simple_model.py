from preprocess_indel_files import preprocess_indel_files
from compute_summary_statistic import compute_summary_statistics
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
import csv
import networkx as nx
from sequence_logos import plot_seq_logo
from sklearn.metrics import jaccard_similarity_score
import collections
from operator import itemgetter
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
import scipy.stats as stats
import math
from scipy.stats import ttest_ind_from_stats
from sequence_logos import plot_seq_logo
from sequence_logos import plot_QQ
from sklearn.metrics import mean_squared_error
from math import sqrt
import glob
from huffman import HuffmanCoding
from sklearn.feature_selection import f_regression
import random


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

indel_fraction_mutant_matrix = indel_count_matrix / np.reshape(np.sum(indel_count_matrix, axis=0), (1, -1))

# ########### Creat all Genomic context file
context_genome_dict = {}
simple_context_genome_dict = {}
with open('sequence_pam_gene_grna_big_file_donor_genomic_context.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    row_counter = 0
    for row in spamreader:
        context = row[0].split(',')[6]
        context = context.replace('a','A')
        context = context.replace('t','T')
        context = context.replace('c','C')
        context = context.replace('g','G')
        context_genome_dict[row[0].split(',')[0]] = context

site = 0
for site_name in name_genes_grna_unique:
    site_name_list = site_name.split('-')
    context = context_genome_dict[site_name_list[1] + '-' + site_name_list[2]]
    simple_context_genome_dict[site] = context
    site+=1


gene_file_dict = {}
list_all_genes = []
ins_table_folder = '/Users/amirali/Projects/ins_data/'
for each_file in glob.glob(ins_table_folder + "ins_sites-*.txt"):
    each_file_splited = each_file.split('ins_sites-')[1]
    if each_file_splited.split('-')[0] not in list_all_genes:
        gene_file_dict[each_file_splited.split('-')[0]] = each_file
        list_all_genes.append(each_file_splited.split('-')[0])

ins_dic = {}
site_index = 0
counter = 0
for site_name in name_genes_grna_unique:
    indel_index = 0
    for indel_type in name_indel_type_unique:
        if 'I' in indel_type and indel_count_matrix[indel_index,site_index]>0:
            #print indel_type
            file_name = gene_file_dict[site_name.split('-')[0]]
            file = open(file_name, 'r')
            for line in file:
                if '"'+indel_type+'"' in line:
                    #print line
                    counter += 1
                    ins_dic[indel_index,site_index] = line.split(',')[2].strip('"')
                    break

        indel_index += 1
    site_index += 1


boundary_dic_A = {}
boundary_dic_T = {}
boundary_dic_C = {}
boundary_dic_G = {}
boundary_dic_control = {}
for nuc1 in ['A', 'C', 'G', 'T']:
    for nuc2 in ['A', 'C', 'G', 'T']:
        boundary_dic_A[nuc1+nuc2] = 0.
        boundary_dic_T[nuc1 + nuc2] = 0.
        boundary_dic_C[nuc1 + nuc2] = 0.
        boundary_dic_G[nuc1 + nuc2] = 0.

max_repeat = 10
for repeat in range(10):
    for nuc1 in ['A', 'C', 'G', 'T']:
        for nuc2 in ['A', 'C', 'G', 'T']:
            boundary_dic_control[repeat , nuc1 + nuc2] = 0.



nAA = 0
nAT = 0
nAC = 0
nAG = 0

nTA = 0
nTT = 0
nTC = 0
nTG = 0

nCA = 0
nCT = 0
nCC = 0
nCG = 0

nGA = 0
nGT = 0
nGC = 0
nGG = 0

nuc_dic = {}
nuc_dic['A'] = 0
nuc_dic['T'] = 1
nuc_dic['C'] = 2
nuc_dic['G'] = 3
nXXL = np.zeros((4,4))
nXXR = np.zeros((4,4))


print ins_dic
for key, indel_seq in ins_dic.iteritems():
    indel_index = key[0]
    site_index = key[1]
    insertion_site = int(name_indel_type_unique[indel_index].split(':')[0])-1
    context = simple_context_genome_dict[site_index]
    nuc1 = context[49+insertion_site]
    nuc2 = context[50+insertion_site]
    if len(indel_seq) == 1:
        nXXL[int(nuc_dic[indel_seq]),int(nuc_dic[nuc1])] += indel_fraction_mutant_matrix[indel_index,site_index]
        nXXR[int(nuc_dic[indel_seq]), int(nuc_dic[nuc2])] += indel_fraction_mutant_matrix[indel_index, site_index]

#print nXXR
#print nXXL

nXX = nXXR
PP = np.zeros((4,4))

for i in range(4):
    PM = np.ones((3,3))
    PM[0,0] +=  nXX[i,3] / nXX[i,0]
    PM[1,1] +=  nXX[i,3] / nXX[i,1]
    PM[2,2] +=  nXX[i,3] / nXX[i,2]

    row = np.linalg.solve(PM, np.ones(3))

    for j in range(3):
        PP[i,j] = row[j]

    PP[i,3] = 1 - np.sum(row)

print PP