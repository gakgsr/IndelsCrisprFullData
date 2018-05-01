from preprocess_indel_files import preprocess_indel_files
from compute_summary_statistic import compute_summary_statistics
import numpy as np
from sklearn import linear_model, metrics
from sklearn.model_selection import KFold
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import time
import pickle
import re
import csv
from scipy.stats import geom
from scipy.stats import expon
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
from scipy.stats import entropy
from scipy.signal import savgol_filter
import pandas as pd
from pandas import read_excel
from scipy.stats import ttest_ind_from_stats

def distance_to_transcribed_sequence(name_genes_grna_unique):
    minDistTSS = []
    minDistTSE = []
    disT = pickle.load(open('/Users/amirali/Projects/disTSS_annotations.pkl', 'rb'))
    location_dict = {}
    with open('sequence_pam_gene_grna_big_file_donor_genomic_context.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            location_dict[row[0].split(',')[0]] = row[0].split(',')[4]

    for site_name in name_genes_grna_unique:
        site_name_list = site_name.split('-')
        location = location_dict[site_name_list[1] + '-' + site_name_list[2]]
        minDistTSS.append(np.mean(disT[location[1:]][:,0]))
        minDistTSE.append(np.mean(disT[location[1:]][:,1]))

    return minDistTSS,minDistTSE

def coding_region_finder(name_genes_grna_unique):
    intron_exon_dict = pickle.load(open('storage/intron_exon_status.pkl', 'rb'))
    location_dict = {}
    with open('sequence_pam_gene_grna_big_file_donor_genomic_context.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        row_counter = 0
        for row in spamreader:
            location_dict[row[0].split(',')[0]]=row[0].split(',')[4]

    intron_exon_label_vec = []

    for site_name in name_genes_grna_unique:
        site_name_list = site_name.split('-')
        location = location_dict[site_name_list[1] + '-' + site_name_list[2]]

        intron_exon_label_vec.append(int(round( np.mean(intron_exon_dict[location][16])  )))

    intron_exon_label_vec = np.asarray(intron_exon_label_vec)
    return intron_exon_label_vec

data_folder = "../IndelsFullData/"
sequence_file_name = "sequence_pam_gene_grna_big_file_donor.csv"
data_folder = "/Users/amirali/Projects/CRISPR-data-Feb18/20nt_counts_only/"

print "loading files"
name_genes_grna_unique = pickle.load(open('storage/name_genes_grna_unique_one_patient_per_site.p', 'rb'))
name_indel_type_unique = pickle.load(open('storage/name_indel_type_unique.p', 'rb'))
indel_count_matrix = pickle.load(open('storage/indel_count_matrix_one_patient_per_site.p', 'rb'))
indel_prop_matrix = pickle.load(open('storage/indel_prop_matrix_one_patient_per_site.p', 'rb'))
length_indel_insertion = pickle.load(open('storage/length_indel_insertion.p', 'rb'))
length_indel_deletion = pickle.load(open('storage/length_indel_deletion.p', 'rb'))

indel_num,site_num =  np.shape(indel_count_matrix)
print 'number of sites = ', site_num
print 'number of indel types = ', indel_num

# here you can pick to work with counts or frction of mutant reads
indel_fraction_mutant_matrix = indel_count_matrix / np.reshape(np.sum(indel_count_matrix, axis=0), (1, -1))
indel_count_matrix_sum = np.sum(indel_fraction_mutant_matrix,axis=1)

deletion_list = []
counter =0
dic_del = {}
dic_del_start = {}
dic_del_stop = {}
dic_del_size_del = {}
dic_del_len = np.zeros(indel_num)
min_start=0
max_stop=0

size_dic = {}
size_vec = []
for indel_index in range(indel_num):
    dic_del[counter]=[]
    dic_del_start[counter] = []
    dic_del_stop[counter] = []
    dic_del_size_del[counter] = []
    indel_locations = re.split('I|D',name_indel_type_unique[indel_index])[:-1]
    indel_types = ''.join(c for c in name_indel_type_unique[indel_index] if (c=='I' or c=='D'))

    sum_of_deletion_size = 0
    for i in range(len(indel_types)):
        if indel_types[i]=='D':
            start, size = indel_locations[i].split(':')
            sum_of_deletion_size += int(size)

    if 'I' not in indel_types:
        dic_del_size_del[counter].append(sum_of_deletion_size)

    for i in range(len(indel_types)):
        if indel_types[i]=='D':
            start,size = indel_locations[i].split(':')
            start = int(start)
            size = int(size)
            if start > 0:
                start = start -1
            stop = start + size

            dic_del[counter].append(start)
            dic_del[counter].append(stop)
            dic_del_start[counter].append(start)
            dic_del_stop[counter].append(stop)
            dic_del_len[counter] += float(size)

            if start<min_start:
                min_start = start
            if int(stop)>max_stop:
                max_stop = stop

            size_dic[size] = 0
            size_vec.append(size)

    counter = counter + 1


#########
vec_3N = []
vec_3N_1 = []
vec_3N_2 = []
vec_3N_control = []
vec_3N_difference = []
size_vec_unique = np.sort(list(set(size_vec)))

for site_index in range(site_num):
    size_dic = dict.fromkeys(size_dic, 0)
    for indel_index in range(indel_num):
        if dic_del_size_del[indel_index]:
            size_dic[dic_del_size_del[indel_index][0]] += indel_fraction_mutant_matrix[indel_index, site_index]

    size_freq = []
    for i in range(np.size(size_vec_unique)):
        size_freq.append(size_dic[size_vec_unique[i]])

    len_dist = np.asarray(size_freq / sum(size_freq))
    len_dist_smooth = savgol_filter(len_dist, 55, 11)
    len_dist_smooth[len_dist_smooth < 0] = 0
    len_dist_smooth = len_dist_smooth / sum(len_dist_smooth)

    index_3N = np.linspace(2+3, 68+6, 24, dtype='int')
    index_3N_1 = np.linspace(1, 67, 23, dtype='int')
    index_3N_2 = np.linspace(0, 66, 23, dtype='int')

    len_dist_3N = len_dist[index_3N]
    len_dist_3N_1 = len_dist[index_3N_1]
    len_dist_3N_2 = len_dist[index_3N_2]

    vec_3N.append(np.sum(len_dist_3N))
    vec_3N_1.append(np.sum(len_dist_3N_1))
    vec_3N_2.append(np.sum(len_dist_3N_2))

    vec_3N_control.append(np.sum(len_dist_smooth[index_3N]))
    vec_3N_difference.append(np.sum(abs(len_dist_3N - len_dist_smooth[index_3N])))

vec_3N = np.asarray(vec_3N)
minDistTSS,minDistTSE = distance_to_transcribed_sequence(name_genes_grna_unique)

print np.mean(vec_3N[minDistTSE>np.mean(minDistTSE)])
print np.std(vec_3N[minDistTSE>np.mean(minDistTSE)])
print "---"
print np.mean(vec_3N[minDistTSE<np.mean(minDistTSE)])
print np.std(vec_3N[minDistTSE<np.mean(minDistTSE)])