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
from pandas import read_excel
import pandas as pd


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
dic_del_len = np.zeros(indel_num)
min_start=0
max_stop=0

size_dic_essential = {}
size_dic_not_essential = {}
size_vec = []
for indel_index in range(indel_num):
    dic_del[counter]=[]
    dic_del_start[counter] = []
    dic_del_stop[counter] = []
    indel_locations = re.split('I|D',name_indel_type_unique[indel_index])[:-1]
    indel_types = ''.join(c for c in name_indel_type_unique[indel_index] if (c=='I' or c=='D'))
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

            size_dic_essential[size] = 0
            size_dic_not_essential[size] = 0
            size_vec.append(size)

    counter = counter + 1


#########

gene_list = []
for site, site_name in enumerate(name_genes_grna_unique):
    site_name_list = site_name.split('-')
    gene_list.append(site_name_list[0])
gene_set = set(gene_list)
print "number of genes", len(gene_set)
print "number of sites", site

## James Nature Paper
# essential_gene_chart = read_excel('/Users/amirali/Projects/nature19057-SI Table 13.xlsx','LoF Intolerant',header=None)
# essential_gene_list =  essential_gene_chart[1].values.tolist()[1:]
data = pd.read_csv('/Users/amirali/Projects/tables4.txt', sep=" ", header=None)
data.columns = ["gene1", "v1", "gene2", "v2"]
essential_gene_list= list(data['gene1'])

essential_counter = 0
essential_gene_dic = {}
for gene in gene_set:
    if gene in essential_gene_list:
        essential_gene_dic[gene] = 1
        essential_counter+=1
    else:
        essential_gene_dic[gene] = 0

print "number of essential genes", essential_counter

for site_index,site_name in enumerate(name_genes_grna_unique):
    gene_name = site_name.split('-')[0]
    for indel_index in range(indel_num):
        list_start = dic_del_start[indel_index]
        list_stop = dic_del_stop[indel_index]
        for i in range(len(list_start)):
            if essential_gene_dic[gene_name] == 1: # essential genes
                size_dic_essential[list_stop[i] - list_start[i]] += indel_fraction_mutant_matrix[indel_index, site_index]
            else:
                size_dic_not_essential[list_stop[i] - list_start[i]] += indel_fraction_mutant_matrix[indel_index, site_index]


size_vec_unique = np.sort(list(set(size_vec)))
size_freq_essential = []
size_freq_notessential = []
for i in range(np.size( size_vec_unique  )):
    size_freq_essential.append(size_dic_essential[size_vec_unique[i]])
    size_freq_notessential.append(size_dic_not_essential[size_vec_unique[i]])

plt.plot(size_vec_unique,size_freq_essential,'o')
plt.ylabel('Sum of Fractions')
plt.xlabel('Length')
#plt.legend(['Empirical Distribution', 'Random Control'], loc=3)
plt.savefig('plots/deletion_length_hist_essential.pdf')
plt.clf()

plt.plot(size_vec_unique,size_freq_notessential,'o')
plt.ylabel('Sum of Fractions')
plt.xlabel('Length')
#plt.legend(['Empirical Distribution', 'Random Control'], loc=3)
plt.savefig('plots/deletion_length_hist_notessential.pdf')
plt.clf()
