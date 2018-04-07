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
from sklearn.feature_selection import f_regression
import random


def eff_vec_finder(indel_count_matrix,name_genes_grna_unique):
    num_indel,num_site = np.shape(indel_count_matrix)
    dict_eff = {}
    for filename in glob.glob('/Users/amirali/Projects/muteff/*.txt'):
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

def longest_substring_passing_cutsite(strng,character):
    len_substring=0
    longest=0
    label_set = []
    midpoint = len(strng)/2
    for i in range(len(strng)):
        if i > 1:
            if strng[i] != strng[i-1] or strng[i] != character:
                len_substring = 0
                label_set = []
        if strng[i] == character:
            label_set.append(i)
            len_substring += 1
        if len_substring > longest and (midpoint-1 in label_set or 3 in label_set):
            longest = len_substring

    return longest


print "loading name_genes_grna_unique ..."
name_genes_grna_unique = pickle.load(open('storage/name_genes_grna_unique_one_patient_per_site.p', 'rb'))
print "loading name_indel_type_unique ..."
name_indel_type_unique = pickle.load(open('storage/name_indel_type_unique.p', 'rb'))
print "loading indel_count_matrix ..."
indel_count_matrix = pickle.load(open('storage/indel_count_matrix_one_patient_per_site.p', 'rb'))
print "loading indel_prop_matrix ..."
indel_prop_matrix = pickle.load(open('storage/indel_prop_matrix_one_patient_per_site.p', 'rb'))
print "loading length_indel ..."
length_indel_insertion = pickle.load(open('storage/length_indel_insertion.p', 'rb'))
length_indel_deletion = pickle.load(open('storage/length_indel_deletion.p', 'rb'))


intron_exon = pickle.load(open('storage/intron_exon_status.pkl', 'rb'))


location_dict = {}
with open('sequence_pam_gene_grna_big_file_donor_genomic_context.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    row_counter = 0
    for row in spamreader:
        location_dict[row[0].split(',')[0]]=row[0].split(',')[4]

intron_exon_label_vec = []
site_count = 0
for site_name in name_genes_grna_unique:
    site_name_list = site_name.split('-')
    location = location_dict[site_name_list[1] + '-' + site_name_list[2]]
    print location
    print intron_exon[location]
    intron_exon_label_vec.append(int(intron_exon[location][16]))

intron_exon_label_vec = np.asarray(intron_exon_label_vec)
print intron_exon_label_vec
print 'number of nongens', np.sum(intron_exon_label_vec==0)
print 'number of Introns', np.sum(intron_exon_label_vec==1)
print 'number of Exons', np.sum(intron_exon_label_vec==2)


# X = np.random.random((10,10))
# beta = np.random.random(10)
# y = np.dot(X,beta)
# print f_regression(X,y)[1]

#print longest_substring_passing_cutsite('AATT','T')

# def longest_substring(strng,character):
#     len_substring=0
#     longest=0
#     for i in range(len(strng)):
#         if i > 1:
#             if strng[i] != strng[i-1] or strng[i] != character:
#                 len_substring = 0
#         if strng[i] == character:
#             len_substring += 1
#         if len_substring > longest:
#             longest = len_substring
#     return longest

# print longest_substring('CGAAGTAAATTTTAAAAAC','F')

# G=nx.Graph()
#
# G.add_edge(1,2,w=0.5, c = 'b')
# G.add_edge(1,3,w=9.8, c = 'g')
# edges = G.edges()
# colors = [G[u][v]['c'] for u, v in edges]
# weights = [G[u][v]['w'] for u, v in edges]
# nx.draw(G, nx.circular_layout(G), font_size=10, node_color='y', with_labels=True, edge_color=colors)
#
# nodes = G.nodes()
# print nodes
# # pos = nx.get_node_attributes(G, 'pos')
# # labels = nx.get_edge_attributes(G, 'weight')
# # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
#
# plt.savefig("test.pdf")
# plt.clf()

# G=nx.Graph()
# i=1
# G.add_node(i,pos=(i,i))
# G.add_node(2,pos=(2,2))
# G.add_node(3,pos=(1,0))
# G.add_edge(1,2,weight=0.5)
# G.add_edge(1,3,weight=9.8)
# pos=nx.get_node_attributes(G,'pos')
# nx.draw(G,pos)
# labels = nx.get_edge_attributes(G,'weight')
# print pos
# print labels
# nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
# plt.savefig("test.pdf")