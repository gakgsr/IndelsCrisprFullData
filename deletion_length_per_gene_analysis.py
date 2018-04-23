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

gene_list = []
for site, site_name in enumerate(name_genes_grna_unique):
    site_name_list = site_name.split('-')
    gene_list.append(site_name_list[0])
gene_set = set(gene_list)
print "number of genes", len(gene_set)
print "number of sites", site+1

## James Nature paper
#essential_gene_chart = read_excel('/Users/amirali/Projects/nature19057-SI Table 13.xlsx','LoF Intolerant',header=None)
#essential_gene_list =  essential_gene_chart[1].values.tolist()[1:]

## Science paper table S6
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

print 'number of essential genes =', essential_counter

vec_3N = []
vec_3N_1 = []
vec_3N_2 = []
vec_3N_control = []
vec_3N_difference = []
size_vec_unique = np.sort(list(set(size_vec)))
for gene in gene_set:
    size_dic = dict.fromkeys(size_dic, 0)
    #print gene
    site_index_list = np.where(np.asarray(gene_list) == gene)[0]
    for site_index in site_index_list:
        for indel_index in range(indel_num):
            #list_start = dic_del_start[indel_index]
            #list_stop = dic_del_stop[indel_index]
            #for i in range(len(list_start)):
            #   size_dic[list_stop[i] - list_start[i]] += indel_fraction_mutant_matrix[indel_index, site_index]
            if dic_del_size_del[indel_index]:
                size_dic[dic_del_size_del[indel_index][0]] += indel_fraction_mutant_matrix[indel_index, site_index]


    size_freq = []
    for i in range(np.size( size_vec_unique  )):
        size_freq.append(size_dic[size_vec_unique[i]])


    len_dist = np.asarray(size_freq/sum(size_freq))
    len_dist_smooth = savgol_filter(len_dist, 55, 11)
    len_dist_smooth[len_dist_smooth<0] = 0
    len_dist_smooth = len_dist_smooth / sum(len_dist_smooth)

    index_3N = np.linspace(2, 68, 23, dtype='int')
    index_3N_1 = np.linspace(1, 67, 23, dtype='int')
    index_3N_2 = np.linspace(0, 66, 23, dtype='int')

    len_dist_3N =  len_dist[index_3N]
    len_dist_3N_1 = len_dist[index_3N_1]
    len_dist_3N_2 = len_dist[index_3N_2]

    vec_3N.append(np.sum(len_dist_3N))
    vec_3N_1.append(np.sum(len_dist_3N_1))
    vec_3N_2.append(np.sum(len_dist_3N_2))

    vec_3N_control.append(np.sum( len_dist_smooth[index_3N]  ))
    vec_3N_difference.append( np.sum(abs(len_dist_3N - len_dist_smooth[index_3N] )))


    # plt.plot(size_vec_unique,len_dist , 'o')
    # plt.plot(size_vec_unique, len_dist_smooth, 'o')
    # plt.ylabel('Sum of Fractions')
    # plt.xlabel('Length')
    # plt.legend(['%s'%gene,"Smoothed Density"], loc=1)
    # plt.savefig('Length_per_gene_folder/%s.pdf'%gene)
    # #plt.savefig('Length_per_gene_folder/all_genes.pdf')
    # plt.clf()

plt.plot(np.sort(vec_3N),'o')
#plt.ylabel('Sum of Fractions')
#plt.xlabel('Length')
#plt.legend(['%s'%gene], loc=1)
#plt.savefig('Length_per_gene_folder/%s.pdf'%gene)
plt.savefig('Length_per_gene_folder/3N_lengths.pdf')
plt.clf()
#
#
plt.hist(vec_3N,20)
plt.hist(vec_3N_1,20)
plt.hist(vec_3N_2,20)
#plt.hist(vec_3N_control,10)
#plt.ylabel('Sum of Fractions')
plt.xlabel('Sum of 3N-x-Deletion Densities')
plt.legend(['3N','3N-1','3N-2'], loc=1)
#plt.savefig('Length_per_gene_folder/%s.pdf'%gene)
plt.savefig('Length_per_gene_folder/3N_lengths_hist2.pdf')
plt.clf()


essential_3N_vec = []
non_essential_3N_vec = []
for gene_counter,gene in enumerate(gene_set):
    if essential_gene_dic[gene] == 1:
        essential_3N_vec.append(vec_3N[gene_counter])
        #essential_3N_vec.append(vec_3N_difference[gene_counter])
    else:
        non_essential_3N_vec.append(vec_3N[gene_counter])
        #non_essential_3N_vec.append(vec_3N_difference[gene_counter])

print 'Essential genes'
print 'mean 3N = ', np.mean(essential_3N_vec)
print 'std 3N = ', np.std(essential_3N_vec)

print 'NonEssential genes'
print 'mean 3N = ', np.mean(non_essential_3N_vec)
print 'std 3N = ', np.std(non_essential_3N_vec)

tstat,pvalue = ttest_ind_from_stats(np.mean(essential_3N_vec), np.std(essential_3N_vec), len(essential_3N_vec),
                     np.mean(non_essential_3N_vec), np.std(non_essential_3N_vec), len(non_essential_3N_vec))

print "pvalue is ",pvalue

print '**** bottom genes'
sorted_index = np.asarray(np.argsort(vec_3N))
for i in range(10):
    if essential_gene_dic[list(gene_set)[sorted_index[i]]] == 1:
        print list(gene_set)[sorted_index[i]] +'\t essential'
    else:
        print list(gene_set)[sorted_index[i]] + '\t not essential'
    print vec_3N[sorted_index[i]]


print '\n **** top genes '
sorted_index = np.asarray(np.argsort(vec_3N))[::-1]
for i in range(10):
    if essential_gene_dic[list(gene_set)[sorted_index[i]]] == 1:
        print list(gene_set)[sorted_index[i]] +'\t essential'
    else:
        print list(gene_set)[sorted_index[i]] + '\t not essential'
    print vec_3N[sorted_index[i]]


essential_counter = 0.
sorted_index = np.asarray(np.argsort(vec_3N))[::-1]
for i in range(len(sorted_index)):
    if essential_gene_dic[list(gene_set)[sorted_index[i]]] == 1:
        essential_counter+=1
    if i==10:
        print "Portion of Essential Genes in top-10 = %.3f", essential_counter/10.
    if i==25:
        print "Portion of Essential Genes in top-25 = %.3f", essential_counter/25.
    if i==50:
        print "Portion of Essential Genes in top-50 = %.3f", essential_counter/50.
    if i==100:
        print "Portion of Essential Genes in top-100 = %.3f", essential_counter/100.
    if i == 200:
        print "Portion of Essential Genes in top-200 = %.3f", essential_counter / 200.
    if i == 300:
        print "Portion of Essential Genes in top-300 = %.3f", essential_counter / 300.






## print the list of genes
# file = open('3N-length-genes.txt',"w")
# for i in range(len(gene_set)):
#     file.write('%s\n' %list(gene_set)[sorted_index[i]])


#### here I plot the hist of top and bottom genes to make sure that they are different
### they are actually pretty different

# top
sorted_index = np.asarray(np.argsort(vec_3N))[::-1]
size_dic = dict.fromkeys(size_dic, 0)
for gene_index in sorted_index[:50]:
    site_index_list = np.where(np.asarray(gene_list) == list(gene_set)[gene_index])[0]
    for site_index in site_index_list:
        for indel_index in range(indel_num):
            if dic_del_size_del[indel_index]:
                size_dic[dic_del_size_del[indel_index][0]] += indel_fraction_mutant_matrix[indel_index, site_index]

size_freq = []
for i in range(np.size(size_vec_unique)):
    size_freq.append(size_dic[size_vec_unique[i]])
len_dist_top = np.asarray(size_freq/sum(size_freq))

# bottom
sorted_index = np.asarray(np.argsort(vec_3N))
size_dic = dict.fromkeys(size_dic, 0)
for gene_index in sorted_index[:50]:
    site_index_list = np.where(np.asarray(gene_list) == list(gene_set)[gene_index])[0]
    for site_index in site_index_list:
        for indel_index in range(indel_num):
            if dic_del_size_del[indel_index]:
                size_dic[dic_del_size_del[indel_index][0]] += indel_fraction_mutant_matrix[indel_index, site_index]

size_freq = []
for i in range(np.size(size_vec_unique)):
    size_freq.append(size_dic[size_vec_unique[i]])
len_dist_bottom = np.asarray(size_freq/sum(size_freq))

print size_freq
plt.plot(size_vec_unique,len_dist_top , 'o')
plt.plot(size_vec_unique,len_dist_bottom , 'o')
plt.ylabel('Sum of Fractions')
plt.xlabel('Length')
plt.legend(['Top 50 Genes',"Bottom 50 Genes"], loc=1)
plt.title('3N')
plt.savefig('Length_per_gene_folder/hist_of_top_and_bottom_genes_3n.pdf')
#plt.savefig('Length_per_gene_folder/all_genes.pdf')
plt.clf()









# #####
# print '**** bottom genes - measure 2'
# sorted_index = np.asarray(np.argsort(vec_3N_difference))
# for i in range(10):
#     print list(gene_set)[sorted_index[i]]
#     print vec_3N_difference[sorted_index[i]]

# print '\n **** top genes  - measure 2'
# sorted_index = np.asarray(np.argsort(vec_3N_difference))[::-1]
# for i in range(10):
#     if essential_gene_dic[list(gene_set)[sorted_index[i]]] == 1:
#         print list(gene_set)[sorted_index[i]] + '\t essential'
#     else:
#         print list(gene_set)[sorted_index[i]] + '\t not essential'
#     print vec_3N_difference[sorted_index[i]]

# print '\n **** top genes  - measure 2'
# sorted_index = np.asarray(np.argsort(vec_3N_difference))[::-1]
# for i in range(10):
#     print list(gene_set)[sorted_index[i]]
#     print vec_3N_difference[sorted_index[i]]