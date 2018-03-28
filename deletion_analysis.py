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


data_folder = "../IndelsFullData/"
sequence_file_name = "sequence_pam_gene_grna_big_file_donor.csv"
data_folder = "/Users/amirali/Projects/CRISPR-data-Feb18/20nt_counts_only/"

print "loading files"
name_genes_grna_unique = pickle.load(open('storage/name_genes_grna_unique_one_patient_per_site.p', 'rb'))
name_indel_type_unique = pickle.load(open('storage/name_indel_type_unique.p', 'rb'))
indel_count_matrix = pickle.load(open('storage/indel_count_matrix_one_patient_per_site.p', 'rb'))
indel_prop_matrix = pickle.load(open('storage/indel_prop_matrix_one_patient_per_site.p', 'rb'))
length_indel = pickle.load(open('storage/length_indel.p', 'rb'))

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

        # if indel_types[i]=='I':
        #     start, stop = indel_locations[i].split(':')
    counter = counter + 1

context_len = max_stop - min_start + 1
context_read_count = np.zeros((1,context_len))
context_read_count_start = np.zeros((1,context_len))
context_read_count_stop = np.zeros((1,context_len))
context_read_count_deleted = np.zeros((1,context_len))

for indel_index in range(indel_num):
    list = dic_del[indel_index]
    list_start = dic_del_start[indel_index]
    list_stop = dic_del_stop[indel_index]
    read_count = indel_count_matrix_sum[indel_index]
    for i in range(len(list)):
        context_read_count[0,list[i]+abs(min_start)] += read_count
    for i in range(len(list_start)):
        context_read_count_start[0,list_start[i]+abs(min_start)] += read_count
    for i in range(len(list_stop)):
        context_read_count_stop[0,list_stop[i]+abs(min_start)] += read_count
    for i in range(len(list_stop)):
        context_read_count_deleted[0, (list_start[i]+abs(min_start)) : (list_stop[i] + abs(min_start))] += read_count


plt.plot(range(min_start,max_stop+1),context_read_count_deleted[0,:]/np.sum(context_read_count_deleted[0,:]))
plt.ylabel('Marginal Prob.')
plt.xlabel('Genomic Context Location')
plt.title('Deleted Locations')
plt.savefig('plots/deletion_context_location_prob.pdf')
plt.clf()

plt.plot(range(min_start,max_stop+1),context_read_count[0,:]/np.sum(context_read_count[0,:]))
plt.ylabel('Marginal Prob.')
plt.xlabel('Genomic Context Location')
plt.title('Deletion Boundary Location')
plt.savefig('plots/deletion_start_stop_context_location_prob.pdf')
plt.clf()

plt.plot(range(min_start,max_stop+1),context_read_count_start[0,:]/np.sum(context_read_count_start[0,:]))
plt.ylabel('Marginal Prob.')
plt.xlabel('Genomic Context Location')
plt.title('Deletion Start Location')
plt.savefig('plots/deletion_start_context_location_prob.pdf')
plt.clf()

plt.plot(range(min_start,max_stop+1),context_read_count_stop[0,:]/np.sum(context_read_count_stop[0,:]))
plt.ylabel('Marginal Prob.')
plt.xlabel('Genomic Context Location')
plt.title('Deletion Stop Location')
plt.savefig('plots/deletion_stop_context_location_prob.pdf')
plt.clf()

# print "variation of lengths"
# print "mean length =", np.dot(dic_del_len,indel_count_matrix_sum) / np.sum(indel_count_matrix_sum)
# print dic_del_len
# print indel_count_matrix_sum
# #print "std length =", np.std(np.dot(dic_del_len,indel_count_matrix_sum))


# ########### Creat Deletion Marginal Probablity Matrix
context_probability_matrix = np.zeros((context_len,site_num))
for indel_index in range(indel_num):
    #list = dic_del[indel_index]
    list_start = dic_del_start[indel_index]
    list_stop = dic_del_stop[indel_index]
    for site in range(site_num):
        for i in range(len(list_start)):
            context_probability_matrix[list_start[i]+abs(min_start):list_stop[i]+abs(min_start),site] += indel_fraction_mutant_matrix[indel_index,site]
deletion_context_probability_matrix = context_probability_matrix /  np.reshape(np.sum(context_probability_matrix, axis=0), (1, -1))
pickle.dump(deletion_context_probability_matrix, open('storage/deletion_context_probability_matrix.p', 'wb'))

for site_id in range(5):
    plt.plot(range(min_start,max_stop+1),deletion_context_probability_matrix[:,site_id])
    plt.ylabel('Marginal Prob.')
    plt.xlabel('Genomic Context Location')
    #plt.title('Deletion Boundary Location')
    plt.savefig('plots/deletion_context_location_prob_single_site'+ str(site_id) +'.pdf')
    plt.clf()


# ########### Creat all Genomic context file
context_genome_dict = {}
with open('sequence_pam_gene_grna_big_file_donor_genomic_context.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    row_counter = 0
    for row in spamreader:
        context_genome_dict[row[0].split(',')[0]]=row[0].split(',')[6]

counter = 0
file=open('storage/genomic_context.txt','w')
for site_name in name_genes_grna_unique:
    site_name_list = site_name.split('-')
    file.write('%s\n' %context_genome_dict[site_name_list[1]+'-'+site_name_list[2]])
    counter += 1

###########  Analyzing TT

# two_letter = 'AG'
# TT_index = []
# TT_index_not = []
# spacer_dict = {}
# with open('sequence_pam_gene_grna_big_file_donor_genomic_context.csv', 'rb') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#     row_counter = 0
#     for row in spamreader:
#         spacer_dict[row[0].split(',')[0]] = row[0].split(',')[2][16:18]
#
# counter = 0
# for site_name in name_genes_grna_unique:
#     site_name_list = site_name.split('-')
#     if spacer_dict[site_name_list[1]+'-'+site_name_list[2]] == two_letter:
#         TT_index.append(counter)
#     else:
#         TT_index_not.append(counter)
#     counter += 1
#
# ##
#
# indel_fraction_mutant_matrix = indel_count_matrix / np.reshape(np.sum(indel_count_matrix, axis=0), (1, -1))
# indel_count_matrix_sum = np.sum(indel_fraction_mutant_matrix[:,TT_index],axis=1)
#
# context_read_count_TT = np.zeros((1,context_len))
# context_read_count_start_TT = np.zeros((1,context_len))
# context_read_count_stop_TT = np.zeros((1,context_len))
#
# for indel_index in range(indel_num):
#     list = dic_del[indel_index]
#     list_start = dic_del_start[indel_index]
#     list_stop = dic_del_stop[indel_index]
#     read_count = indel_count_matrix_sum[indel_index]
#     for i in range(len(list)):
#         context_read_count_TT[0,list[i]+abs(min_start)] += read_count
#     for i in range(len(list_start)):
#         context_read_count_start_TT[0,list_start[i]+abs(min_start)] += read_count
#     for i in range(len(list_stop)):
#         context_read_count_stop_TT[0,list_stop[i]+abs(min_start)] += read_count
#
# ##
#
# indel_fraction_mutant_matrix = indel_count_matrix / np.reshape(np.sum(indel_count_matrix, axis=0), (1, -1))
# indel_count_matrix_sum = np.sum(indel_fraction_mutant_matrix[:,TT_index_not],axis=1)
#
# context_read_count_no_TT = np.zeros((1,context_len))
# context_read_count_start_no_TT = np.zeros((1,context_len))
# context_read_count_stop_no_TT = np.zeros((1,context_len))
#
# for indel_index in range(indel_num):
#     list = dic_del[indel_index]
#     list_start = dic_del_start[indel_index]
#     list_stop = dic_del_stop[indel_index]
#     read_count = indel_count_matrix_sum[indel_index]
#     for i in range(len(list)):
#         context_read_count_no_TT[0,list[i]+abs(min_start)] += read_count
#     for i in range(len(list_start)):
#         context_read_count_start_no_TT[0,list_start[i]+abs(min_start)] += read_count
#     for i in range(len(list_stop)):
#         context_read_count_stop_no_TT[0,list_stop[i]+abs(min_start)] += read_count
#
# plt.plot(range(min_start,max_stop+1),context_read_count_TT[0,:]/np.sum(context_read_count_TT[0,:]))
# plt.plot(range(min_start,max_stop+1),context_read_count_no_TT[0,:]/np.sum(context_read_count_no_TT[0,:]),'r')
# plt.ylabel('Marginal Prob.')
# plt.xlabel('Genomic Context Location')
# plt.title('Deletion Boundary Location')
# plt.legend(['TT Type','Wild Type'])
# plt.savefig('plots/deletion_context_location_prob.pdf')
# plt.clf()
#
# print "KL distance"
# print entropy(context_read_count_TT[0,:]/np.sum(context_read_count_TT[0,:]), context_read_count_no_TT[0,:]/np.sum(context_read_count_no_TT[0,:]))
#
# plt.plot(range(min_start,max_stop+1),context_read_count_start_TT[0,:]/np.sum(context_read_count_start_TT[0,:]))
# plt.plot(range(min_start,max_stop+1),context_read_count_start_no_TT[0,:]/np.sum(context_read_count_start_no_TT[0,:]),'r')
# plt.ylabel('Marginal Prob.')
# plt.xlabel('Genomic Context Location')
# plt.title('Deletion Boundary Location')
# plt.legend(['TT Type','Wild Type'])
# plt.savefig('plots/deletion_start_context_location_prob.pdf')
#
# plt.clf()
#
# plt.plot(range(min_start,max_stop+1),context_read_count_stop_TT[0,:]/np.sum(context_read_count_stop_TT[0,:]))
# plt.plot(range(min_start,max_stop+1),context_read_count_stop_no_TT[0,:]/np.sum(context_read_count_stop_no_TT[0,:]),'r')
# plt.ylabel('Marginal Prob.')
# plt.xlabel('Genomic Context Location')
# plt.title('Deletion Boundary Location')
# plt.legend(['TT Type','Wild Type'])
# plt.savefig('plots/deletion_stop_context_location_prob.pdf')
# plt.clf()


####### Analyzing T

# TT_index = []
# TT_index_not = []
# spacer_dict = {}
# with open('sequence_pam_gene_grna_big_file_donor_genomic_context.csv', 'rb') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#     row_counter = 0
#     for row in spamreader:
#         spacer_dict[row[0].split(',')[0]] = row[0].split(',')[2][17]
#
# counter = 0
# for site_name in name_genes_grna_unique:
#     site_name_list = site_name.split('-')
#     if spacer_dict[site_name_list[1]+'-'+site_name_list[2]] == 'T':
#         TT_index.append(counter)
#     else:
#         TT_index_not.append(counter)
#     counter += 1
#
# print len(TT_index)
#
# ##
#
# indel_fraction_mutant_matrix = indel_count_matrix / np.reshape(np.sum(indel_count_matrix, axis=0), (1, -1))
# indel_count_matrix_sum = np.sum(indel_fraction_mutant_matrix[:,TT_index],axis=1)
#
# context_read_count_TT = np.zeros((1,context_len))
# context_read_count_start_TT = np.zeros((1,context_len))
# context_read_count_stop_TT = np.zeros((1,context_len))
#
# for indel_index in range(indel_num):
#     list = dic_del[indel_index]
#     list_start = dic_del_start[indel_index]
#     list_stop = dic_del_stop[indel_index]
#     read_count = indel_count_matrix_sum[indel_index]
#     for i in range(len(list)):
#         context_read_count_TT[0,list[i]+abs(min_start)] += read_count
#     for i in range(len(list_start)):
#         context_read_count_start_TT[0,list_start[i]+abs(min_start)] += read_count
#     for i in range(len(list_stop)):
#         context_read_count_stop_TT[0,list_stop[i]+abs(min_start)] += read_count
#
# ##
#
# indel_fraction_mutant_matrix = indel_count_matrix / np.reshape(np.sum(indel_count_matrix, axis=0), (1, -1))
# indel_count_matrix_sum = np.sum(indel_fraction_mutant_matrix[:,TT_index_not],axis=1)
#
# context_read_count_no_TT = np.zeros((1,context_len))
# context_read_count_start_no_TT = np.zeros((1,context_len))
# context_read_count_stop_no_TT = np.zeros((1,context_len))
#
# for indel_index in range(indel_num):
#     list = dic_del[indel_index]
#     list_start = dic_del_start[indel_index]
#     list_stop = dic_del_stop[indel_index]
#     read_count = indel_count_matrix_sum[indel_index]
#     for i in range(len(list)):
#         context_read_count_no_TT[0,list[i]+abs(min_start)] += read_count
#     for i in range(len(list_start)):
#         context_read_count_start_no_TT[0,list_start[i]+abs(min_start)] += read_count
#     for i in range(len(list_stop)):
#         context_read_count_stop_no_TT[0,list_stop[i]+abs(min_start)] += read_count
#
# plt.plot(range(min_start,max_stop+1),context_read_count_TT[0,:]/np.sum(context_read_count_TT[0,:]))
# plt.plot(range(min_start,max_stop+1),context_read_count_no_TT[0,:]/np.sum(context_read_count_no_TT[0,:]),'r')
# plt.ylabel('Marginal Prob.')
# plt.xlabel('Genomic Context Location')
# plt.title('Deletion Boundary Location')
# plt.savefig('plots/deletion_context_location_prob.pdf')
# plt.clf()
#
# print "KL distance"
# print entropy(context_read_count_TT[0,:]/np.sum(context_read_count_TT[0,:]), context_read_count_no_TT[0,:]/np.sum(context_read_count_no_TT[0,:]))
#
# plt.plot(range(min_start,max_stop+1),context_read_count_start_TT[0,:]/np.sum(context_read_count_start_TT[0,:]))
# plt.plot(range(min_start,max_stop+1),context_read_count_start_no_TT[0,:]/np.sum(context_read_count_start_no_TT[0,:]),'r')
# plt.ylabel('Marginal Prob.')
# plt.xlabel('Genomic Context Location')
# plt.title('Deletion Boundary Location')
# plt.savefig('plots/deletion_start_context_location_prob.pdf')
# plt.clf()
#
# plt.plot(range(min_start,max_stop+1),context_read_count_stop_TT[0,:]/np.sum(context_read_count_stop_TT[0,:]))
# plt.plot(range(min_start,max_stop+1),context_read_count_stop_no_TT[0,:]/np.sum(context_read_count_stop_no_TT[0,:]),'r')
# plt.ylabel('Marginal Prob.')
# plt.xlabel('Genomic Context Location')
# plt.title('Deletion Boundary Location')
# plt.savefig('plots/deletion_stop_context_location_prob.pdf')
# plt.clf()