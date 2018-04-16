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

boundary_dic = {}
boundary_dic_control = {}
for nuc1 in ['A', 'C', 'G', 'T']:
    for nuc2 in ['A', 'C', 'G', 'T']:
        boundary_dic[nuc1+nuc2] = 0.

max_repeat = 10
for repeat in range(max_repeat):
    for nuc1 in ['A', 'C', 'G', 'T']:
        for nuc2 in ['A', 'C', 'G', 'T']:
            boundary_dic_control[repeat , nuc1 + nuc2] = 0.


size_dic = {}
size_coding_dic = {}
size_noncoding_dic = {}
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

            size_dic[size] = 0
            size_coding_dic[size] = 0
            size_noncoding_dic[size] = 0
            size_vec.append(size)

    counter = counter + 1

# context_len = max_stop - min_start + 1
# context_read_count = np.zeros((1,context_len))
# context_read_count_start = np.zeros((1,context_len))
# context_read_count_stop = np.zeros((1,context_len))
# context_read_count_deleted = np.zeros((1,context_len))
#
# for indel_index in range(indel_num):
#     list = dic_del[indel_index]
#     list_start = dic_del_start[indel_index]
#     list_stop = dic_del_stop[indel_index]
#     read_count = indel_count_matrix_sum[indel_index]
#     for i in range(len(list)):
#         context_read_count[0,list[i]+abs(min_start)] += read_count
#     for i in range(len(list_start)):
#         context_read_count_start[0,list_start[i]+abs(min_start)] += read_count
#     for i in range(len(list_stop)):
#         context_read_count_stop[0,list_stop[i]+abs(min_start)] += read_count
#     for i in range(len(list_stop)):
#         context_read_count_deleted[0, (list_start[i]+abs(min_start)) : (list_stop[i] + abs(min_start))] += read_count

#
# plt.plot(range(min_start,max_stop+1),context_read_count_deleted[0,:]/np.sum(context_read_count_deleted[0,:]))
# plt.ylabel('Marginal Prob.')
# plt.xlabel('Genomic Context Location')
# plt.title('Deleted Locations')
# plt.savefig('plots/deletion_context_location_prob.pdf')
# plt.clf()
#
# plt.plot(range(min_start,max_stop+1),context_read_count[0,:]/np.sum(context_read_count[0,:]))
# plt.ylabel('Marginal Prob.')
# plt.xlabel('Genomic Context Location')
# plt.title('Deletion Boundary Location')
# plt.savefig('plots/deletion_start_stop_context_location_prob.pdf')
# plt.clf()
#
# plt.plot(range(min_start,max_stop+1),context_read_count_start[0,:]/np.sum(context_read_count_start[0,:]))
# plt.ylabel('Marginal Prob.')
# plt.xlabel('Genomic Context Location')
# plt.title('Deletion Start Location')
# plt.savefig('plots/deletion_start_context_location_prob.pdf')
# plt.clf()
#
# plt.plot(range(min_start,max_stop+1),context_read_count_stop[0,:]/np.sum(context_read_count_stop[0,:]))
# plt.ylabel('Marginal Prob.')
# plt.xlabel('Genomic Context Location')
# plt.title('Deletion Stop Location')
# plt.savefig('plots/deletion_stop_context_location_prob.pdf')
# plt.clf()

# print "variation of lengths"
# print "mean length =", np.dot(dic_del_len,indel_count_matrix_sum) / np.sum(indel_count_matrix_sum)
# print dic_del_len
# print indel_count_matrix_sum
# #print "std length =", np.std(np.dot(dic_del_len,indel_count_matrix_sum))


# ########### Creat Deletion Marginal Probablity Matrix
# context_probability_matrix = np.zeros((context_len,site_num))
# for indel_index in range(indel_num):
#     #list = dic_del[indel_index]
#     list_start = dic_del_start[indel_index]
#     list_stop = dic_del_stop[indel_index]
#     for site in range(site_num):
#         for i in range(len(list_start)):
#             context_probability_matrix[list_start[i]+abs(min_start):list_stop[i]+abs(min_start),site] += indel_fraction_mutant_matrix[indel_index,site]
# deletion_context_probability_matrix = context_probability_matrix /  np.reshape(np.sum(context_probability_matrix, axis=0), (1, -1))
# pickle.dump(deletion_context_probability_matrix, open('storage/deletion_context_probability_matrix.p', 'wb'))
#
# for site_id in range(5):
#     plt.plot(range(min_start,max_stop+1),deletion_context_probability_matrix[:,site_id])
#     plt.ylabel('Marginal Prob.')
#     plt.xlabel('Genomic Context Location')
#     #plt.title('Deletion Boundary Location')
#     plt.savefig('plots/deletion_context_location_prob_single_site'+ str(site_id) +'.pdf')
#     plt.clf()

# ########### Creat all Genomic context file
context_genome_dict = {}
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

counter = 0
file=open('storage/genomic_context.txt','w')
for site_name in name_genes_grna_unique:
    site_name_list = site_name.split('-')
    file.write('%s\n' %context_genome_dict[site_name_list[1]+'-'+site_name_list[2]])
    counter += 1

print "context size =", len(context_genome_dict[site_name_list[1]+'-'+site_name_list[2]])
print "min start = ", min_start

###########

# ### here we find the distribution of boundaries
#
# for site, site_name in enumerate(name_genes_grna_unique):
#     site_name_list = site_name.split('-')
#     context = context_genome_dict[site_name_list[1] + '-' + site_name_list[2]]
#     for indel_index in range(indel_num):
#         list_start = dic_del_start[indel_index]
#         list_stop = dic_del_stop[indel_index]
#         for i in range(len(list_start)):
#
#             size_dic[list_stop[i] - list_start[i]] += indel_fraction_mutant_matrix[indel_index, site]
#
#             #print list_start[i]+abs(min_start)
#             if list_start[i]+50 < 99 and list_start[i]+50 >= 1 and list_stop[i]+50 < 99 and list_stop[i]+50 >= 1:
#
#                 nuc1 = context[list_start[i]+49]
#                 nuc2 = context[list_start[i]+50]
#
#                 nuc3 = context[list_stop[i]-1+50]
#                 nuc4 = context[list_stop[i]-1+51]
#
#                 # boundary correlation
#                 if nuc1!=nuc2 and nuc3!=nuc4: # alignment correction
#                     boundary_dic[nuc1 + nuc4] += indel_fraction_mutant_matrix[indel_index, site]
#                     repeat = 0
#                     while repeat<max_repeat:
#                         ## boundary correlation
#                         # this is uniform random
#                         #random_start = int(np.random.randint(0, 99-(list_stop[i] - list_start[i]+1), size=1))
#                         #nuc5 = context[random_start]
#                         #nuc8 = context[random_start + list_stop[i] - list_start[i]+1]
#
#                         # this is expoential
#                         random_start = 50 - int(np.random.exponential(4.5))/2
#                         random_stop = 50 + int(np.random.exponential(4.5))/2
#                         nuc5 = context[random_start]
#                         nuc8 = context[random_stop]
#                         #if nuc5!=context[random_start+1] and nuc8!=context[random_start + list_stop[i] - list_start[i]]:
#                         if nuc5!=context[random_start+1] and nuc8!=context[random_stop-1] and random_start!=random_stop and random_start+1!=random_stop:
#                             boundary_dic_control[repeat, nuc5 + nuc8] += indel_fraction_mutant_matrix[indel_index, site]
#                             repeat += 1
#
#
#                 # ## boundary
#                 # boundary_dic[nuc1 + nuc2] += indel_fraction_mutant_matrix[indel_index, site]
#                 # boundary_dic[nuc3 + nuc4] += indel_fraction_mutant_matrix[indel_index, site]
#                 #
#                 # repeat = 0
#                 # while repeat < max_repeat:
#                 #     ## boundary
#                 #     random_start = 50 - int(np.random.exponential(4.5)) / 2
#                 #     random_stop = 50 + int(np.random.exponential(4.5)) / 2
#                 #
#                 #     nuc5 = context[random_start]
#                 #     nuc6 = context[random_start+1]
#                 #
#                 #     nuc7 = context[random_stop]
#                 #     nuc8 = context[random_stop+1]
#                 #
#                 #     if random_start != random_stop:
#                 #         boundary_dic_control[repeat,nuc5+nuc6] += indel_fraction_mutant_matrix[indel_index,site]
#                 #         boundary_dic_control[repeat,nuc7+nuc8] += indel_fraction_mutant_matrix[indel_index,site]
#                 #         repeat += 1
#         # add stop
#
#
# print boundary_dic
#
# legend_nuc = []
# vec_to_plot = []
# vec_to_plot_control = []
# for repeat in range(max_repeat):
#     vec_to_plot_control.append([])
# for nuc1 in ['A', 'C', 'G', 'T']:
#     for nuc2 in ['A', 'C', 'G', 'T']:
#         vec_to_plot.append(boundary_dic[nuc1+nuc2])
#         legend_nuc.append(nuc1+nuc2)
#
# for repeat in range(max_repeat):
#     for nuc1 in ['A', 'C', 'G', 'T']:
#         for nuc2 in ['A', 'C', 'G', 'T']:
#             vec_to_plot_control[repeat].append(boundary_dic_control[repeat,nuc1 + nuc2])
#
# vec_to_plot_control = np.asarray(vec_to_plot_control)
# vec_to_plot_control = vec_to_plot_control / np.reshape(np.sum(vec_to_plot_control, axis=1), (-1, 1))
#
# print vec_to_plot_control
#
# plt.stem(range(0,16),vec_to_plot/np.sum(vec_to_plot))
# plt.errorbar(range(0,16),np.mean(vec_to_plot_control,axis=0) , yerr = np.std(vec_to_plot_control,axis=0),color='r')
# #plt.stem(vec_to_plot_control/np.sum(vec_to_plot_control),'r')
# plt.ylabel('Prob.')
# plt.xticks(range(0,16),legend_nuc)
# plt.legend(['Empirical Distribution','Random Control'],loc=3)
# plt.savefig('plots/deletion_boundary_corrrelation_exp.pdf')
# plt.clf()

# #####
#
# boundary_dic = {}
# boundary_dic_control = {}
# for nuc1 in ['A', 'C', 'G', 'T']:
#     for nuc2 in ['A', 'C', 'G', 'T']:
#         for nuc3 in ['A', 'C', 'G', 'T']:
#             for nuc4 in ['A', 'C', 'G', 'T']:
#                 boundary_dic[nuc1+nuc2+nuc3+nuc4] = 0.
#                 boundary_dic_control[nuc1+nuc2+nuc3+nuc4] = 0.
#
#
# site = 0
# for site_name in name_genes_grna_unique:
#     site_name_list = site_name.split('-')
#     context = context_genome_dict[site_name_list[1] + '-' + site_name_list[2]]
#     for indel_index in range(indel_num):
#         list_start = dic_del_start[indel_index]
#         list_stop = dic_del_stop[indel_index]
#         for i in range(len(list_start)):
#             #print list_start[i]+abs(min_start)
#             if list_start[i]+50 < 99 and list_start[i]+50 >= 1 and list_stop[i]+50 < 99 and list_stop[i]+50 >= 1:
#
#                 nuc1 = context[list_start[i]+49]
#                 nuc2 = context[list_start[i]+50]
#                 nuc3 = context[list_stop[i]-1+50]
#                 nuc4 = context[list_stop[i]-1+51]
#
#                 random_start = int(np.random.randint(0,99-list_stop[i]+list_start[i],size=1))
#                 nuc5 = context[random_start]
#                 nuc6 = context[random_start+1]
#                 nuc7 = context[random_start+list_stop[i]-list_start[i]]
#                 nuc8 = context[random_start+list_stop[i]-list_start[i]]
#
#                 size_dic[list_stop[i]-list_start[i]] += indel_fraction_mutant_matrix[indel_index,site]
#                 #print indel_fraction_mutant_matrix[indel_index,site]
#
#                 boundary_dic[nuc1+nuc2+nuc3+nuc4] += indel_fraction_mutant_matrix[indel_index,site]
#                 boundary_dic_control[nuc5+nuc6+nuc7+nuc8] += indel_fraction_mutant_matrix[indel_index,site]
#         # add stop
#     site += 1
#
# print boundary_dic
#
# legend_nuc = []
# vec_to_plot = []
# vec_to_plot_control = []
# for nuc1 in ['A', 'C', 'G', 'T']:
#     for nuc2 in ['A', 'C', 'G', 'T']:
#         for nuc3 in ['A', 'C', 'G', 'T']:
#             for nuc4 in ['A', 'C', 'G', 'T']:
#                 vec_to_plot.append(boundary_dic[nuc1+nuc2+nuc3+nuc4])
#                 vec_to_plot_control.append(boundary_dic_control[nuc1+nuc2+nuc3+nuc4])
#                 legend_nuc.append(nuc1+nuc2+nuc3+nuc4)
#
# aa = vec_to_plot/np.sum(vec_to_plot)
# bb = vec_to_plot_control/np.sum(vec_to_plot_control)
# plt.stem(aa[0:20])
# plt.stem(bb[0:20],'r')
# plt.ylabel('Prob.')
# plt.xticks(range(0,20),legend_nuc[0:20])
# plt.legend(['Empirical Distribution','Random Control'],loc=3)
# plt.savefig('plots/deletion_boundary.pdf')
# plt.clf()
# # ###############
#
#### plottign the length of deletions

site = 0
intron_exon_label_vec = coding_region_finder(name_genes_grna_unique)
print "# of exons =",np.sum(intron_exon_label_vec==2)
print "# of introns =",np.sum(intron_exon_label_vec==1)
print "# of nongenes =", np.sum(intron_exon_label_vec==0)


for site_name in name_genes_grna_unique:
    site_name_list = site_name.split('-')
    #context = context_genome_dict[site_name_list[1] + '-' + site_name_list[2]]
    for indel_index in range(indel_num):
        list_start = dic_del_start[indel_index]
        list_stop = dic_del_stop[indel_index]
        for i in range(len(list_start)):
            size_dic[list_stop[i] - list_start[i]] += indel_fraction_mutant_matrix[indel_index, site]
            if intron_exon_label_vec[site] == 2:
                size_coding_dic[list_stop[i] - list_start[i]] += indel_fraction_mutant_matrix[indel_index, site]
            else:
                size_noncoding_dic[list_stop[i] - list_start[i]] += indel_fraction_mutant_matrix[indel_index, site]
    site += 1

size_vec_unique = np.sort(list(set(size_vec)))
size_freq = []
size_freq_coding = []
size_freq_noncoding = []
for i in range(np.size( size_vec_unique  )):
    size_freq.append(size_dic[size_vec_unique[i]])
    size_freq_coding.append(size_coding_dic[size_vec_unique[i]])
    size_freq_noncoding.append(size_noncoding_dic[size_vec_unique[i]])


error = 100
for p in np.linspace(0.001,0.99,100):
    new_error = np.linalg.norm(geom.pmf(np.asarray(size_vec_unique), p) - size_freq/np.sum(size_freq))
    if new_error<error:
        error = new_error
        bestpgeom = p

print "geometric error =", error
error = 100000
for lam in np.linspace(2,6,100): ## ***here lam is actual scaling***
    new_error = np.linalg.norm(expon.pdf(np.asarray(size_vec_unique),0, lam) - size_freq/np.sum(size_freq))
    if new_error<error:
        error = new_error
        bestlamexp = lam

print "exponential error =", error

print "p = ", bestpgeom
print "lambda = ", bestlamexp
size_freq_geom = geom.pmf(np.asarray(size_vec_unique), bestpgeom)
size_freq_exp = expon.pdf(np.asarray(size_vec_unique),0, bestlamexp)
plt.plot(size_vec_unique,size_freq/np.sum(size_freq),'o')
plt.plot(size_vec_unique,size_freq_geom,'ro')
plt.plot(size_vec_unique,size_freq_exp,'go')
plt.ylabel('Sum of Fractions')
plt.xlabel('Length')
plt.legend(['Empirical Distribution', 'Geometric Distribution p=%.2f'%bestpgeom,'Exponential Distribution scale=%.2f'%bestlamexp ], loc=1)
plt.savefig('plots/deletion_length_hist.pdf')
plt.clf()

plt.plot(size_vec_unique,size_freq_coding,'o')
plt.ylabel('Sum of Fractions')
plt.xlabel('Length')
#plt.legend(['Empirical Distribution', 'Random Control'], loc=3)
plt.savefig('plots/deletion_length_hist_coding.pdf')
plt.clf()

plt.plot(size_vec_unique,size_freq_noncoding,'o')
plt.ylabel('Sum of Fractions')
plt.xlabel('Length')
#plt.legend(['Empirical Distribution', 'Random Control'], loc=3)
plt.savefig('plots/deletion_length_hist_noncoding.pdf')
plt.clf()
#
