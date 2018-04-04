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


def top_indel_finder(indel_count_matrix,name_indel_type_unique):
    indel_num,site_num = np.shape(indel_count_matrix)
    top_indel_type_vector = np.zeros(site_num)
    for site in range(site_num):
        if 'I' in name_indel_type_unique[np.argmax(indel_count_matrix[:,site])]:
            top_indel_type_vector[site] = 1
    return top_indel_type_vector

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

def longest_substring(strng,character):
    len_substring=0
    longest=0
    for i in range(len(strng)):
        if i > 1:
            if strng[i] != strng[i-1] or strng[i] != character:
                len_substring = 0
        if strng[i] == character:
            len_substring += 1
        if len_substring > longest:
            longest = len_substring
    return longest

def indel_length_finder(indel_count_matrix,length_indel_insertion,length_indel_deletion,consider_length):
  indel_num,site_num = np.shape(indel_count_matrix)

  prop_insertions_gene_grna = np.zeros(site_num,dtype=float)
  prop_deletions_gene_grna = np.zeros(site_num,dtype=float)

  if consider_length == 0:
    length_indel_insertion[length_indel_insertion>0]=1.
    length_indel_deletion[length_indel_deletion>0]=1.

  indel_fraction_mutant_matrix = indel_count_matrix / np.reshape(np.sum(indel_count_matrix, axis=0), (1, -1))

  for site_index in range(site_num):
    prop_insertions_gene_grna[site_index] = np.inner(length_indel_insertion,indel_fraction_mutant_matrix[:,site_index])
    prop_deletions_gene_grna[site_index] = np.inner(length_indel_deletion, indel_fraction_mutant_matrix[:, site_index])

  return prop_insertions_gene_grna,prop_deletions_gene_grna

data_folder = "../IndelsFullData/"
sequence_file_name = "sequence_pam_gene_grna_big_file_donor_genomic_context.csv"
#data_folder = "/Users/amirali/Projects/CRISPR-data/R data/AM_TechMerg_Summary/"
data_folder = "/Users/amirali/Projects/CRISPR-data-Feb18/20nt_counts_only/"

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

num_indels,num_sites = np.shape(indel_count_matrix)


# extract genomic context
context_genome_dict = {}
spacer_dict = {}
with open('sequence_pam_gene_grna_big_file_donor_genomic_context.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    row_counter = 0
    for row in spamreader:
        context = row[0].split(',')[6]
        context = context.replace('a','A')
        context = context.replace('c','C')
        context = context.replace('t','T')
        context = context.replace('g','G')

        spacer_dict[row[0].split(',')[0]]=row[0].split(',')[2]
        context_genome_dict[row[0].split(',')[0]] = context


# extract homology matrix
homology_matrix =  np.zeros((4,num_sites))
site_count = 0
for site_name in name_genes_grna_unique:
  site_name_list = site_name.split('-')
  context = context_genome_dict[site_name_list[1] + '-' + site_name_list[2]]
  nuc_count = 0
  for nuc in ['A', 'C', 'G', 'T']:
    homology_matrix[nuc_count,site_count] = int(longest_substring_passing_cutsite(context[50-2:50+2], nuc))
    #print context[50-3:50+3]
    nuc_count+=1
  site_count+=1

print homology_matrix
print 'max', np.max(homology_matrix)
print 'min', np.min(homology_matrix)
homopolymer_matrix = np.copy(homology_matrix)
pickle.dump(homopolymer_matrix, open('storage/homopolymer_matrix_w-2:2.p', 'wb'))

consider_length = 1 #0 means do not consider lengths, 1 mean expected length
prop_insertions_gene_grna,prop_deletions_gene_grna = indel_length_finder(indel_count_matrix,length_indel_insertion,length_indel_deletion,consider_length)


# homopolymer length vs deletion length
nuc_count = 0
for nuc in ['A', 'C', 'G', 'T']:
  print nuc
  mean_vec = []
  std_vec = []
  length_vec = []
  for string_length in range(0,4):
    print 'string length =', string_length
    mean_vec.append(np.mean(prop_deletions_gene_grna[homology_matrix[nuc_count,:] == string_length]))
    std_vec.append(np.std(prop_deletions_gene_grna[homology_matrix[nuc_count,:] == string_length]))
    length_vec.append(np.size(prop_deletions_gene_grna[homology_matrix[nuc_count,:] == string_length]) )

  print length_vec
  #plt.plot(homology_matrix[nuc_count,:],prop_deletions_gene_grna,'o')
  plt.errorbar([0,1,2,3], mean_vec, yerr=std_vec)
  # plt.ylabel('Marginal Prob.')
  plt.xticks([-1,0,1,2,3,4])
  plt.xlabel('Homopolymer Length')
  plt.ylabel('Expected Deletion Length')
  plt.title('Nucleotide ' + nuc)
  # plt.legend(['TT Type','Wild Type'])
  plt.savefig('plots/homolohy_deletion_length_'+ nuc +'.pdf')
  plt.clf()
  nuc_count+=1

  for i in range(np.size(mean_vec)-1):
    tstat, pvalue = ttest_ind_from_stats(mean_vec[i], std_vec[i], length_vec[i],
                                         mean_vec[i+1], std_vec[i+1], length_vec[i+1])
    print pvalue


print '--------------------------'

# homopolymer length vs top top indel type
top_indel_vec = top_indel_finder(indel_count_matrix, name_indel_type_unique)
nuc_count = 0
for nuc in ['A', 'C', 'G', 'T']:
  print nuc
  mean_vec = []
  std_vec = []
  length_vec = []

  for string_length in range(0, 4):
    print 'string length =', string_length
    mean_vec.append(np.mean(top_indel_vec[homology_matrix[nuc_count, :] == string_length]))
    std_vec.append(np.std(top_indel_vec[homology_matrix[nuc_count, :] == string_length]))
    length_vec.append(np.size(top_indel_vec[homology_matrix[nuc_count, :] == string_length]))

  print length_vec
  # plt.plot(homology_matrix[nuc_count,:],prop_deletions_gene_grna,'o')
  plt.errorbar([0, 1, 2, 3], mean_vec, yerr=std_vec)
  # plt.ylabel('Marginal Prob.')
  plt.xticks([-1, 0, 1, 2, 3, 4])
  plt.xlabel('Homopolymer Length')
  plt.ylabel('Indel Type')
  plt.title('Nucleotide ' + nuc)
  plt.savefig('plots/homolohy_indel_type_' + nuc + '.pdf')
  plt.clf()

  for i in range(np.size(mean_vec) - 1):
    tstat, pvalue = ttest_ind_from_stats(mean_vec[i], std_vec[i], length_vec[i],
                                         mean_vec[i + 1], std_vec[i + 1], length_vec[i + 1])
    print pvalue

  nuc_count+=1