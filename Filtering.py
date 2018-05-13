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
import networkx as nx
from sequence_logos import plot_seq_logo
from sklearn.metrics import jaccard_similarity_score
import collections
from operator import itemgetter
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import chi2
import csv




data_folder = "../IndelsFullData/"
sequence_file_name = "sequence_pam_gene_grna_big_file_donor_genomic_context.csv"
#data_folder = "/Users/amirali/Projects/CRISPR-data/R data/AM_TechMerg_Summary/"
data_folder = "/Users/amirali/Projects/CRISPR-data-Feb18/20nt_counts_only/"



# name_genes_grna_unique = pickle.load(open('storage/name_genes_grna_unique.p', 'rb'))
# name_indel_type_unique = pickle.load(open('storage/name_indel_type_unique.p', 'rb'))
# indel_count_matrix = pickle.load(open('storage/indel_count_matrix.p', 'rb'))
# indel_prop_matrix = pickle.load(open('storage/indel_prop_matrix.p', 'rb'))
# length_indel_insertion = pickle.load(open('storage/length_indel_insertion.p', 'rb'))
# length_indel_deletion = pickle.load(open('storage/length_indel_deletion.p', 'rb'))
#
# # input
# spcer_dict={}
# with open(sequence_file_name, 'rb') as csvfile:
#   spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#   row_counter = 0
#   for row in spamreader:
#     spcer_dict[row[0].split(',')[1] + '-' +row[0].split(',')[0]] = row[0].split(',')[2] + row[0].split(',')[3]
#
# spacer_pam_list = []
# for site in name_genes_grna_unique:
#   spacer_pam_list.append(spcer_dict[site])
#
# pickle.dump(name_genes_grna_unique, open('Tcell-files/name_genes_grna_ALL.p', 'wb'))
# pickle.dump(name_indel_type_unique, open('Tcell-files/name_indel_type_ALL.p', 'wb'))
# pickle.dump(indel_count_matrix, open('Tcell-files/indel_count_matrix_ALL.p', 'wb'))
# pickle.dump(indel_prop_matrix, open('Tcell-files/indel_prop_matrix_ALL.p', 'wb'))
# pickle.dump(length_indel_insertion, open('Tcell-files/length_indel_insertion_ALL.p', 'wb'))
# pickle.dump(length_indel_deletion, open('Tcell-files/length_indel_deletion_ALL.p', 'wb'))
# pickle.dump(spacer_pam_list, open('Tcell-files/spacer_pam_list_ALL.p', 'wb'))



#name_genes_grna_unique = pickle.load(open('Tcell-files/name_genes_grna_ALL.p', 'rb'))
#name_indel_type_unique = pickle.load(open('Tcell-files/name_indel_type_ALL.p', 'rb'))
#indel_count_matrix = pickle.load(open('Tcell-files/indel_count_matrix_ALL.p', 'rb'))
#indel_prop_matrix = pickle.load(open('Tcell-files/indel_prop_matrix_ALL.p', 'rb'))
#length_indel_insertion = pickle.load(open('Tcell-files/length_indel_insertion_ALL.p', 'rb'))
#length_indel_deletion = pickle.load(open('Tcell-files/length_indel_deletion_ALL.p', 'rb'))
#spacer_pam_list = pickle.load(open('Tcell-files/spacer_pam_list_ALL.p', 'rb'))

##############
#
# selected_spacers = []
# selected_index = []
# for counter1,spacer1 in enumerate(spacer_pam_list):
#   if spacer1 not in selected_spacers:
#     local_counter = []
#     local_count_sum = []
#     local_spacer = []
#     for counter2,spacer2 in enumerate(spacer_pam_list):
#       if spacer2==spacer1:
#         local_counter.append(counter2)
#         local_count_sum.append(np.sum(indel_count_matrix[:,counter2]))
#         local_spacer.append(spacer2)
#
#     selected_index.append(local_counter[np.argmax(local_count_sum)])
#     selected_spacers.append(local_spacer[np.argmax(local_count_sum)])
#
#
# name_genes_grna_unique = list(np.asarray(name_genes_grna_unique)[selected_index])
# indel_count_matrix = indel_count_matrix[:,selected_index]
# indel_prop_matrix = indel_prop_matrix[:,selected_index]


# pickle.dump(name_genes_grna_unique, open('Tcell-files/name_genes_grna_UNIQUE.p', 'wb'))
# pickle.dump(indel_count_matrix, open('Tcell-files/indel_count_matrix_UNIQUE.p', 'wb'))
# pickle.dump(indel_prop_matrix, open('Tcell-files/indel_prop_matrix_UNIQUE.p', 'wb'))
# pickle.dump(selected_spacers, open('Tcell-files/spacer_pam_list_UNIQUE.p', 'wb'))




eff_vec_mean_no_others = []

spacer_pam_list_ALL = pickle.load(open('Tcell-files/spacer_pam_list_ALL.p', 'rb'))
spacer_pam_list_UNIQUE = pickle.load(open('Tcell-files/spacer_pam_list_UNIQUE.p', 'rb'))
eff_vec_BIG = pickle.load(open('storage/eff_vec_BIG_no_others.p', 'rb'))

high_variance_indicator = np.zeros(len(spacer_pam_list_UNIQUE))

for count1,spacer1 in enumerate(spacer_pam_list_UNIQUE):
  local_eff_vec = []
  for count2,spacer2 in enumerate(spacer_pam_list_ALL):
    if spacer1==spacer2:
      local_eff_vec.append(eff_vec_BIG[count2])
  eff_vec_mean_no_others.append(np.mean(local_eff_vec))

  if np.std(local_eff_vec)>0.04:
    high_variance_indicator[count1]=1



  print np.mean(local_eff_vec)
  print np.std(local_eff_vec)
  print np.shape(local_eff_vec)
  print "----"


print sum(high_variance_indicator)
pickle.dump(high_variance_indicator, open('Tcell-files/high_variance_indicator.p', 'wb'))

#pickle.dump(eff_vec_mean_no_others, open('Tcell-files/eff_vec_mean_UNIQUE_others_in_numinator.p', 'wb'))