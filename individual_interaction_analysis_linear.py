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
import scipy.stats as stats
import math
from scipy.stats import ttest_ind_from_stats
from sequence_logos import plot_seq_logo
from sequence_logos import plot_QQ
from sklearn.metrics import mean_squared_error
from math import sqrt


def my_length_finder(indel_count_matrix,length_indel_insertion,length_indel_deletion,consider_length=1):
  indel_num,site_num = np.shape(indel_count_matrix)

  prop_insertions_gene_grna = np.zeros(site_num,dtype=float)
  prop_deletions_gene_grna = np.zeros(site_num,dtype=float)

  if consider_length ==0:
    length_indel_insertion[length_indel_insertion>0]=1.
    length_indel_deletion[length_indel_deletion>0]=1.


  indel_fraction_mutant_matrix = indel_count_matrix / np.reshape(np.sum(indel_count_matrix, axis=0), (1, -1))

  for site_index in range(site_num):
    prop_insertions_gene_grna[site_index] = np.inner(length_indel_insertion,indel_fraction_mutant_matrix[:,site_index])
    prop_deletions_gene_grna[site_index] = np.inner(length_indel_deletion, indel_fraction_mutant_matrix[:, site_index])

  return prop_insertions_gene_grna,prop_deletions_gene_grna

def plot_interaction_network(adj_list,accuracy_list,rmse_list,accu_no_interaction,rmse_no_interaction,coef_list,name_val):
  table_file = open('plots/table_' + name_val + '.text', 'w')
  table_file.write('Interaction\tR2 Gain\tRMSE Drop\t-log(p-value)\tCoefficient\n')

  G = nx.Graph()
  #num_edges = 5
  adj_sorted = np.sort(np.abs(adj_list), axis = None)
  #min_wt = adj_sorted[-num_edges]
  min_wt = -np.log10(0.05/np.size(adj_list))
  nucleotide_array = ['A', 'C', 'G', 'T']
  ij_counter = 0
  for i1 in range(4):
    for i2 in range(i1 + 1, 4):
      for i3 in range(4):
        for i4 in range(4):
          if np.abs(adj_list[ij_counter, i3, i4]) >= min_wt and accuracy_list[ij_counter, i3, i4]-accu_no_interaction > 0.0001:
            table_file.write('%s-%s\t\t%.4f\t%.4f\t\t\t%.2f\t\t\t%.4f\n'%(str(i1+16) + nucleotide_array[i3],str(i2+16) + nucleotide_array[i4], accuracy_list[ij_counter, i3, i4]-accu_no_interaction,  rmse_list[ij_counter, i3, i4]-rmse_no_interaction,adj_list[ij_counter, i3, i4],coef_list[ij_counter, i3, i4]))
            if coef_list[ij_counter, i3, i4] >= 0:
              G.add_edge(str(i1+16) + nucleotide_array[i3], str(i2+16) + nucleotide_array[i4], w = np.abs(adj_list[ij_counter, i3, i4]), c='b')
            if coef_list[ij_counter, i3, i4] < 0:
              G.add_edge(str(i1 + 16) + nucleotide_array[i3], str(i2 + 16) + nucleotide_array[i4],w=np.abs(adj_list[ij_counter, i3, i4]), c='g')

      ij_counter += 1

  plt.figure()
  edges = G.edges()
  colors = [G[u][v]['c'] for u,v in edges]
  weights = [G[u][v]['w'] for u,v in edges]
  #nx.draw(G, nx.circular_layout(G), font_size = 10, node_color = 'y', with_labels=True, edges=edges, edge_color=colors, width=weights)
  nx.draw(G, nx.circular_layout(G), font_size=10, node_color='y', with_labels=True, edge_color=colors)
  pos = nx.get_node_attributes(G, 'pos')
  labels = nx.get_edge_attributes(G, 'weight')
  nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
  plt.savefig('plots/'+name_val + 'interaction_network.pdf')
  plt.clf()
  table_file.close()

def one_hot_index(nucleotide):
  nucleotide_array = ['A', 'C', 'G', 'T']
  return nucleotide_array.index(nucleotide)

def load_gene_sequence_interaction(sequence_file_name, name_genes_grna_unique,loc1,loc2,nuc1,nuc2):
  # Create numpy matrix of size len(name_genes_grna_unique) * 23, to store the sequence as one-hot encoded
  sequence_pam_per_gene_grna = np.zeros((len(name_genes_grna_unique), 24, 4), dtype = bool)
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
        if  l[2][loc1] == nuc1 and l[2][loc2] == nuc2:
          sequence_pam_per_gene_grna[index_in_name_genes_grna_unique, 23, 0] = 1
          #print "yes!"

  #plot_seq_logo(np.mean(sequence_pam_per_gene_grna, axis=0), "input_spacer")
  # Scikit needs only a 2-d matrix as input, so reshape and return
  return np.reshape(sequence_pam_per_gene_grna, (len(name_genes_grna_unique), -1))


def load_gene_sequence(sequence_file_name, name_genes_grna_unique):
  # Create numpy matrix of size len(name_genes_grna_unique) * 23, to store the sequence as one-hot encoded
  sequence_pam_per_gene_grna = np.zeros((len(name_genes_grna_unique), 23, 4), dtype = bool)
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
  plot_seq_logo(np.mean(sequence_pam_per_gene_grna, axis=0), "input_spacer")
  # Scikit needs only a 2-d matrix as input, so reshape and return
  return np.reshape(sequence_pam_per_gene_grna, (len(name_genes_grna_unique), -1)), np.reshape(sequence_pam_per_gene_grna[:, :20, :], (len(name_genes_grna_unique), -1)), np.reshape(sequence_pam_per_gene_grna[:, 20:, :], (len(name_genes_grna_unique), -1))



def perform_linear_regression(sequence_pam_per_gene_grna, count_insertions_gene_grna_binary, count_deletions_gene_grna_binary, train_index, test_index, ins_coeff, del_coeff, to_plot = False):
  #lin_reg = linear_model.Lasso(alpha=0.001)
  lin_reg = linear_model.Ridge(alpha=100)
  lin_reg.fit(sequence_pam_per_gene_grna[train_index], count_insertions_gene_grna_binary[train_index])
  lin_reg_pred = lin_reg.predict(sequence_pam_per_gene_grna[test_index])
  insertions_r2_score = lin_reg.score(sequence_pam_per_gene_grna[test_index], count_insertions_gene_grna_binary[test_index])
  insertion_rmse = sqrt(mean_squared_error(lin_reg_pred,count_insertions_gene_grna_binary[test_index]))
  ins_coeff.append(lin_reg.coef_)
  insertion_interaction_coefficient = lin_reg.coef_[-4]
  if to_plot:
    plot_QQ(lin_reg_pred,count_insertions_gene_grna_binary[test_index],'QQ_linear_insertion')
    plot_seq_logo(lin_reg.coef_[0:92], "Insertion_linear")


  lin_reg.fit(sequence_pam_per_gene_grna[train_index], count_deletions_gene_grna_binary[train_index])
  lin_reg_pred = lin_reg.predict(sequence_pam_per_gene_grna[test_index])
  deletions_r2_score = lin_reg.score(sequence_pam_per_gene_grna[test_index], count_deletions_gene_grna_binary[test_index])
  deletion_rmse = sqrt(mean_squared_error(lin_reg_pred, count_deletions_gene_grna_binary[test_index]))
  del_coeff.append(lin_reg.coef_)
  deletion_interaction_coefficient = lin_reg.coef_[-4]
  if to_plot:
    plot_QQ(lin_reg_pred, count_deletions_gene_grna_binary[test_index], 'QQ_linear_deletion')
    plot_seq_logo(lin_reg.coef_[0:92], "Deletion_linear")

  return insertions_r2_score, deletions_r2_score, insertion_rmse, deletion_rmse, insertion_interaction_coefficient, deletion_interaction_coefficient


def cross_validation_model(sequence_pam_per_gene_grna, count_insertions_gene_grna, count_deletions_gene_grna,number_of_repeats):
  total_insertion_avg_r2_score = []
  total_deletion_avg_r2_score = []
  total_insertion_avg_rmse = []
  total_deletion_avg_rmse = []
  ins_coeff = []
  del_coeff = []
  insertion_interaction_coefficient_list = []
  deletion_interaction_coefficient_list = []
  for repeat in range(number_of_repeats):
    #print "repeat ", repeat
    number_of_splits = 3
    fold_valid = KFold(n_splits = number_of_splits, shuffle = True, random_state = repeat)

    insertion_avg_r2_score = 0.0
    deletion_avg_r2_score = 0.0
    insertion_avg_rmse = 0.0
    deletion_avg_rmse = 0.0

    count_insertions_gene_grna_binary = np.copy(count_insertions_gene_grna)
    count_deletions_gene_grna_binary = np.copy(count_deletions_gene_grna)

    fold = 0
    for train_index, test_index in fold_valid.split(sequence_pam_per_gene_grna):
      to_plot = False
      if repeat == 1 and fold == 1:
        to_plot = True
      insertions_r2_score, deletions_r2_score, insertion_rmse, deletion_rmse, insertion_interaction_coefficient, deletion_interaction_coefficient = perform_linear_regression(sequence_pam_per_gene_grna, count_insertions_gene_grna_binary, count_deletions_gene_grna_binary, train_index, test_index, ins_coeff, del_coeff, to_plot)

      insertion_avg_r2_score += insertions_r2_score
      deletion_avg_r2_score += deletions_r2_score
      insertion_avg_rmse += insertion_rmse
      deletion_avg_rmse += deletion_rmse

      insertion_interaction_coefficient_list.append(insertion_interaction_coefficient)
      deletion_interaction_coefficient_list.append(deletion_interaction_coefficient)

      fold += 1

    insertion_avg_r2_score /= float(number_of_splits)
    deletion_avg_r2_score /= float(number_of_splits)
    insertion_avg_rmse /= float(number_of_splits)
    deletion_avg_rmse /= float(number_of_splits)

    total_insertion_avg_r2_score.append(float(insertion_avg_r2_score))
    total_deletion_avg_r2_score.append(float(deletion_avg_r2_score))
    total_insertion_avg_rmse.append(float(insertion_avg_rmse))
    total_deletion_avg_rmse.append(float(deletion_avg_rmse))

  return np.mean(total_insertion_avg_r2_score), np.mean(total_deletion_avg_r2_score), np.std(total_insertion_avg_r2_score), np.std(total_deletion_avg_r2_score), np.mean(total_insertion_avg_rmse), np.mean(total_deletion_avg_rmse), np.mean(insertion_interaction_coefficient_list), np.mean(deletion_interaction_coefficient_list)

data_folder = "../IndelsFullData/"
sequence_file_name = "sequence_pam_gene_grna_big_file_donor.csv"
#data_folder = "/Users/amirali/Projects/CRISPR-data/R data/AM_TechMerg_Summary/"
data_folder = "/Users/amirali/Projects/CRISPR-data-Feb18/20nt_counts_only/"

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

number_of_repeats = 100
#count_insertions_gene_grna, count_deletions_gene_grna = compute_summary_statistics(name_genes_grna_unique, name_indel_type_unique, indel_count_matrix, indel_prop_matrix)
prop_insertions_gene_grna, prop_deletions_gene_grna = my_length_finder(indel_count_matrix,length_indel_insertion,length_indel_deletion,consider_length=0)
sequence_pam_per_gene_grna, sequence_per_gene_grna, pam_per_gene_grna = load_gene_sequence(sequence_file_name, name_genes_grna_unique)
r2_insertion_no_interaction, r2_deletion_no_interaction, std_r2_insertion_no_interaction, std_r2_deletion_no_interaction, rmse_insertion_no_interaction, rmse_deletion_no_interaction, ins_coef, del_coef = cross_validation_model(sequence_pam_per_gene_grna, prop_insertions_gene_grna, prop_deletions_gene_grna,number_of_repeats)

print "No Interaction"
print "Insertion r2 and rmse"
print r2_insertion_no_interaction
print rmse_insertion_no_interaction
print "Deletion r2 and rmse"
print r2_deletion_no_interaction
print rmse_deletion_no_interaction

insertion_rmse_list = np.zeros((4*3/2,4,4))
insertion_r2_list = np.zeros((4*3/2,4,4))
insertion_r2_list_std = np.zeros((4*3/2,4,4))
insertion_coefficient_list = np.zeros((4*3/2,4,4))

deletion_rmse_list = np.zeros((4*3/2,4,4))
deletion_r2_list = np.zeros((4*3/2,4,4))
deletion_r2_list_std = np.zeros((4*3/2,4,4))
deletion_coefficient_list = np.zeros((4*3/2,4,4))

p_matrix_insertion = np.zeros((4*3/2,4,4))
p_matrix_deletion = np.zeros((4*3/2,4,4))
count_loc = 0
for loc1 in range(15,19):
    for loc2 in range(loc1+1,19):
        print "loc1,loc2 = ", loc1,loc2
        count_nuc1 = 0
        for nuc1 in ['A', 'C', 'G', 'T']:
            count_nuc2 = 0
            for nuc2 in ['A', 'C', 'G', 'T']:
                #print "nuc1,nuc2" , nuc1,nuc2
                sequence_pam_per_gene_grna = load_gene_sequence_interaction(sequence_file_name, name_genes_grna_unique,loc1,loc2,nuc1,nuc2)
                r2_insertion, r2_deletion, std_r2_insertion, std_r2_deletion, rmse_insertion, rmse_deletion, ins_coef, del_coef = cross_validation_model(sequence_pam_per_gene_grna, prop_insertions_gene_grna, prop_deletions_gene_grna,number_of_repeats)
                #### insertion
                insertion_rmse_list[count_loc,count_nuc1,count_nuc2]=rmse_insertion
                insertion_r2_list[count_loc,count_nuc1,count_nuc2]=r2_insertion
                insertion_r2_list_std[count_loc, count_nuc1, count_nuc2] = std_r2_insertion
                tstat, pvalue = ttest_ind_from_stats(r2_insertion_no_interaction, std_r2_insertion_no_interaction, number_of_repeats,
                                                     r2_insertion, std_r2_insertion, number_of_repeats)
                p_matrix_insertion[count_loc,count_nuc1,count_nuc2] = -np.log10(pvalue)
                insertion_coefficient_list[count_loc,count_nuc1,count_nuc2]=ins_coef

                #### deletion
                deletion_rmse_list[count_loc, count_nuc1, count_nuc2] = rmse_deletion
                deletion_r2_list[count_loc,count_nuc1,count_nuc2]=r2_deletion
                deletion_r2_list_std[count_loc,count_nuc1,count_nuc2]=r2_deletion
                tstat, pvalue = ttest_ind_from_stats(r2_deletion_no_interaction, std_r2_deletion_no_interaction, number_of_repeats,
                                                     r2_deletion, std_r2_deletion, number_of_repeats)
                p_matrix_deletion[count_loc, count_nuc1, count_nuc2] = -np.log10(pvalue)
                deletion_coefficient_list[count_loc,count_nuc1,count_nuc2]=del_coef

                count_nuc2 = count_nuc2 + 1
            count_nuc1 = count_nuc1 + 1
        count_loc = count_loc + 1


            #print "insertion,deletion", accu_insertion,accu_deletion
                #print "----"


plot_interaction_network(p_matrix_insertion,insertion_r2_list,insertion_rmse_list ,r2_insertion_no_interaction,rmse_insertion_no_interaction, insertion_coefficient_list, "pvalue_linreg_individual_insertion")
plot_interaction_network(p_matrix_deletion ,deletion_r2_list,deletion_rmse_list ,r2_deletion_no_interaction,rmse_deletion_no_interaction, deletion_coefficient_list, "pvalue_linreg_individual_deletion")
