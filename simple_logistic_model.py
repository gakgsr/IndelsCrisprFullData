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
from sklearn.metrics import f1_score
import csv
from xgboost import XGBClassifier, XGBRegressor
from scipy.stats import entropy

def entrop_finder(indel_count_matrix):
    num_indels, num_sites = np.shape(indel_count_matrix)
    indel_fraction_mutant_matrix = indel_count_matrix / np.reshape(np.sum(indel_count_matrix, axis=0), (1, -1))
    entrop = []
    for col in range(num_sites):
        vec = np.copy(indel_fraction_mutant_matrix[:, col])
        vec = np.sort(vec)[::-1]
        entrop.append(entropy(vec))

    return np.asarray(entrop)

def fraction_of_deletion_insertion(indel_count_matrix,length_indel_insertion,length_indel_deletion):
  indel_num,site_num = np.shape(indel_count_matrix)

  prop_insertions_gene_grna = np.zeros(site_num,dtype=float)
  prop_deletions_gene_grna = np.zeros(site_num,dtype=float)


  insertion_indicator = np.copy(length_indel_insertion)
  deletion_indicator = np.copy(length_indel_deletion)

  insertion_indicator[insertion_indicator>0]=1.
  deletion_indicator[deletion_indicator>0]=1.

  indel_fraction_mutant_matrix = indel_count_matrix / np.reshape(np.sum(indel_count_matrix, axis=0), (1, -1))

  for site_index in range(site_num):
    prop_insertions_gene_grna[site_index] = np.inner(insertion_indicator,indel_fraction_mutant_matrix[:,site_index])
    prop_deletions_gene_grna[site_index] = np.inner(deletion_indicator, indel_fraction_mutant_matrix[:, site_index])

  return prop_insertions_gene_grna,prop_deletions_gene_grna

def expected_deletion_insertion_length(indel_count_matrix,length_indel_insertion,length_indel_deletion):
  indel_num,site_num = np.shape(indel_count_matrix)

  exp_insertion_length = np.zeros(site_num,dtype=float)
  exp_deletion_length = np.zeros(site_num,dtype=float)

  insertion_indicator = np.copy(length_indel_insertion)
  deletion_indicator = np.copy(length_indel_deletion)

  insertion_indicator[insertion_indicator>0]=1.
  deletion_indicator[deletion_indicator>0]=1.

  indel_fraction_mutant_matrix = indel_count_matrix / np.reshape(np.sum(indel_count_matrix, axis=0), (1, -1))

  insertion_only_fraction_matrix =  np.multiply(indel_fraction_mutant_matrix, np.reshape(insertion_indicator,(-1,1)) )
  deletion_only_fraction_matrix = np.multiply(indel_fraction_mutant_matrix,  np.reshape(deletion_indicator,(-1,1)) )

  insertion_only_fraction_matrix = insertion_only_fraction_matrix / np.reshape(np.sum(insertion_only_fraction_matrix, axis=0), (1, -1))
  deletion_only_fraction_matrix = deletion_only_fraction_matrix / np.reshape(np.sum(deletion_only_fraction_matrix, axis=0), (1, -1))


  for site_index in range(site_num):
    exp_insertion_length[site_index] = np.inner(length_indel_insertion,insertion_only_fraction_matrix[:,site_index])
    exp_deletion_length[site_index] = np.inner(length_indel_deletion, deletion_only_fraction_matrix[:, site_index])

  # some sites do not get any insertions. this cuase nan. we make those entries zero.
  for i in range(np.size(exp_insertion_length)):
    if np.isnan(exp_insertion_length[i]):
      exp_insertion_length[i] = 0

  return exp_insertion_length,exp_deletion_length


def top_indel_finder(indel_count_matrix,name_indel_type_unique):
    indel_num,site_num = np.shape(indel_count_matrix)
    top_indel_type_vector = np.zeros(site_num)
    for site in range(site_num):
        if 'I' in name_indel_type_unique[np.argmax(indel_count_matrix[:,site])]:
            top_indel_type_vector[site] = 1
    return top_indel_type_vector


def jaccard_distance(set1,set2):
  return 1 - float(len(list(set(set1) & set(set2)))) / len(list(set(set1) | set(set2)))

def variation_patients_and_lump(indel_count_matrix,sequence_file_name, name_genes_grna_unique):
  topk = 15
  num_indel, num_crispr = np.shape(indel_count_matrix)
  indel_set_matrix = np.zeros((topk,num_crispr))
  for crispr in range(num_crispr):
    indel_set_matrix[:,crispr] = np.argsort(indel_count_matrix[:, crispr])[-topk:]

  all_sites = []
  map2 = {}
  with open(sequence_file_name) as f:
    for line in f:
      line = line.replace('"', '')
      line = line.replace(' ', '')
      line = line.replace('\n', '')
      l = line.split(',')
      all_sites.append(l[4])
      map2[l[0]] = l[4]
  # ********
  all_sites = list(set(all_sites))
  site_map = np.zeros(len(name_genes_grna_unique)) - 1
  for i in range(len(name_genes_grna_unique)):
    gene_grna_name = name_genes_grna_unique[i].split('-')
    if map2.get(gene_grna_name[1] + '-' + gene_grna_name[2]) != None:
      site_map[i] = all_sites.index(map2[gene_grna_name[1] + '-' + gene_grna_name[2]])
    else:
      print "Some keys missing"



  howmanytoshow = num_crispr
  #howmanytoshow = 100

  indel_set_matrix = indel_set_matrix[:,np.asarray(np.argsort(site_map))]
  site_counter_list_Counter = collections.Counter(np.asarray(np.sort(site_map)))
  site_counter_list = site_counter_list_Counter.values()

  name_genes_grna_unique_sorted = np.copy(name_genes_grna_unique)
  name_genes_grna_unique_sorted = name_genes_grna_unique_sorted[np.asarray(np.argsort(site_map))]

  #jaccard_matrix = np.zeros((num_crispr,num_crispr))
  jaccard_matrix = np.zeros((howmanytoshow, howmanytoshow))
  for crispr1 in range(howmanytoshow):
    for crispr2 in range(howmanytoshow):
      jaccard_matrix[crispr1,crispr2] = jaccard_distance(indel_set_matrix[:,crispr1],indel_set_matrix[:,crispr2])

  unique_patient_per_site_index_list = []
  indel_count_matrix_sum = indel_count_matrix.sum(axis=0)
  for site_counter in range(np.size(site_counter_list)):
    indexxx = np.argmax(indel_count_matrix_sum[range(sum(site_counter_list[0:site_counter]),sum(site_counter_list[0:site_counter])+site_counter_list[site_counter])])
    unique_patient_per_site_index_list.append(indexxx+sum(site_counter_list[0:site_counter]))

  #af = AffinityPropagation().fit(jaccard_matrix)
  #cluster_centers_indices = af.cluster_centers_indices_
  #labels = af.labels_
  #n_clusters_ = len(cluster_centers_indices)

  kmeans = KMeans(n_clusters=np.size(site_counter_list), init='k-means++', random_state=0).fit(jaccard_matrix)
  labels = kmeans.labels_
  ARI = adjusted_rand_score(list(labels), list(np.sort(site_map)))
  print "ARI = ", ARI




  # # this is to find the inner and outer distance variations
  # iner_distances = []
  # all_distances = []
  # for site_counter in range(np.size(site_counter_list)):
  #   for i in range(site_counter_list[site_counter]):
  #     for j in range(i+1,site_counter_list[site_counter]):
  #       new_jaccard = jaccard_matrix[sum(site_counter_list[0:site_counter])+i ,sum(site_counter_list[0:site_counter])+j]
  #       iner_distances.append(new_jaccard)
  #       #if new_jaccard > 0.8:
  #       #  print name_genes_grna_unique_sorted[sum(site_counter_list[0:site_counter])+i],name_genes_grna_unique_sorted[sum(site_counter_list[0:site_counter])+j]
  # print "inner jaccard distance mean", np.mean(iner_distances)
  # print "inner jaccard distance std", np.std(iner_distances)
  # for i in range(num_crispr):
  #   for j in range(i+1,num_crispr):
  #     all_distances.append(jaccard_matrix[i,j])
  # print "all jaccard distance mean", np.mean(all_distances)
  # print "all jaccard distance std", np.std(all_distances)


  n, bins, patches = plt.hist(site_counter_list, facecolor='green')
  plt.xlabel('Number of Patients in Sites')
  plt.ylabel('Count')
  plt.title('Histogram')
  plt.savefig('histogram_patient_site.pdf')
  plt.clf()

  print "n = " , n
  print "bins = ", bins
  print "patches = ", patches

  print 'max number of pat per site', np.max(site_counter_list)
  print 'total number of crispr outcomes', np.sum(site_counter_list)

  # # this is to plot the jaccard distance matrix
  # plt.imshow(jaccard_matrix, cmap='hot', interpolation='nearest')
  # plt.colorbar()
  # ax = plt.gca()
  # ax.set_xticklabels([])
  # ax.set_yticklabels([])
  # #ax.set_xticks(np.arange(0, howmanytoshow, 1))
  # #ax.set_yticks(np.arange(0, howmanytoshow, 1))
  # #ax.set_xticklabels(map(int, np.sort(site_map)[0:howmanytoshow]))
  # #ax.set_yticklabels(map(int, np.sort(site_map)[0:howmanytoshow]))
  # #ax.set_xticklabels(itemgetter(*map(int, np.sort(site_map)[0:howmanytoshow]))(all_sites))
  # #ax.set_yticklabels(itemgetter(*map(int, np.sort(site_map)[0:howmanytoshow]))(all_sites))
  # plt.ylabel('Cut-site Index')
  # plt.xlabel('Cut-site Index')
  # plt.savefig('jaccard.pdf')
  # plt.clf()

  return indel_set_matrix,jaccard_matrix,unique_patient_per_site_index_list


def plot_interaction_network(adj_list, name_val):
  adj_list = adj_list.reshape([190, 4, 4])
  G = nx.Graph()
  num_edges = 15
  adj_sorted = np.sort(np.abs(adj_list), axis = None)
  min_wt = adj_sorted[-num_edges]
  print "min weight",name_val, min_wt
  nucleotide_array = ['A', 'C', 'G', 'T']
  ij_counter = 0
  for i1 in range(20):
    for i2 in range(i1 + 1, 20):
      for i3 in range(4):
        for i4 in range(4):
          if(np.abs(adj_list[ij_counter, i3, i4]) >= min_wt and adj_list[ij_counter, i3, i4] > 0):
            print adj_list[ij_counter, i3, i4]
            G.add_edge(str(i1+1) + nucleotide_array[i3], str(i2+1) + nucleotide_array[i4], w = np.abs(adj_list[ij_counter, i3, i4]), c = 'b')
          if(np.abs(adj_list[ij_counter, i3, i4]) >= min_wt and adj_list[ij_counter, i3, i4] <= 0):
            G.add_edge(str(i1+1) + nucleotide_array[i3], str(i2+1) + nucleotide_array[i4], w = np.abs(adj_list[ij_counter, i3, i4]), c = 'g')
            print adj_list[ij_counter, i3, i4]
      ij_counter += 1

  plt.figure()
  edges = G.edges()
  colors = [G[u][v]['c'] for u,v in edges]
  weights = [G[u][v]['w'] for u,v in edges]
  #nx.draw(G, nx.circular_layout(G), font_size = 10, node_color = 'y', with_labels=True, edges=edges, edge_color=colors, width=weights)
  nx.draw(G, nx.circular_layout(G), font_size=10, node_color='y', with_labels=True)
  plt.savefig(name_val + 'interaction_network.pdf')
  plt.clf()


def one_hot_index(nucleotide):
  if nucleotide == 'g':
    nucleotide = 'G'
  elif nucleotide == 'a':
    nucleotide = 'A'
  elif nucleotide == 'c':
    nucleotide = 'C'
  elif nucleotide == 't':
    nucleotide = 'T'
  nucleotide_array = ['A', 'C', 'G', 'T']
  return nucleotide_array.index(nucleotide)


def load_gene_sequence(sequence_file_name, name_genes_grna_unique,homopolymer_matrix):
  # Create numpy matrix of size len(name_genes_grna_unique) * 23, to store the sequence as one-hot encoded
  sequence_pam_per_gene_grna = np.zeros((len(name_genes_grna_unique), 23, 4), dtype = bool)
  sequence_pam_homop_per_gene_grna = np.zeros((len(name_genes_grna_unique), 24, 4))
  sequence_genom_context_gene_grna = np.zeros((len(name_genes_grna_unique), 100, 4), dtype=bool)
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
          sequence_pam_homop_per_gene_grna[index_in_name_genes_grna_unique, i, one_hot_index(l[2][i])] = 1
        for i in range(3):
          sequence_pam_per_gene_grna[index_in_name_genes_grna_unique, 20 + i, one_hot_index(l[3][i])] = 1
          sequence_pam_homop_per_gene_grna[index_in_name_genes_grna_unique, 20 + i, one_hot_index(l[3][i])] = 1

        sequence_pam_homop_per_gene_grna[index_in_name_genes_grna_unique, 23 , :] = homopolymer_matrix[:,index_in_name_genes_grna_unique]

        for i in range(100):
          sequence_genom_context_gene_grna[index_in_name_genes_grna_unique, i, one_hot_index(l[6][i])] = 1

  plot_seq_logo(np.mean(sequence_pam_per_gene_grna, axis=0), "input_spacer")
  # Scikit needs only a 2-d matrix as input, so reshape and return
  return np.reshape(sequence_genom_context_gene_grna, (len(sequence_genom_context_gene_grna), -1)), np.reshape(sequence_pam_homop_per_gene_grna, (len(sequence_pam_homop_per_gene_grna), -1)),np.reshape(sequence_pam_per_gene_grna, (len(name_genes_grna_unique), -1)), np.reshape(sequence_pam_per_gene_grna[:, :20, :], (len(name_genes_grna_unique), -1)), np.reshape(sequence_pam_per_gene_grna[:, 20:, :], (len(name_genes_grna_unique), -1))


def load_gene_sequence_interaction(sequence_file_name, name_genes_grna_unique):
  # Create numpy matrix of size len(name_genes_grna_unique) * 23, to store the sequence as one-hot encoded
  sequence_pam_per_gene_grna = np.zeros((len(name_genes_grna_unique), 23, 4), dtype = bool)
  sequence_pam_per_gene_grna_interaction = np.zeros((len(name_genes_grna_unique), 20*(20-1)/2, 4, 4), dtype=bool) # 15 to 20
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
        # interaction term
        ij_counter = 0
        for i in range(20):
          for j in range(i+1,20):
            sequence_pam_per_gene_grna_interaction[index_in_name_genes_grna_unique,ij_counter, one_hot_index(l[2][i]),one_hot_index(l[2][j])] = 1
            ij_counter = ij_counter + 1

  # Scikit needs only a 2-d matrix as input, so reshape and return
  #return np.reshape(sequence_pam_per_gene_grna, (len(name_genes_grna_unique), -1)), np.reshape(sequence_pam_per_gene_grna[:, :20, :], (len(name_genes_grna_unique), -1)), np.reshape(sequence_pam_per_gene_grna[:, 20:, :], (len(name_genes_grna_unique), -1))
  #print np.shape(np.reshape(sequence_pam_per_gene_grna, (len(name_genes_grna_unique), -1)))
  #print np.shape(np.reshape(sequence_pam_per_gene_grna_interaction, (len(name_genes_grna_unique), -1)))

  ### plot the input logo
  plot_seq_logo(np.mean(sequence_pam_per_gene_grna, axis=0), "input_spacer")

  ### linear + interaction
  return np.concatenate((np.reshape(sequence_pam_per_gene_grna, (len(name_genes_grna_unique), -1)),np.reshape(sequence_pam_per_gene_grna_interaction, (len(name_genes_grna_unique), -1))),axis=1)
  ### interaction
  # return np.reshape(sequence_pam_per_gene_grna_interaction, (len(name_genes_grna_unique), -1))
  ### linear
  # return np.reshape(sequence_pam_per_gene_grna, (len(name_genes_grna_unique), -1))

def load_gene_sequence_k_mer(sequence_file_name, name_genes_grna_unique, k):
  # Create numpy matrix of size len(name_genes_grna_unique) * 23, to store the sequence first
  sequence_pam_per_gene_grna = np.zeros((len(name_genes_grna_unique), 23), dtype = int)
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
          sequence_pam_per_gene_grna[index_in_name_genes_grna_unique, i] = one_hot_index(l[2][i])
        for i in range(3):
          sequence_pam_per_gene_grna[index_in_name_genes_grna_unique, 20 + i] = one_hot_index(l[3][i])
  # Compute k_mers
  k_mer_list = np.zeros((len(name_genes_grna_unique), 4**k), dtype = int)
  for i in range(len(name_genes_grna_unique)):
    for j in range(23 - k + 1):
      k_mer = 0
      for r in range(k):
        k_mer += sequence_pam_per_gene_grna[i][j + r]*(4**(k - r - 1))
      k_mer_list[i, k_mer] += 1
  return k_mer_list


def perform_logistic_regression(sequence_pam_per_gene_grna, count_insertions_gene_grna_binary, count_deletions_gene_grna_binary, train_index, test_index, ins_coeff, del_coeff,log_reg, to_plot = False):
  #print "----"
  #print "Number of positive testing samples in insertions is %f" % np.sum(count_insertions_gene_grna_binary[test_index])
  #print "Total number of testing samples %f" % np.size(test_index)
  #print "Number of positive training samples in insertions is %f" % np.sum(count_insertions_gene_grna_binary[train_index])
  #print "Total number of training samples %f" % np.size(train_index)
  log_reg.fit(sequence_pam_per_gene_grna[train_index], count_insertions_gene_grna_binary[train_index])
  log_reg_pred = log_reg.predict(sequence_pam_per_gene_grna[test_index])
  log_reg_pred_train = log_reg.predict(sequence_pam_per_gene_grna[train_index])
  insertions_accuracy = metrics.accuracy_score(count_insertions_gene_grna_binary[test_index], log_reg_pred)
  insertions_f1 = f1_score(count_insertions_gene_grna_binary[test_index], log_reg_pred)
  # ins_coeff.append(log_reg.coef_[0, :])
  # if to_plot:
  #   #plt.plot(log_reg.coef_[0, 0:92])
  #   #plt.savefig('ins_log_coeff.pdf')
  #   #plt.clf()
  #   #pvalue_vec = f_regression(sequence_pam_per_gene_grna[test_index], log_reg_pred)[1]
  #   scores, pvalue_vec = chi2(sequence_pam_per_gene_grna[test_index], log_reg_pred)
  #   plot_seq_logo(-np.log10(pvalue_vec), "Insertion_logistic_pvalue")
  #   print -np.log10(pvalue_vec)[-4:]
  #   plot_seq_logo(log_reg.coef_[0, :], "Insertion_logistic")

    #plot_interaction_network(log_reg.coef_[0, 92:], "Insertion_logistic")

  #print "Test accuracy score for insertions: %f" % insertions_accuracy
  #print "Train accuracy score for insertions: %f" % metrics.accuracy_score(count_insertions_gene_grna_binary[train_index], log_reg_pred_train)
  #print "----"
  #print "Number of positive testing samples in deletions is %f" % np.sum(count_deletions_gene_grna_binary[test_index])
  #print "Total number of testing samples %f" % np.size(test_index)
  #print "Number of positive training samples in deletions is %f" % np.sum(count_deletions_gene_grna_binary[train_index])
  #print "Total number of training samples %f" % np.size(train_index)
  log_reg.fit(sequence_pam_per_gene_grna[train_index], count_deletions_gene_grna_binary[train_index])
  log_reg_pred = log_reg.predict(sequence_pam_per_gene_grna[test_index])
  log_reg_pred_train = log_reg.predict(sequence_pam_per_gene_grna[train_index])
  deletions_accuracy = metrics.accuracy_score(count_deletions_gene_grna_binary[test_index], log_reg_pred)
  deletions_f1 = f1_score(count_deletions_gene_grna_binary[test_index], log_reg_pred)
  # del_coeff.append(log_reg.coef_[0, :])
  # if to_plot:
  #   #plt.plot(log_reg.coef_[0, 0:92])
  #   #plt.savefig('del_log_coeff.pdf')
  #   #plt.clf()
  #   #pvalue_vec = f_regression(sequence_pam_per_gene_grna[test_index], log_reg_pred)[1]
  #   scores, pvalue_vec = chi2(sequence_pam_per_gene_grna[test_index], log_reg_pred)
  #   plot_seq_logo(-np.log10(pvalue_vec), "Deletion_logistic_pvalue")
  #   plot_seq_logo(log_reg.coef_[0, :], "Deletion_logistic")
  #   print -np.log10(pvalue_vec)[-4:]
    #plot_interaction_network(log_reg.coef_[0, 92:], "Deletion_logistic")
  #print log_reg_pred
  #print "Test accuracy score for deletions: %f" % deletions_accuracy
  #print "Train accuracy score for deletions: %f" % metrics.accuracy_score(count_deletions_gene_grna_binary[train_index], log_reg_pred_train)
  return insertions_accuracy, deletions_accuracy, insertions_f1, deletions_f1


def cross_validation_model(sequence_pam_per_gene_grna, count_insertions_gene_grna, count_deletions_gene_grna,log_reg):
  total_insertion_avg_accuracy = []
  total_deletion_avg_accuracy = []
  total_insertion_avg_f1 = []
  total_deletion_avg_f1 = []
  ins_coeff = []
  del_coeff = []
  for repeat in range(10):
    print "repeat ", repeat
    number_of_splits = 5
    fold_valid = KFold(n_splits = number_of_splits, shuffle = True, random_state = repeat)
    insertion_avg_accuracy = 0.0
    deletion_avg_accuracy = 0.0
    insertion_avg_f1 = 0.0
    deletion_avg_f1 = 0.0

    # threshold_insertions = 1
    count_insertions_gene_grna_binary = np.copy(count_insertions_gene_grna)
    # count_insertions_gene_grna_binary[count_insertions_gene_grna >= threshold_insertions] = 1
    # count_insertions_gene_grna_binary[count_insertions_gene_grna < threshold_insertions] = 0
    # threshold_deletions = 1
    count_deletions_gene_grna_binary = np.copy(count_deletions_gene_grna)
    # count_deletions_gene_grna_binary[count_deletions_gene_grna >= threshold_deletions] = 1
    # count_deletions_gene_grna_binary[count_deletions_gene_grna < threshold_deletions] = 0
    fold = 0
    for train_index, test_index in fold_valid.split(sequence_pam_per_gene_grna):
      to_plot = False
      if repeat == 1 and fold == 1:
        to_plot = True
      accuracy_score = perform_logistic_regression(sequence_pam_per_gene_grna, count_insertions_gene_grna_binary, count_deletions_gene_grna_binary, train_index, test_index, ins_coeff, del_coeff,log_reg, to_plot)
      insertion_avg_accuracy += accuracy_score[0]
      deletion_avg_accuracy += accuracy_score[1]
      insertion_avg_f1 += accuracy_score[2]
      deletion_avg_f1 += accuracy_score[3]
      fold += 1
    insertion_avg_accuracy /= float(number_of_splits)
    deletion_avg_accuracy /= float(number_of_splits)
    insertion_avg_f1 /= float(number_of_splits)
    deletion_avg_f1 /= float(number_of_splits)
    total_insertion_avg_accuracy.append(insertion_avg_accuracy)
    total_deletion_avg_accuracy.append(deletion_avg_accuracy)
    total_insertion_avg_f1.append(insertion_avg_f1)
    total_deletion_avg_f1.append(deletion_avg_f1)

  print "--"
  print "Average accuracy for insertions predictions is %f" % np.mean(total_insertion_avg_accuracy)
  print "Std in accuracy for insertions predictions is %f" % np.std(total_insertion_avg_accuracy)
  print "Average accuracy for deletions predictions is %f" % np.mean(total_deletion_avg_accuracy)
  print "Std in accuracy for deletions predictions is %f" % np.std(total_deletion_avg_accuracy)
  print "--"
  print "Average f1 for insertions predictions is %f" % np.mean(total_insertion_avg_f1)
  print "Std in f1 for insertions predictions is %f" % np.std(total_insertion_avg_f1)
  print "Average f1 for deletions predictions is %f" % np.mean(total_deletion_avg_f1)
  print "Std in f1 for deletions predictions is %f" % np.std(total_deletion_avg_f1)
  print "--"


  ins_coeff = np.array(ins_coeff)
  del_coeff = np.array(del_coeff)
  #print "Average coefficients for insertions predictions is "
  #print np.mean(ins_coeff, axis = 0)
  print "Max std in coefficients for insertions predictions is "
  print np.max(np.std(ins_coeff, axis = 0))
  #print "Average coefficients for deletions predictions is "
  #print np.mean(del_coeff, axis = 0)
  print "Max std in coefficients for deletions predictions is "
  print np.max(np.std(del_coeff, axis = 0))


data_folder = "../IndelsFullData/"
sequence_file_name = "sequence_pam_gene_grna_big_file_donor_genomic_context.csv"
#data_folder = "/Users/amirali/Projects/CRISPR-data/R data/AM_TechMerg_Summary/"
data_folder = "/Users/amirali/Projects/CRISPR-data-Feb18/20nt_counts_only/"

name_genes_grna_unique = pickle.load(open('Tcell-files/name_genes_grna_UNIQUE.p', 'rb'))
name_indel_type_unique = pickle.load(open('Tcell-files/name_indel_type_ALL.p', 'rb'))
indel_count_matrix = pickle.load(open('Tcell-files/indel_count_matrix_UNIQUE.p', 'rb'))
indel_prop_matrix = pickle.load(open('Tcell-files/indel_prop_matrix_UNIQUE.p', 'rb'))
length_indel_insertion = pickle.load(open('Tcell-files/length_indel_insertion_ALL.p', 'rb'))
length_indel_deletion = pickle.load(open('Tcell-files/length_indel_deletion_ALL.p', 'rb'))
homopolymer_matrix = pickle.load(open('Tcell-files/homology_matrix_UNIQUE.p', 'rb'))
my_eff_vec = pickle.load(open('Tcell-files/my_eff_vec_UNIQUE_no_others.p', 'rb'))
my_eff_vec = np.asarray(my_eff_vec)

#indel_set_matrix,jaccard_matrix,unique_patient_per_site_index_list = variation_patients_and_lump(indel_count_matrix,sequence_file_name, name_genes_grna_unique)

fraction_insertions, fraction_deletions = fraction_of_deletion_insertion(indel_count_matrix,length_indel_insertion,length_indel_deletion)
exp_insertion_length, exp_deletion_length = expected_deletion_insertion_length(indel_count_matrix,length_indel_insertion,length_indel_deletion)
count_insertions_gene_grna, count_deletions_gene_grna = compute_summary_statistics(name_genes_grna_unique, name_indel_type_unique, indel_count_matrix, indel_prop_matrix)
sequence_genom_context_gene_grna,sequence_pam_homop_per_gene_grna,sequence_pam_per_gene_grna, sequence_per_gene_grna, pam_per_gene_grna = load_gene_sequence(sequence_file_name, name_genes_grna_unique,homopolymer_matrix)
sequence_pam_per_gene_grna = load_gene_sequence_interaction(sequence_file_name, name_genes_grna_unique)
top_indel_vector = top_indel_finder(indel_count_matrix,name_indel_type_unique)
entrop = entrop_finder(indel_count_matrix)


#log_reg = linear_model.LogisticRegression(penalty='l2', C=1000)
log_reg = XGBClassifier(n_estimators=30, max_depth=1)

#print "Using all genomic context"
#cross_validation_model(sequence_genom_context_gene_grna, count_insertions_gene_grna, count_deletions_gene_grna)
#cross_validation_model(sequence_genom_context_gene_grna, top_indel_vector, top_indel_vector)
#print "Using both Spacer and PAM"
#cross_validation_model(sequence_pam_homop_per_gene_grna, top_indel_vector, top_indel_vector)
#cross_validation_model(sequence_pam_homop_per_gene_grna, top_indel_vector, top_indel_vector)
#print np.shape(sequence_pam_per_gene_grna)
#print count_insertions_gene_grna
#cross_validation_model(sequence_pam_homop_per_gene_grna, count_insertions_gene_grna, count_deletions_gene_grna)
#print "Using only Spacer"
#cross_validation_model(sequence_per_gene_grna, count_insertions_gene_grna, count_deletions_gene_grna)
#print "Using only PAM"
#cross_validation_model(pam_per_gene_grna, count_insertions_gene_grna, count_deletions_gene_grna)
#print "Using Kmers of Spacer + PAM"
#k_mer_list = load_gene_sequence_k_mer(sequence_file_name, name_genes_grna_unique, 3)
#cross_validation_model(k_mer_list, count_insertions_gene_grna, count_deletions_gene_grna)

my_eff_vec_binary_tcel = np.zeros(len(name_genes_grna_unique))
my_eff_vec_binary_tcel[my_eff_vec>np.median(my_eff_vec)]=1

# entropy_binary = np.zeros(len(name_genes_grna_unique))
# entropy_binary[entrop>np.median(entrop)] = 1

print "number of + samples in insertions is ", np.sum(my_eff_vec_binary_tcel)
print "number of - samples in insertions is ", np.shape(name_genes_grna_unique)[0] - np.sum(my_eff_vec_binary_tcel)


cross_validation_model(sequence_pam_per_gene_grna, my_eff_vec_binary_tcel, my_eff_vec_binary_tcel,log_reg)





#### here we run insertion accuracy
#
# exp_insertion_length_binary = np.zeros(len(name_genes_grna_unique))
#
# okay_insertion_index = list(set(np.where(exp_insertion_length>0)[0]))
# ###
# #okay_insertion_index = list(set(np.where(fraction_insertions>0.1)[0]))
# ###
# print "insertion mean = ", np.mean(exp_insertion_length[okay_insertion_index])
# print "insertion median =", np.median(exp_insertion_length[okay_insertion_index])
#
# exp_insertion_length_binary[exp_insertion_length>np.median(exp_insertion_length[okay_insertion_index])] = 1
#
#
#
# exp_insertion_length_binary = exp_insertion_length_binary[okay_insertion_index]
# sequence_pam_per_gene_grna = sequence_pam_per_gene_grna[okay_insertion_index,:]
#
# print np.shape(exp_insertion_length_binary)
# print np.shape(sequence_pam_per_gene_grna)
#
# print "number of + samples in insertions is ", np.sum(exp_insertion_length_binary)
# print "number of - samples in insertions is ", np.shape(exp_insertion_length_binary)[0] - np.sum(exp_insertion_length_binary)
# cross_validation_model(sequence_pam_per_gene_grna, exp_insertion_length_binary, exp_insertion_length_binary,log_reg)


# exp_deletion_length_binary = np.zeros(len(name_genes_grna_unique))
# print "deletion mean = ", np.mean(exp_deletion_length)
# print "deletion median =", np.median(exp_deletion_length)
# exp_deletion_length_binary[exp_deletion_length> np.median(exp_deletion_length)] = 1
# cross_validation_model(sequence_pam_per_gene_grna, exp_deletion_length_binary, exp_deletion_length_binary,log_reg)


########

#
# okay_insertion_index = list(set(np.where(fraction_deletions >0)[0]))
# toplot = np.copy(exp_deletion_length)
# toplot = toplot[okay_insertion_index]
# plt.hist(toplot, bins=100)
# plt.savefig('plots/Expected_deletion_length_hist.pdf')
# plt.clf()
#
# print "deletion mean = ", np.mean(exp_deletion_length)
# print "deletion median =", np.median(exp_deletion_length)
#
# # plt.hist(exp_deletion_length)
# # plt.savefig('plots/Expected_deletion_length_hist.pdf')
# # plt.clf()