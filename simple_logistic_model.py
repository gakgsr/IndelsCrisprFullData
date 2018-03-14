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


def jaccard_distance(set1,set2):
  return 1 - len(list(set(set1) & set(set2))) / len(list(set(set1) | set(set2)))

def variation_patients_and_lump(indel_count_matrix,sequence_file_name, name_genes_grna_unique):
  topk = 20
  num_indel, num_crispr = np.shape(indel_count_matrix)
  indel_set_matrix = np.zeros((topk,num_crispr))
  for crispr in range(num_crispr):
    indel_set_matrix[:,crispr] = np.argsort(indel_count_matrix[:, crispr])[-topk:]

  map1 = {}
  for i in range(len(name_genes_grna_unique)):
    gene_grna_name = name_genes_grna_unique[i].split('-')
    map1['gene_grna_name'] = i

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


  #print np.sort(all_sites)
  #print np.asarray(np.argsort(all_sites))

  indel_set_matrix = indel_set_matrix[:,np.asarray(np.argsort(site_map))]

  #jaccard_matrix = np.zeros((num_crispr,num_crispr))
  jaccard_matrix = np.zeros((100, 100))
  for crispr1 in range(100):
    for crispr2 in range(100):
      jaccard_matrix[crispr1,crispr2] = jaccard_distance(indel_set_matrix[:,crispr1],indel_set_matrix[:,crispr2])

  return indel_set_matrix,jaccard_matrix


def plot_interaction_network(adj_list, name_val):
  adj_list = adj_list.reshape([190, 4, 4])
  G = nx.Graph()
  num_edges = 10
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
            G.add_edge(str(i1+1) + nucleotide_array[i3], str(i2+1) + nucleotide_array[i4], w = np.abs(adj_list[ij_counter, i3, i4]), c = 'b')
          if(np.abs(adj_list[ij_counter, i3, i4]) >= min_wt and adj_list[ij_counter, i3, i4] <= 0):
            G.add_edge(str(i1+1) + nucleotide_array[i3], str(i2+1) + nucleotide_array[i4], w = np.abs(adj_list[ij_counter, i3, i4]), c = 'g')
      ij_counter += 1

  plt.figure()
  edges = G.edges()
  colors = [G[u][v]['c'] for u,v in edges]
  weights = [G[u][v]['w'] for u,v in edges]
  nx.draw(G, nx.circular_layout(G), font_size = 10, node_color = 'y', with_labels=True, edges=edges, edge_color=colors, width=weights)
  plt.savefig(name_val + 'interaction_network.pdf')
  plt.clf()


def one_hot_index(nucleotide):
  nucleotide_array = ['A', 'C', 'G', 'T']
  return nucleotide_array.index(nucleotide)


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


def perform_logistic_regression(sequence_pam_per_gene_grna, count_insertions_gene_grna_binary, count_deletions_gene_grna_binary, train_index, test_index, ins_coeff, del_coeff, to_plot = False):
  log_reg = linear_model.LogisticRegression(penalty='l2', C=10)
  #print "----"
  #print "Number of positive testing samples in insertions is %f" % np.sum(count_insertions_gene_grna_binary[test_index])
  #print "Total number of testing samples %f" % np.size(test_index)
  #print "Number of positive training samples in insertions is %f" % np.sum(count_insertions_gene_grna_binary[train_index])
  #print "Total number of training samples %f" % np.size(train_index)
  log_reg.fit(sequence_pam_per_gene_grna[train_index], count_insertions_gene_grna_binary[train_index])
  log_reg_pred = log_reg.predict(sequence_pam_per_gene_grna[test_index])
  log_reg_pred_train = log_reg.predict(sequence_pam_per_gene_grna[train_index])
  insertions_accuracy = metrics.accuracy_score(count_insertions_gene_grna_binary[test_index], log_reg_pred)
  ins_coeff.append(log_reg.coef_[0, :])
  if to_plot:
    plt.plot(log_reg.coef_[0, 0:92])
    plt.savefig('ins_log_coeff.pdf')
    plt.clf()
    plot_seq_logo(log_reg.coef_[0, 0:92], "Insertion_logistic")
    plot_interaction_network(log_reg.coef_[0, 92:], "Insertion_logistic")

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
  del_coeff.append(log_reg.coef_[0, :])
  if to_plot:
    plt.plot(log_reg.coef_[0, 0:92])
    plt.savefig('del_log_coeff.pdf')
    plt.clf()
    plot_seq_logo(log_reg.coef_[0, 0:92], "Deletion_logistic")
    plot_interaction_network(log_reg.coef_[0, 92:], "Deletion_logistic")
  #print log_reg_pred
  #print "Test accuracy score for deletions: %f" % deletions_accuracy
  #print "Train accuracy score for deletions: %f" % metrics.accuracy_score(count_deletions_gene_grna_binary[train_index], log_reg_pred_train)
  return insertions_accuracy, deletions_accuracy


def cross_validation_model(sequence_pam_per_gene_grna, count_insertions_gene_grna, count_deletions_gene_grna):
  total_insertion_avg_accuracy = []
  total_deletion_avg_accuracy = []
  ins_coeff = []
  del_coeff = []
  for repeat in range(5):
    #print "repeat ", repeat
    number_of_splits = 3
    fold_valid = KFold(n_splits = number_of_splits, shuffle = True, random_state = repeat)
    insertion_avg_accuracy = 0.0
    deletion_avg_accuracy = 0.0
    threshold_insertions = 1
    count_insertions_gene_grna_binary = np.copy(count_insertions_gene_grna)
    count_insertions_gene_grna_binary[count_insertions_gene_grna >= threshold_insertions] = 1
    count_insertions_gene_grna_binary[count_insertions_gene_grna < threshold_insertions] = 0
    threshold_deletions = 1
    count_deletions_gene_grna_binary = np.copy(count_deletions_gene_grna)
    count_deletions_gene_grna_binary[count_deletions_gene_grna >= threshold_deletions] = 1
    count_deletions_gene_grna_binary[count_deletions_gene_grna < threshold_deletions] = 0
    fold = 0
    for train_index, test_index in fold_valid.split(sequence_pam_per_gene_grna):
      to_plot = False
      if repeat == 1 and fold == 1:
        to_plot = True
      accuracy_score = perform_logistic_regression(sequence_pam_per_gene_grna, count_insertions_gene_grna_binary, count_deletions_gene_grna_binary, train_index, test_index, ins_coeff, del_coeff, to_plot)
      insertion_avg_accuracy += accuracy_score[0]
      deletion_avg_accuracy += accuracy_score[1]
      fold += 1
    insertion_avg_accuracy /= float(number_of_splits)
    deletion_avg_accuracy /= float(number_of_splits)
    total_insertion_avg_accuracy.append(insertion_avg_accuracy)
    total_deletion_avg_accuracy.append(deletion_avg_accuracy)


  print "Average accuracy for insertions predictions is %f" % np.mean(total_insertion_avg_accuracy)
  print "Std in accuracy for insertions predictions is %f" % np.std(total_insertion_avg_accuracy)
  print "Average accuracy for deletions predictions is %f" % np.mean(total_deletion_avg_accuracy)
  print "Std in accuracy for deletions predictions is %f" % np.std(total_deletion_avg_accuracy)


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
sequence_file_name = "sequence_pam_gene_grna_big_file_donor.csv"
#data_folder = "/Users/amirali/Projects/CRISPR-data/R data/AM_TechMerg_Summary/"
#data_folder = "/Users/amirali/Projects/CRISPR-data-Feb18/20nt_counts_only/"


#name_genes_unique, name_genes_grna_unique, name_indel_type_unique, indel_count_matrix, indel_prop_matrix, length_indel = preprocess_indel_files(data_folder)

#pickle.dump(name_genes_unique, open('storage/name_genes_unique.p', 'wb'))
#pickle.dump(name_genes_grna_unique, open('storage/name_genes_grna_unique.p', 'wb'))
#pickle.dump(name_indel_type_unique, open('storage/name_indel_type_unique.p', 'wb'))
#pickle.dump(indel_count_matrix, open('storage/indel_count_matrix.p', 'wb'))
#pickle.dump(indel_prop_matrix, open('storage/indel_prop_matrix.p', 'wb'))
#pickle.dump(length_indel, open('storage/length_indel.p', 'wb'))


print "loading name_genes_unique ..."
name_genes_unique = pickle.load(open('storage/name_genes_unique.p', 'rb'))
print "loading name_genes_grna_unique ..."
name_genes_grna_unique = pickle.load(open('storage/name_genes_grna_unique.p', 'rb'))
print "loading name_indel_type_unique ..."
name_indel_type_unique = pickle.load(open('storage/name_indel_type_unique.p', 'rb'))
print "loading indel_count_matrix ..."
indel_count_matrix = pickle.load(open('storage/indel_count_matrix.p', 'rb'))
print "loading indel_prop_matrix ..."
indel_prop_matrix = pickle.load(open('storage/indel_prop_matrix.p', 'rb'))
print "loading length_indel ..."
length_indel = pickle.load(open('storage/length_indel.p', 'rb'))


indel_set_matrix,jaccard_matrix = variation_patients_and_lump(indel_count_matrix,sequence_file_name, name_genes_grna_unique)
plt.imshow(jaccard_matrix, cmap='hot', interpolation='nearest')
plt.savefig('jaccard.pdf')
plt.clf()



#count_insertions_gene_grna, count_deletions_gene_grna = compute_summary_statistics(name_genes_grna_unique, name_indel_type_unique, indel_count_matrix, indel_prop_matrix)

#sequence_pam_per_gene_grna, sequence_per_gene_grna, pam_per_gene_grna = load_gene_sequence(sequence_file_name, name_genes_grna_unique)
#sequence_pam_per_gene_grna = load_gene_sequence_interaction(sequence_file_name, name_genes_grna_unique)
#print np.shape(sequence_pam_per_gene_grna)
#print "Using both grna sequence and PAM"
#cross_validation_model(sequence_pam_per_gene_grna, count_insertions_gene_grna, count_deletions_gene_grna)
#print "Using only grna sequence"
#cross_validation_model(sequence_per_gene_grna, count_insertions_gene_grna, count_deletions_gene_grna)
#print "Using only PAM"
#cross_validation_model(pam_per_gene_grna, count_insertions_gene_grna, count_deletions_gene_grna)
#print "Using Kmers"
#k_mer_list = load_gene_sequence_k_mer(sequence_file_name, name_genes_grna_unique, 1)
#cross_validation_model(k_mer_list, count_insertions_gene_grna, count_deletions_gene_grna)

