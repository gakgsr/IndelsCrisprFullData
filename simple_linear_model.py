from preprocess_indel_files import preprocess_indel_files
from compute_summary_statistic import compute_summary_statistics
import numpy as np
from sklearn import linear_model, metrics
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def one_hot_index(nucleotide):
  nucleotide_array = ['A', 'C', 'G', 'T']
  return nucleotide_array.index(nucleotide)


def load_gene_sequence(sequence_file_name, name_genes_grna_unique):
  # Create numpy matrix of size len(name_genes_grna_unique) * 23 * 4, to store the sequence as one-hot encoded
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
  # Scikit needs only a 2-d matrix as input, so reshape and return
  return np.reshape(sequence_pam_per_gene_grna, (len(name_genes_grna_unique), -1)), np.reshape(sequence_pam_per_gene_grna[:, :20, :], (len(name_genes_grna_unique), -1)), np.reshape(sequence_pam_per_gene_grna[:, 20:, :], (len(name_genes_grna_unique), -1))

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


def perform_logistic_regression(sequence_pam_per_gene_grna, count_insertions_gene_grna_binary, count_deletions_gene_grna_binary, train_index, test_index, to_plot = False):
  lin_reg = linear_model.LinearRegression()
  #print "----"
  #print "Number of positive testing samples in insertions is %f" % np.sum(count_insertions_gene_grna_binary[test_index])
  #print "Total number of testing samples %f" % np.size(test_index)
  #print "Number of positive training samples in insertions is %f" % np.sum(count_insertions_gene_grna_binary[train_index])
  #print "Total number of training samples %f" % np.size(train_index)
  lin_reg.fit(sequence_pam_per_gene_grna[train_index], count_insertions_gene_grna_binary[train_index])
  lin_reg_pred = lin_reg.predict(sequence_pam_per_gene_grna[test_index])
  insertions_rmse = np.sqrt(np.mean((lin_reg_pred - count_insertions_gene_grna_binary[test_index])**2))
  if to_plot:
    plt.plot(lin_reg.coef_)
    plt.savefig('ins_lin_coeff.pdf')
    plt.clf()
  #insertions_r2_score = lin_reg.score(sequence_pam_per_gene_grna[test_index], count_insertions_gene_grna_binary[test_index])
  #print "Test mse_score score for insertions: %f" % insertions_r2_score
  #print "Train mse_score score for insertions: %f" % lin_reg.score(sequence_pam_per_gene_grna[train_index], count_insertions_gene_grna_binary[train_index])
  #print "----"
  #print "Number of positive testing samples in deletions is %f" % np.sum(count_deletions_gene_grna_binary[test_index])
  #print "Total number of testing samples %f" % np.size(test_index)
  #print "Number of positive training samples in deletions is %f" % np.sum(count_deletions_gene_grna_binary[train_index])
  #print "Total number of training samples %f" % np.size(train_index)
  lin_reg.fit(sequence_pam_per_gene_grna[train_index], count_deletions_gene_grna_binary[train_index])
  lin_reg_pred = lin_reg.predict(sequence_pam_per_gene_grna[test_index])
  deletions_rmse = np.sqrt(np.mean((lin_reg_pred - count_deletions_gene_grna_binary[test_index])**2))
  deletions_r2_score = lin_reg.score(sequence_pam_per_gene_grna[test_index], count_deletions_gene_grna_binary[test_index])
  if to_plot:
    plt.plot(lin_reg.coef_)
    plt.savefig('del_lin_coeff.pdf')
    plt.clf()
  #print "Test r2_score score for deletions: %f" % deletions_r2_score
  #print "Train r2_score score for deletions: %f" % lin_reg.score(sequence_pam_per_gene_grna[train_index], count_deletions_gene_grna_binary[train_index])
  return insertions_rmse, deletions_rmse


def cross_validation_model(sequence_pam_per_gene_grna, count_insertions_gene_grna, count_deletions_gene_grna):
  total_insertion_avg_r2_score = []
  total_deletion_avg_r2_score = []
  for repeat in range(2000):
    fold_valid = KFold(n_splits = 3, shuffle = True, random_state = repeat)
    insertion_avg_r2_score = 0.0
    deletion_avg_r2_score = 0.0
    #count_insertions_gene_grna_copy = np.reshape(count_insertions_gene_grna, (-1, 1))
    #count_deletions_gene_grna_copy = np.reshape(count_deletions_gene_grna, (-1, 1))
    fold = 0
    for train_index, test_index in fold_valid.split(sequence_pam_per_gene_grna):
      to_plot = False
      if repeat == 1999 and fold == 2:
        to_plot = True
      r2_score_score = perform_logistic_regression(sequence_pam_per_gene_grna, count_insertions_gene_grna, count_deletions_gene_grna, train_index, test_index, to_plot)
      insertion_avg_r2_score += r2_score_score[0]
      deletion_avg_r2_score += r2_score_score[1]
      fold += 1
    insertion_avg_r2_score /= 3.0
    deletion_avg_r2_score /= 3.0
    total_insertion_avg_r2_score.append(float(insertion_avg_r2_score))
    total_deletion_avg_r2_score.append(float(deletion_avg_r2_score))
  # Some float overflows are happening, I will fix this sometime next week. Printing the array, it seems fine.
  print "Average rmse for insertions predictions is %f" % np.mean(np.array(total_insertion_avg_r2_score, dtype = float))
  print "Variation in rmse for insertions predictions is %f" % np.var(np.array(total_insertion_avg_r2_score, dtype = float))
  print "Average rmse for deletions predictions is %f" % np.mean(np.array(total_deletion_avg_r2_score, dtype = float))
  print "Variation in rmse for deletions predictions is %f" % np.var(np.array(total_deletion_avg_r2_score, dtype = float))



data_folder = "../IndelsData/"
sequence_file_name = "sequence_pam_gene_grna.csv"
#data_folder = "/Users/amirali/Projects/CRISPR-data/R data/AM_TechMerg_Summary/"
name_genes_unique, name_genes_grna_unique, name_indel_type_unique, indel_count_matrix, indel_prop_matrix, length_indel = preprocess_indel_files(data_folder)
#count_insertions_gene_grna, count_deletions_gene_grna = compute_summary_statistics(name_genes_grna_unique, name_indel_type_unique, indel_count_matrix, indel_prop_matrix)
##
# Compute the proportions of insertions and deletions in each file
prop_insertions_gene_grna = np.zeros(len(name_genes_grna_unique), dtype = float)
prop_deletions_gene_grna = np.zeros(len(name_genes_grna_unique), dtype = float)
for i in range(len(name_genes_grna_unique)):
  for j in range(indel_prop_matrix.shape[0]):
    # across repeats
    if name_indel_type_unique[j].find('I') != -1:
      prop_insertions_gene_grna[i] += np.mean(indel_prop_matrix[j][3*i:3*i+3], dtype = float)
    if name_indel_type_unique[j].find('D') != -1:
      prop_deletions_gene_grna[i] += np.mean(indel_prop_matrix[j][3*i:3*i+3], dtype = float)
##
'''
sequence_pam_per_gene_grna, sequence_per_gene_grna, pam_per_gene_grna = load_gene_sequence(sequence_file_name, name_genes_grna_unique)
print "Using both grna sequence and PAM"
cross_validation_model(sequence_pam_per_gene_grna, prop_insertions_gene_grna, prop_deletions_gene_grna)
print "Using only grna sequence"
cross_validation_model(sequence_per_gene_grna, prop_insertions_gene_grna, prop_deletions_gene_grna)
print "Using only PAM"
cross_validation_model(pam_per_gene_grna, prop_insertions_gene_grna, prop_deletions_gene_grna)
'''
k_mer_list = load_gene_sequence_k_mer(sequence_file_name, name_genes_grna_unique, 3)
print k_mer_list
cross_validation_model(k_mer_list, prop_insertions_gene_grna, prop_deletions_gene_grna)