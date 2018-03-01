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


def perform_logistic_regression(sequence_pam_per_gene_grna, count_insertions_gene_grna_binary, count_deletions_gene_grna_binary, train_index, test_index):
  log_reg = linear_model.LogisticRegression(C=1e5)
  #print "----"
  #print "Number of positive testing samples in insertions is %f" % np.sum(count_insertions_gene_grna_binary[test_index])
  #print "Total number of testing samples %f" % np.size(test_index)
  #print "Number of positive training samples in insertions is %f" % np.sum(count_insertions_gene_grna_binary[train_index])
  #print "Total number of training samples %f" % np.size(train_index)
  log_reg.fit(sequence_pam_per_gene_grna[train_index], count_insertions_gene_grna_binary[train_index])
  log_reg_pred = log_reg.predict(sequence_pam_per_gene_grna[test_index])
  log_reg_pred_train = log_reg.predict(sequence_pam_per_gene_grna[train_index])
  #insertions_accuracy = metrics.accuracy_score(count_insertions_gene_grna_binary[test_index], log_reg_pred)
  #temp = np.copy(count_insertions_gene_grna_binary[test_index])
  number_of_ones_in_training = np.sum(count_insertions_gene_grna_binary[train_index])
  number_of_ones_in_training = int( float(number_of_ones_in_training)/float(np.size(train_index))*float(np.size(test_index))  )
  temp = number_of_ones_in_training*[1]+(np.size(test_index)-number_of_ones_in_training)*[0]
  np.random.shuffle(temp)
  insertions_accuracy = metrics.accuracy_score(temp,count_insertions_gene_grna_binary[test_index])
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
  #deletions_accuracy = metrics.accuracy_score(count_deletions_gene_grna_binary[test_index], log_reg_pred)
  #print "Test accuracy score for deletions: %f" % deletions_accuracy
  #print "Train accuracy score for deletions: %f" % metrics.accuracy_score(count_deletions_gene_grna_binary[train_index], log_reg_pred_train)
  #temp = np.copy(count_insertions_gene_grna_binary[test_index])
  number_of_ones_in_training = np.sum(count_deletions_gene_grna_binary[train_index])
  number_of_ones_in_training = int( float(number_of_ones_in_training)/float(np.size(train_index))*float(np.size(test_index))  )
  temp = number_of_ones_in_training*[1]+(np.size(test_index)-number_of_ones_in_training)*[0]
  np.random.shuffle(temp)
  deletions_accuracy = metrics.accuracy_score(temp,count_insertions_gene_grna_binary[test_index])
  return insertions_accuracy, deletions_accuracy


def cross_validation_model(sequence_pam_per_gene_grna, count_insertions_gene_grna, count_deletions_gene_grna):
  total_insertion_avg_accuracy = []
  total_deletion_avg_accuracy = []
  for repeat in range(2000):
    fold_valid = KFold(n_splits = 3, shuffle = True, random_state = repeat)
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
    for train_index, test_index in fold_valid.split(sequence_pam_per_gene_grna):
      accuracy_score = perform_logistic_regression(sequence_pam_per_gene_grna, count_insertions_gene_grna_binary, count_deletions_gene_grna_binary, train_index, test_index)
      insertion_avg_accuracy += accuracy_score[0]
      deletion_avg_accuracy += accuracy_score[1]
    insertion_avg_accuracy /= 3.0
    deletion_avg_accuracy /= 3.0
    total_insertion_avg_accuracy.append(insertion_avg_accuracy)
    total_deletion_avg_accuracy.append(deletion_avg_accuracy)
  print "Average accuracy for insertions predictions is %f" % np.mean(total_insertion_avg_accuracy)
  print "Variation in accuracy for insertions predictions is %f" % np.var(total_insertion_avg_accuracy)
  print "Average accuracy for deletions predictions is %f" % np.mean(total_deletion_avg_accuracy)
  print "Variation in accuracy for deletions predictions is %f" % np.var(total_deletion_avg_accuracy)



#data_folder = "../IndelsData/"
sequence_file_name = "sequence_pam_gene_grna.csv"
data_folder = "/Users/amirali/Projects/CRISPR-data/R data/AM_TechMerg_Summary/"
name_genes_unique, name_genes_grna_unique, name_indel_type_unique, indel_count_matrix, indel_prop_matrix = preprocess_indel_files(data_folder)
count_insertions_gene_grna, count_deletions_gene_grna = compute_summary_statistics(name_genes_grna_unique, name_indel_type_unique, indel_count_matrix, indel_prop_matrix)
sequence_pam_per_gene_grna, sequence_per_gene_grna, pam_per_gene_grna = load_gene_sequence(sequence_file_name, name_genes_grna_unique)
print "Using both grna sequence and PAM"
cross_validation_model(sequence_pam_per_gene_grna, count_insertions_gene_grna, count_deletions_gene_grna)
print "Using only grna sequence"
cross_validation_model(sequence_per_gene_grna, count_insertions_gene_grna, count_deletions_gene_grna)
print "Using only PAM"
cross_validation_model(pam_per_gene_grna, count_insertions_gene_grna, count_deletions_gene_grna)