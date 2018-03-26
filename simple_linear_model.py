from preprocess_indel_files import preprocess_indel_files
from compute_summary_statistic import compute_summary_statistics
from simple_summary_analysis import avg_length_pred
import numpy as np
from sklearn import linear_model, metrics
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle
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

def load_gene_sequence(sequence_file_name, name_genes_grna_unique):
  # Create numpy matrix of size len(name_genes_grna_unique) * 23, to store the sequence as one-hot encoded
  sequence_pam_per_gene_grna = np.zeros((len(name_genes_grna_unique), 23, 4), dtype = bool)
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
        for i in range(3):
          sequence_pam_per_gene_grna[index_in_name_genes_grna_unique, 20 + i, one_hot_index(l[3][i])] = 1
        for i in range(100):
          sequence_genom_context_gene_grna[index_in_name_genes_grna_unique, i, one_hot_index(l[6][i])] = 1

  plot_seq_logo(np.mean(sequence_pam_per_gene_grna, axis=0), "input_spacer")
  # Scikit needs only a 2-d matrix as input, so reshape and return
  return np.reshape(sequence_genom_context_gene_grna, (len(sequence_genom_context_gene_grna), -1)),np.reshape(sequence_pam_per_gene_grna, (len(name_genes_grna_unique), -1)), np.reshape(sequence_pam_per_gene_grna[:, :20, :], (len(name_genes_grna_unique), -1)), np.reshape(sequence_pam_per_gene_grna[:, 20:, :], (len(name_genes_grna_unique), -1))

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


def perform_linear_regression(sequence_pam_per_gene_grna, count_insertions_gene_grna_binary, count_deletions_gene_grna_binary, train_index, test_index, ins_coeff, del_coeff, to_plot = False):
  #lin_reg = linear_model.Lasso(alpha=0.001)
  lin_reg = linear_model.Ridge(alpha=100)
  #print "----"
  #print "Number of positive testing samples in insertions is %f" % np.sum(count_insertions_gene_grna_binary[test_index])
  #print "Total number of testing samples %f" % np.size(test_index)
  #print "Number of positive training samples in insertions is %f" % np.sum(count_insertions_gene_grna_binary[train_index])
  #print "Total number of training samples %f" % np.size(train_index)
  lin_reg.fit(sequence_pam_per_gene_grna[train_index], count_insertions_gene_grna_binary[train_index])
  lin_reg_pred = lin_reg.predict(sequence_pam_per_gene_grna[test_index])
  insertions_r2_score = lin_reg.score(sequence_pam_per_gene_grna[test_index], count_insertions_gene_grna_binary[test_index])
  insertion_rmse = sqrt(mean_squared_error(lin_reg_pred,count_insertions_gene_grna_binary[test_index]))
  ins_coeff.append(lin_reg.coef_)
  if to_plot:
    plot_QQ(lin_reg_pred,count_insertions_gene_grna_binary[test_index],'QQ_linear_insertion')
    plot_seq_logo(lin_reg.coef_, "Insertion_linear")
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
  deletions_r2_score = lin_reg.score(sequence_pam_per_gene_grna[test_index], count_deletions_gene_grna_binary[test_index])
  deletion_rmse = sqrt(mean_squared_error(lin_reg_pred, count_deletions_gene_grna_binary[test_index]))
  del_coeff.append(lin_reg.coef_)
  if to_plot:
    plot_QQ(lin_reg_pred, count_deletions_gene_grna_binary[test_index], 'QQ_linear_deletion')
    plot_seq_logo(lin_reg.coef_, "Deletion_linear")
  #print "Test r2_score score for deletions: %f" % deletions_r2_score
  #print "Train r2_score score for deletions: %f" % lin_reg.score(sequence_pam_per_gene_grna[train_index], count_deletions_gene_grna_binary[train_index])
  return insertions_r2_score, deletions_r2_score, insertion_rmse, deletion_rmse


def cross_validation_model(sequence_pam_per_gene_grna, count_insertions_gene_grna, count_deletions_gene_grna):
  total_insertion_avg_r2_score = []
  total_deletion_avg_r2_score = []
  total_insertion_avg_rmse = []
  total_deletion_avg_rmse = []

  ins_coeff = []
  del_coeff = []
  for repeat in range(200):
    fold_valid = KFold(n_splits = 3, shuffle = True, random_state = repeat)
    insertion_avg_r2_score = 0.0
    deletion_avg_r2_score = 0.0
    insertion_avg_rmse = 0.0
    deletion_avg_rmse = 0.0
    #count_insertions_gene_grna_copy = np.reshape(count_insertions_gene_grna, (-1, 1))
    #count_deletions_gene_grna_copy = np.reshape(count_deletions_gene_grna, (-1, 1))
    fold = 0
    for train_index, test_index in fold_valid.split(sequence_pam_per_gene_grna):
      to_plot = False
      if repeat == 1 and fold == 2:
        to_plot = True
      score_score = perform_linear_regression(sequence_pam_per_gene_grna, count_insertions_gene_grna, count_deletions_gene_grna, train_index, test_index, ins_coeff, del_coeff, to_plot)
      insertion_avg_r2_score += score_score[0]
      deletion_avg_r2_score += score_score[1]
      insertion_avg_rmse += score_score[2]
      deletion_avg_rmse += score_score[3]

      fold += 1
    insertion_avg_r2_score /= 3.0
    deletion_avg_r2_score /= 3.0
    insertion_avg_rmse /= 3.0
    deletion_avg_rmse /= 3.0

    total_insertion_avg_r2_score.append(float(insertion_avg_r2_score))
    total_deletion_avg_r2_score.append(float(deletion_avg_r2_score))
    total_insertion_avg_rmse.append(float(insertion_avg_rmse))
    total_deletion_avg_rmse.append(float(deletion_avg_rmse))
  # Some float overflows are happening, I will fix this sometime next week. Printing the array, it seems fine.
  print "Average r2 for insertions predictions is %f" % np.mean(np.array(total_insertion_avg_r2_score, dtype = float))
  print "Std in r2 for insertions predictions is %f" % np.std(np.array(total_insertion_avg_r2_score, dtype = float))

  print "Average rmse for insertions predictions is %f" % np.mean(np.array(total_insertion_avg_rmse, dtype = float))
  print "Std in rmse for insertions predictions is %f" % np.std(np.array(total_insertion_avg_rmse, dtype = float))

  print "Average r2 for deletions predictions is %f" % np.mean(np.array(total_deletion_avg_r2_score, dtype = float))
  print "Std in r2 for deletions predictions is %f" % np.std(np.array(total_deletion_avg_r2_score, dtype = float))

  print "Average rmse for deletions predictions is %f" % np.mean(np.array(total_deletion_avg_rmse, dtype = float))
  print "Std in rmse for deletions predictions is %f" % np.std(np.array(total_deletion_avg_rmse, dtype = float))


  #ins_coeff = np.array(ins_coeff)
  #del_coeff = np.array(del_coeff)
  #print "Average coefficients for insertions predictions is "
  #print np.mean(ins_coeff, axis = 0)
  #print "Variation in coefficients for insertions predictions is "
  #print np.var(ins_coeff, axis = 0)
  #print "Average coefficients for deletions predictions is "
  #print np.mean(del_coeff, axis = 0)
  #print "Variation in coefficients for deletions predictions is "
  #print np.var(del_coeff, axis = 0)


data_folder = "../IndelsFullData/"
sequence_file_name = "sequence_pam_gene_grna_big_file_donor_genomic_context.csv"
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

#count_insertions_gene_grna, count_deletions_gene_grna = compute_summary_statistics(name_genes_grna_unique, name_indel_type_unique, indel_count_matrix, indel_prop_matrix)
#prop_insertions_gene_grna, prop_deletions_gene_grna = avg_length_pred()

consider_length = 0
prop_insertions_gene_grna, prop_deletions_gene_grna = my_length_finder(indel_count_matrix,length_indel_insertion,length_indel_deletion,consider_length)

sequence_genom_context_gene_grna, sequence_pam_per_gene_grna, sequence_per_gene_grna, pam_per_gene_grna = load_gene_sequence(sequence_file_name, name_genes_grna_unique)

print "Using all Genomic Context"
cross_validation_model(sequence_genom_context_gene_grna, prop_insertions_gene_grna, prop_deletions_gene_grna)
#print "Using both grna sequence and PAM"
#cross_validation_model(sequence_pam_per_gene_grna, prop_insertions_gene_grna, prop_deletions_gene_grna)
#print "Using only grna sequence"
#cross_validation_model(sequence_per_gene_grna, prop_insertions_gene_grna, prop_deletions_gene_grna)
#print "Using only PAM"
#cross_validation_model(pam_per_gene_grna, prop_insertions_gene_grna, prop_deletions_gene_grna)
#k_mer_list = load_gene_sequence_k_mer(sequence_file_name, name_genes_grna_unique, 3)
#print k_mer_list
#cross_validation_model(k_mer_list, prop_insertions_gene_grna, prop_deletions_gene_grna)