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
from sklearn.feature_selection import f_regression
import glob
import csv
from scipy import stats
from sklearn.feature_selection import chi2
import copy
from sklearn.neural_network import MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy.stats import entropy
from sklearn.metrics import r2_score

def top_indel_length_matrix_finder(indel_count_matrix,indel_length_deletion,indel_length_insertion):
    [indel_num,site_num]=np.shape(indel_count_matrix)

    top_deletion_length_matrix = np.zeros((3,site_num))
    top_insertion_length_matrix = np.zeros((3,site_num))
    for siteidx in range(site_num):
        top_deletion_length_matrix[:,siteidx] = indel_length_deletion[np.argsort(indel_count_matrix[:,siteidx])[::-1][0:3]]
        top_insertion_length_matrix[:, siteidx] = indel_length_insertion[np.argsort(indel_count_matrix[:,siteidx])[::-1][0:3]]


    return top_deletion_length_matrix
    return top_insertion_length_matrix

def oneI_oneD_fraction_finder(indel_count_matrix,name_indel_type_unique):
    indel_num, site_num = np.shape(indel_count_matrix)
    oneI_indicator = np.zeros(indel_num)
    oneI_fraction = np.zeros(site_num)
    oneD_fraction = np.zeros(site_num)

    for counter, cigar in enumerate(name_indel_type_unique):
        if ":1I" in cigar:
            oneI_indicator[counter] = 1

    oneD_indicator = np.zeros(indel_num)
    for counter, cigar in enumerate(name_indel_type_unique):
        if ":1D" in cigar:# and cigar.count(':')==1:
            oneD_indicator[counter] = 1

    indel_fraction_mutant_matrix = indel_count_matrix / np.reshape(np.sum(indel_count_matrix, axis=0), (1, -1))
    for site_index in range(site_num):
        oneI_fraction[site_index] = np.inner(oneI_indicator, indel_fraction_mutant_matrix[:, site_index])
        oneD_fraction[site_index] = np.inner(oneD_indicator, indel_fraction_mutant_matrix[:, site_index])

    return oneI_fraction,oneD_fraction


def entrop_finder(indel_count_matrix):
    num_indels, num_sites = np.shape(indel_count_matrix)
    indel_fraction_mutant_matrix = indel_count_matrix / np.reshape(np.sum(indel_count_matrix, axis=0), (1, -1))
    entrop = []
    for col in range(num_sites):
        vec = np.copy(indel_fraction_mutant_matrix[:, col])
        vec = np.sort(vec)[::-1]
        entrop.append(entropy(vec))

    return np.asarray(entrop)

def flatness_finder(indel_count_matrix):
    num_indels, num_sites = np.shape(indel_count_matrix)
    indel_fraction_mutant_matrix = indel_count_matrix / np.reshape(np.sum(indel_count_matrix, axis=0), (1, -1))
    max_grad = []
    for col in range(num_sites):
        vec = np.copy(indel_fraction_mutant_matrix[:, col])
        vec = np.sort(vec)[::-1]
        max_grad.append(max(abs(np.gradient(vec))))

    return np.asarray(max_grad)



def flatness_finder(indel_count_matrix):
    num_indels, num_sites = np.shape(indel_count_matrix)
    indel_fraction_mutant_matrix = indel_count_matrix / np.reshape(np.sum(indel_count_matrix, axis=0), (1, -1))
    max_grad = []
    for col in range(num_sites):
        vec = np.copy(indel_fraction_mutant_matrix[:, col])
        vec = np.sort(vec)[::-1]
        max_grad.append(max(abs(np.gradient(vec))))

    return np.asarray(max_grad)

def length_of_repeat_finder(seq):
    maxlen = 2
    start = 0
    while start < len(seq) - 1:
        pointer = 2
        nuc1 = seq[start]
        nuc2 = seq[start + 1]
        templen = 2
        while start + pointer < len(seq) and nuc1 != nuc2:
            # print templen
            if pointer % 2 == 0:
                if seq[start + pointer] != nuc1:
                    pointer += 1
                    break
                templen += 1
                if templen > maxlen:
                    maxlen = templen

            if pointer % 2 == 1:
                if seq[start + pointer] != nuc2:
                    pointer += 1
                    if templen > maxlen:
                        maxlen = templen
                    break
                templen += 1
                if templen > maxlen:
                    maxlen = templen

            pointer += 1
        start = start + 1
    return maxlen


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

        if 2. in intron_exon_dict[location]: # if we find ANY 2 we count as exon
            intron_exon_label_vec.append(2)
        elif 1. in intron_exon_dict[location]:
            intron_exon_label_vec.append(1)
        else:
            intron_exon_label_vec.append(0)

    intron_exon_label_vec = np.asarray(intron_exon_label_vec)
    return intron_exon_label_vec

def eff_vec_finder(indel_count_matrix,name_genes_grna_unique):
    num_indel,num_site = np.shape(indel_count_matrix)
    dict_eff = {}
    for filename in glob.glob('/Users/amirali/Projects/muteff/*.txt'):
        file = open(filename)
        for line in file:
            if 'RL384' in line:
                line = line.replace('_','-')
                line = line.replace('"', '')
                if 'N' not in line.split(',')[1]:
                    eff = float(line.split(',')[1])
                    line_list = (line.split(',')[0]).split('-')
                    dict_eff[line_list[1]+'-'+line_list[2]] = eff


    eff_vec = np.zeros(num_site)
    site = 0
    for site_name in name_genes_grna_unique:
        site_name_list = site_name.split('-')
        eff_vec[site] = dict_eff[site_name_list[1] + '-' + site_name_list[2]]
        site += 1

    return eff_vec



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




def fraction_of_deletion_insertion_porportion(indel_prop_matrix,length_indel_insertion,length_indel_deletion):
  indel_num,site_num = np.shape(indel_prop_matrix)

  prop_insertions_gene_grna = np.zeros(site_num,dtype=float)
  prop_deletions_gene_grna = np.zeros(site_num,dtype=float)


  insertion_indicator = np.copy(length_indel_insertion)
  deletion_indicator = np.copy(length_indel_deletion)

  insertion_indicator[insertion_indicator>0]=1.
  deletion_indicator[deletion_indicator>0]=1.

  for site_index in range(site_num):
    prop_insertions_gene_grna[site_index] = np.inner(insertion_indicator,indel_prop_matrix[:,site_index])
    prop_deletions_gene_grna[site_index] = np.inner(deletion_indicator, indel_prop_matrix[:, site_index])

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

def load_gene_sequence(sequence_file_name, name_genes_grna_unique,homopolymer_matrix,intron_exon_label_vec):
  # Create numpy matrix of size len(name_genes_grna_unique) * 23, to store the sequence as one-hot encoded
  sequence_pam_per_gene_grna = np.zeros((len(name_genes_grna_unique), 23, 4), dtype = bool)
  sequence_pam_homop_per_gene_grna = np.zeros((len(name_genes_grna_unique), 24, 4))
  sequence_pam_repeat_per_gene_grna = np.zeros((len(name_genes_grna_unique), 24, 4))
  sequence_pam_chromatin_per_gene_grna = np.zeros((len(name_genes_grna_unique), 24, 4))
  sequence_pam_coding_gccontent_per_gene_grna = np.zeros((len(name_genes_grna_unique), 24, 4))
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
          sequence_pam_coding_gccontent_per_gene_grna[index_in_name_genes_grna_unique, i, one_hot_index(l[2][i])] = 1
          sequence_pam_repeat_per_gene_grna[index_in_name_genes_grna_unique, i, one_hot_index(l[2][i])] = 1
          sequence_pam_chromatin_per_gene_grna[index_in_name_genes_grna_unique, i, one_hot_index(l[2][i])] = 1
        for i in range(3):
          sequence_pam_per_gene_grna[index_in_name_genes_grna_unique, 20 + i, one_hot_index(l[3][i])] = 1
          sequence_pam_homop_per_gene_grna[index_in_name_genes_grna_unique, 20 + i, one_hot_index(l[3][i])] = 1
          sequence_pam_coding_gccontent_per_gene_grna[index_in_name_genes_grna_unique, 20 + i, one_hot_index(l[3][i])] = 1
          sequence_pam_repeat_per_gene_grna[index_in_name_genes_grna_unique, 20 + i, one_hot_index(l[3][i])] = 1
          sequence_pam_chromatin_per_gene_grna[index_in_name_genes_grna_unique, 20 + i, one_hot_index(l[3][i])] = 1

        if length_of_repeat_finder(l[2])>4:
          sequence_pam_repeat_per_gene_grna[index_in_name_genes_grna_unique, 23 , 0] = 1
        #print 'sequence', l[2]
        #print 'repeat', length_of_repeat_finder(l[2])

        #sequence_pam_chromatin_per_gene_grna[index_in_name_genes_grna_unique, 23 , 0] = np.nanmean(chrom_mat[l[1] + '-' + l[0]],axis=0)[chrom_col]

        sequence_pam_homop_per_gene_grna[index_in_name_genes_grna_unique, 23 , :] = homopolymer_matrix[:,index_in_name_genes_grna_unique]
        if intron_exon_label_vec[index_in_name_genes_grna_unique] == 2: # if exon
          sequence_pam_coding_gccontent_per_gene_grna[index_in_name_genes_grna_unique, 23 , 0] = 1
        for i in range(100):
          sequence_genom_context_gene_grna[index_in_name_genes_grna_unique, i, one_hot_index(l[6][i])] = 1

        # sequence_pam_coding_gccontent_per_gene_grna[index_in_name_genes_grna_unique, 23, 1] = np.sum(sequence_pam_per_gene_grna[index_in_name_genes_grna_unique,:20,1:3]) / float(np.sum(sequence_pam_per_gene_grna[index_in_name_genes_grna_unique,:20,:]))
        sequence_pam_coding_gccontent_per_gene_grna[index_in_name_genes_grna_unique, 23, 1] = np.sum(sequence_genom_context_gene_grna[index_in_name_genes_grna_unique, :100, 1:3]) / float(np.sum(sequence_genom_context_gene_grna[index_in_name_genes_grna_unique, :100, :]))

  #plot_seq_logo(np.mean(sequence_pam_per_gene_grna, axis=0), "input_spacer")
  # Scikit needs only a 2-d matrix as input, so reshape and return
  return np.reshape(sequence_genom_context_gene_grna, (len(sequence_pam_repeat_per_gene_grna), -1)),  np.reshape(sequence_pam_chromatin_per_gene_grna, (len(sequence_pam_chromatin_per_gene_grna), -1))      ,np.reshape(sequence_pam_repeat_per_gene_grna, (len(sequence_genom_context_gene_grna), -1)) ,np.reshape(sequence_pam_coding_gccontent_per_gene_grna, (len(sequence_pam_coding_gccontent_per_gene_grna), -1))  ,np.reshape(sequence_pam_homop_per_gene_grna, (len(sequence_pam_homop_per_gene_grna), -1)),np.reshape(sequence_pam_per_gene_grna, (len(name_genes_grna_unique), -1)), np.reshape(sequence_pam_per_gene_grna[:, :20, :], (len(name_genes_grna_unique), -1)), np.reshape(sequence_pam_per_gene_grna[:, 20:, :], (len(name_genes_grna_unique), -1))

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


def perform_linear_regression(sequence_pam_per_gene_grna, count_insertions_gene_grna_binary, count_deletions_gene_grna_binary, train_index, test_index, ins_coeff, del_coeff,lin_reg, to_plot = False):

  #print "----"
  #print "Number of positive testing samples in insertions is %f" % np.sum(count_insertions_gene_grna_binary[test_index])
  #print "Total number of testing samples %f" % np.size(test_index)
  #print "Number of positive training samples in insertions is %f" % np.sum(count_insertions_gene_grna_binary[train_index])
  #print "Total number of training samples %f" % np.size(train_index)
  lin_reg.fit(sequence_pam_per_gene_grna[train_index], count_insertions_gene_grna_binary[train_index])
  lin_reg_pred = lin_reg.predict(sequence_pam_per_gene_grna[test_index])
  insertions_r2_score = lin_reg.score(sequence_pam_per_gene_grna[test_index], count_insertions_gene_grna_binary[test_index])
  insertion_rmse = sqrt(mean_squared_error(lin_reg_pred,count_insertions_gene_grna_binary[test_index]))
  # ins_coeff.append(lin_reg.coef_)
  # if to_plot:
  #   pvalue_vec = f_regression(sequence_pam_per_gene_grna[test_index],lin_reg_pred, center=True)[1]
  #   #print np.shape(sequence_pam_per_gene_grna[test_index])
  #   #print np.shape(lin_reg_pred)
  #   #scores, pvalue_vec = chi2(sequence_pam_per_gene_grna[test_index], lin_reg_pred)
  #   #pvalue_vec = f_regression(sequence_pam_per_gene_grna[train_index], count_deletions_gene_grna_binary[train_index])[1]
  #   plot_QQ(lin_reg_pred,count_insertions_gene_grna_binary[test_index],'QQ_linear_insertion')
  #   plot_seq_logo(lin_reg.coef_, "Insertion_linear")
  #   plot_seq_logo(-np.log10(pvalue_vec), "Insertion_linear_pvalue")
  #   print 'Insertion -log10(p-value) of last 4 entries', -np.log10(pvalue_vec)[-4:]
  #   print 'Insertion last four coefficients', lin_reg.coef_[-4:]

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
  # del_coeff.append(lin_reg.coef_)
  # if to_plot:
  #   #print np.shape(sequence_pam_per_gene_grna[test_index])
  #   #print np.shape(lin_reg_pred)
  #   pvalue_vec = f_regression(sequence_pam_per_gene_grna[test_index], lin_reg_pred, center=True)[1]
  #   #scores, pvalue_vec = chi2(sequence_pam_per_gene_grna[test_index], lin_reg_pred)
  #   #pvalue_vec = f_regression(sequence_pam_per_gene_grna[train_index], count_deletions_gene_grna_binary[train_index])[1]
  #   plot_seq_logo(-np.log10(pvalue_vec)*np.sign(lin_reg.coef_), "Deletion_linear_pvalue" )
  #   plot_QQ(lin_reg_pred, count_deletions_gene_grna_binary[test_index], 'QQ_linear_deletion')
  #   plot_seq_logo(lin_reg.coef_, "Deletion_linear")
  #   print 'Deletion -log10(p-value) of last 4 entries', -np.log10(pvalue_vec)[-4:]
  #   print 'Deletion last four coefficients', lin_reg.coef_[-4:]
  #print "Test r2_score score for deletions: %f" % deletions_r2_score
  #print "Train r2_score score for deletions: %f" % lin_reg.score(sequence_pam_per_gene_grna[train_index], count_deletions_gene_grna_binary[train_index])
  return insertions_r2_score, deletions_r2_score, insertion_rmse, deletion_rmse


def cross_validation_model(sequence_pam_per_gene_grna, count_insertions_gene_grna, count_deletions_gene_grna,lin_reg):
  total_insertion_avg_r2_score = []
  total_deletion_avg_r2_score = []
  total_insertion_avg_rmse = []
  total_deletion_avg_rmse = []

  ins_coeff = []
  del_coeff = []
  for repeat in range(10):
    print "repeat", repeat
    number_of_splits = 5
    fold_valid = KFold(n_splits = number_of_splits, shuffle = True, random_state = repeat)
    insertion_avg_r2_score = 0.0
    deletion_avg_r2_score = 0.0
    insertion_avg_rmse = 0.0
    deletion_avg_rmse = 0.0
    #count_insertions_gene_grna_copy = np.reshape(count_insertions_gene_grna, (-1, 1))
    #count_deletions_gene_grna_copy = np.reshape(count_deletions_gene_grna, (-1, 1))
    fold = 0
    for train_index, test_index in fold_valid.split(sequence_pam_per_gene_grna):
      to_plot = False
      if repeat == 2 and fold == 1:
        to_plot = True
      score_score = perform_linear_regression(sequence_pam_per_gene_grna, count_insertions_gene_grna, count_deletions_gene_grna, train_index, test_index, ins_coeff, del_coeff,lin_reg,to_plot)
      insertion_avg_r2_score += score_score[0]
      deletion_avg_r2_score += score_score[1]
      insertion_avg_rmse += score_score[2]
      deletion_avg_rmse += score_score[3]

      fold += 1
    insertion_avg_r2_score /= float(number_of_splits)
    deletion_avg_r2_score /= float(number_of_splits)
    insertion_avg_rmse /= float(number_of_splits)
    deletion_avg_rmse /= float(number_of_splits)

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


print "files loading ..."
name_genes_grna_unique = pickle.load(open('Tcell-files/name_genes_grna_UNIQUE.p', 'rb'))
name_indel_type_unique = pickle.load(open('Tcell-files/name_indel_type_ALL.p', 'rb'))
indel_count_matrix = pickle.load(open('Tcell-files/indel_count_matrix_UNIQUE.p', 'rb'))
indel_prop_matrix = pickle.load(open('Tcell-files/indel_prop_matrix_UNIQUE.p', 'rb'))
length_indel_insertion = pickle.load(open('Tcell-files/length_indel_insertion_ALL.p', 'rb'))
length_indel_deletion = pickle.load(open('Tcell-files/length_indel_deletion_ALL.p', 'rb'))
homopolymer_matrix = pickle.load(open('Tcell-files/homology_matrix_UNIQUE.p', 'rb'))
chrom_label_matrix = pickle.load(open('Tcell-files/chrom_label_matrix_UNIQUE.p', 'rb'))
thermo_matrix = pickle.load(open('/Users/amirali/Projects/thermo_properties.p', 'rb'))
top_deletion_length_matrix = pickle.load(open('Tcell-files/top_deletion_length_matrix_UNIQUE.p', 'rb'))
top_insertion_length_matrix = pickle.load(open('Tcell-files/top_insertion_length_matrix_UNIQUE.p', 'rb'))
high_variance_indicator =  pickle.load(open('Tcell-files/high_variance_indicator.p', 'rb'))
insertion_matrix = pickle.load(open('Tcell-files/insertion_matrix_UNIQUE.p', 'rb'))




# #
# # length_one_counter = 0
# # deletion_counter = 0
# # homo3_counter = 0
# # LL = 0.
# # for i in range(len(name_genes_grna_unique)):
# #     if 2 in homopolymer_matrix[:,i] and 3 not in homopolymer_matrix[:,i]:
# #         homo3_counter += 1
# #         inde_type = name_indel_type_unique[np.argsort(indel_count_matrix[:,i])[-1]]
# #         if 'D' in inde_type:
# #             deletion_counter += 1
# #             if int(inde_type.strip('D').split(':')[1]) != 2 and int(inde_type.strip('D').split(':')[1]) != 1:
# #                 length_one_counter+=1
# #                 LL += int(inde_type.strip('D').split(':')[1])
# #                 print inde_type
# #
# # print "Number of sites with a deletion of length 1:", length_one_counter
# # print "Total number of sites with top-deletion:", deletion_counter
# # print "Total homo counter:", homo3_counter
# # print LL/length_one_counter
#
# #count_insertions_gene_grna, count_deletions_gene_grna = compute_summary_statistics(name_genes_grna_unique, name_indel_type_unique, indel_count_matrix, indel_prop_matrix)
# #prop_insertions_gene_grna, prop_deletions_gene_grna = avg_length_pred()
#
# #consider_length = 1
fraction_insertions, fraction_deletions = fraction_of_deletion_insertion(indel_count_matrix,length_indel_insertion,length_indel_deletion)
fraction_insertions_all, fraction_deletions_all = fraction_of_deletion_insertion_porportion(indel_prop_matrix,length_indel_insertion,length_indel_deletion)
exp_insertion_length, exp_deletion_length = expected_deletion_insertion_length(indel_count_matrix,length_indel_insertion,length_indel_deletion)
eff_vec = eff_vec_finder(indel_count_matrix,name_genes_grna_unique)
intron_exon_label_vec = coding_region_finder(name_genes_grna_unique)
max_grad = flatness_finder(indel_count_matrix)
entrop = entrop_finder(indel_count_matrix)
oneI_frac,oneD_frac = oneI_oneD_fraction_finder(indel_count_matrix,name_indel_type_unique)
my_eff_vec = pickle.load(open('Tcell-files/my_eff_vec_UNIQUE_no_others.p', 'rb'))
my_eff_vec = np.asarray(my_eff_vec)
mean_eff_vec = pickle.load(open('Tcell-files/eff_vec_mean_UNIQUE_no_others.p', 'rb'))
mean_eff_vec = np.asarray(mean_eff_vec)

# plt.plot(my_eff_vec,eff_vec,'o')
# plt.xlabel('from data')
# plt.ylabel('muteff')
# plt.savefig('test.pdf')
# plt.clf()

# max_dif_index = np.argsort(np.abs(my_eff_vec-eff_vec))[::-1]
# print my_eff_vec[max_dif_index[0]]
# print eff_vec[max_dif_index[0]]
# print np.asarray(name_genes_grna_unique)[max_dif_index[0]]
#
# print indel_count_matrix[0:10,max_dif_index[240]]
# print name_indel_type_unique[0:10]
# print np.sum(indel_count_matrix[:,max_dif_index[240]])


# chrom_mat = pickle.load(open('storage/chrom_label_dic.p', 'rb'))
# chrom_mat_name = pickle.load(open('storage/chrom_label_dic_name.p', 'rb'))

# #for chrom_col in range(len(chrom_mat_name)):
# for chrom_col in [len(chrom_mat_name)-2]:
#     if chrom_mat_name[chrom_col] not in  ['GerpRS','bStatistic','fitCons','EncExp','EncNucleo']: #these ones has all nan
#         print "*****"
#         print chrom_mat_name[chrom_col]
#         sequence_genom_context_gene_grna, sequence_pam_chromatin_per_gene_grna, sequence_pam_repeat_per_gene_grna, sequence_pam_coding_gccontent_per_gene_grna, sequence_pam_homop_per_gene_grna , sequence_pam_per_gene_grna, sequence_per_gene_grna, pam_per_gene_grna = load_gene_sequence(sequence_file_name, name_genes_grna_unique,homopolymer_matrix,intron_exon_label_vec,chrom_mat,chrom_col)
#         cross_validation_model(sequence_pam_chromatin_per_gene_grna, eff_vec, eff_vec)
#         #cross_validation_model(sequence_pam_chromatin_per_gene_grna, fraction_insertions, fraction_deletions)

sequence_genom_context_gene_grna, sequence_pam_chromatin_per_gene_grna, sequence_pam_repeat_per_gene_grna, sequence_pam_coding_gccontent_per_gene_grna, sequence_pam_homop_per_gene_grna , sequence_pam_per_gene_grna, sequence_per_gene_grna, pam_per_gene_grna = load_gene_sequence(sequence_file_name, name_genes_grna_unique,homopolymer_matrix,intron_exon_label_vec)
#sequence_pam_per_gene_grna = np.concatenate((sequence_pam_per_gene_grna, chrom_label_matrix), axis=1)
#sequence_pam_per_gene_grna = np.concatenate((sequence_pam_per_gene_grna, thermo_matrix[0]), axis=1)



# good for edit eff
# lin_reg = linear_model.Ridge(alpha=200)
# good for fractions
#lin_reg = linear_model.Ridge(alpha=10)
# lin_reg = MLPRegressor(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(2,), random_state=1)
#lin_reg = RandomForestRegressor(n_estimators = 100)
lin_reg = XGBRegressor(n_estimators=600, max_depth=1)
#lin_reg = linear_model.LinearRegression()
#eff_vec = np.sum(indel_prop_matrix,axis=0)

#print "Using all Genomic Context"
#cross_validation_model(sequence_genom_context_gene_grna, prop_insertions_gene_grna, prop_deletions_gene_grna)
#cross_validation_model(sequence_genom_context_gene_grna, eff_vec, eff_vec)

#print "Using both grna sequence and PAM"
#cross_validation_model(sequence_pam_chromatin_per_gene_grna, eff_vec, eff_vec)
#cross_validation_model(sequence_pam_chromatin_per_gene_grna, eff_vec, eff_vec)

#mean_eff_vec = mean_eff_vec[high_variance_indicator==0]
#sequence_pam_per_gene_grna = sequence_pam_per_gene_grna[high_variance_indicator==0,:]


#cross_validation_model(sequence_pam_per_gene_grna, insertion_matrix[2,:], insertion_matrix[3,:], lin_reg)
#cross_validation_model(sequence_pam_per_gene_grna, oneI_frac, oneD_frac, lin_reg)
#cross_validation_model(sequence_pam_per_gene_grna, max_grad, max_grad)
#cross_validation_model(sequence_pam_per_gene_grna, entrop, entrop,lin_reg)
#cross_validation_model(sequence_pam_per_gene_grna, mean_eff_vec, mean_eff_vec, lin_reg)
#cross_validation_model(sequence_pam_per_gene_grna, fraction_insertions, fraction_deletions, lin_reg)
#cross_validation_model(sequence_pam_per_gene_grna, fraction_insertions_all, fraction_deletions_all, lin_reg)
#cross_validation_model(sequence_pam_per_gene_grna, fraction_insertions, fraction_deletions)
#cross_validation_model(sequence_pam_per_gene_grna, exp_insertion_length, exp_deletion_length, lin_reg)
#cross_validation_model(sequence_pam_per_gene_grna, top_deletion_length_matrix[0,:], top_deletion_length_matrix[1,:], lin_reg)

#print "Using only grna sequence"
#cross_validation_model(sequence_per_gene_grna, prop_insertions_gene_grna, prop_deletions_gene_grna)
#print "Using only PAM"
#cross_validation_model(pam_per_gene_grna, prop_insertions_gene_grna, prop_deletions_gene_grna)
#k_mer_list = load_gene_sequence_k_mer(sequence_file_name, name_genes_grna_unique, 3)
#print k_mer_list
#cross_validation_model(k_mer_list, prop_insertions_gene_grna, prop_deletions_gene_grna)

###

# print np.shape(eff_vec)
# plt.hist(eff_vec, bins=20)
# #plt.hist(eff_vec)
# plt.savefig('plots/Edit_efficiency_hist.pdf')
# plt.clf()
#
# print "edi mean = ", np.mean(eff_vec)
# print "edit median =", np.median(eff_vec)

#plt.hist(top_insertion_length_matrix[2,:], bins=100)
#plt.savefig('plots/Top3-insertion-length.pdf')

plt.hist(insertion_matrix[0,:])
plt.savefig('plots/InsertionA.pdf')
plt.clf()

plt.hist(insertion_matrix[1,:])
plt.savefig('plots/InsertionC.pdf')
plt.clf()

plt.hist(insertion_matrix[2,:])
plt.savefig('plots/InsertionG.pdf')
plt.clf()

plt.hist(insertion_matrix[3,:])
plt.savefig('plots/InsertionT.pdf')
plt.clf()