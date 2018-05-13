import numpy as np
import pickle
from xgboost import XGBClassifier, XGBRegressor
import glob
import csv
from scipy import stats
from sklearn.feature_selection import chi2
import copy
from sklearn.neural_network import MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy.stats import entropy
from scipy.stats import kendalltau

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


def oneI_oneD_fraction_over_total_finder(indel_prop_matrix,name_indel_type_unique):
    indel_num, site_num = np.shape(indel_prop_matrix)
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

    for site_index in range(site_num):
        oneI_fraction[site_index] = np.inner(oneI_indicator, indel_prop_matrix[:, site_index])
        oneD_fraction[site_index] = np.inner(oneD_indicator, indel_prop_matrix[:, site_index])

    return oneI_fraction,oneD_fraction

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


name_genes_grna_unique = pickle.load(open('Tcell-files/name_genes_grna_UNIQUE.p', 'rb'))
name_indel_type_unique = pickle.load(open('Tcell-files/name_indel_type_ALL.p', 'rb'))
indel_count_matrix = pickle.load(open('Tcell-files/indel_count_matrix_UNIQUE.p', 'rb'))
indel_prop_matrix = pickle.load(open('Tcell-files/indel_prop_matrix_UNIQUE.p', 'rb'))
length_indel_insertion = pickle.load(open('Tcell-files/length_indel_insertion_ALL.p', 'rb'))
length_indel_deletion = pickle.load(open('Tcell-files/length_indel_deletion_ALL.p', 'rb'))
homopolymer_matrix = pickle.load(open('Tcell-files/homology_matrix_UNIQUE.p', 'rb'))
oneI_frac,oneD_frac = oneI_oneD_fraction_finder(indel_count_matrix,name_indel_type_unique)

intron_exon_label_vec = coding_region_finder(name_genes_grna_unique)
sequence_file_name = "sequence_pam_gene_grna_big_file_donor_genomic_context.csv"
sequence_genom_context_gene_grna_tcel, sequence_pam_chromatin_per_gene_grna_tcel, sequence_pam_repeat_per_gene_grna_tcel, sequence_pam_coding_gccontent_per_gene_grna_tcel, sequence_pam_homop_per_gene_grna_tcel , sequence_pam_per_gene_grna_tcel, sequence_per_gene_grna_tcel, pam_per_gene_grna_tcel = load_gene_sequence(sequence_file_name, name_genes_grna_unique,homopolymer_matrix,intron_exon_label_vec)

gene_list = []
#oneI_frac,oneD_frac = oneI_oneD_fraction_finder(indel_count_matrix,name_indel_type_unique)
oneI_frac,oneD_frac = oneI_oneD_fraction_over_total_finder(indel_prop_matrix,name_indel_type_unique)
for site in name_genes_grna_unique:
    gene_list.append(site.split('-')[0])

lin_reg = XGBRegressor(n_estimators=30, max_depth=4) # 20,5 for frac insertion
lin_reg.fit(sequence_pam_per_gene_grna_tcel[0:1203]   ,oneD_frac[0:1203])
lin_reg_pred = lin_reg.predict(sequence_pam_per_gene_grna_tcel)

kendalltau_vec = []
selected_counter = 0
useful_counter = 0
test_genes = np.unique(gene_list[1203:])
for counter, gene in enumerate(test_genes):
    print gene
    local_ind =  np.where(np.asarray(gene_list) == gene)[0]
    list1 = np.argsort(lin_reg_pred[local_ind])
    list2 = np.argsort(oneD_frac[local_ind])
    t, p = kendalltau(list1, list2)
    kendalltau_vec.append(t)
    print t
    print list1
    print list2
    if t == 1.0:
        selected_counter+=1
    if np.isnan(t) == False:
        useful_counter+=1

print "total number of genes", counter
print "number of genes with more than >1 sites",useful_counter
print "exactly correct genes", selected_counter

print np.nanmean(kendalltau_vec)

