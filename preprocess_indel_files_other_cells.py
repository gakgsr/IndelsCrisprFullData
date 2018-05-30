import numpy as np
from os import listdir
import re
import csv
import pickle
from scipy.stats import entropy
import numbers
import matplotlib.pyplot as plt

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

def load_gene_sequence(spacer_list):

  sequence_pam_per_gene_grna = np.zeros((len(spacer_list), 23, 4), dtype = bool)

  for counter,spacer in enumerate(spacer_list):
    for i in range(23):
      sequence_pam_per_gene_grna[counter, i, one_hot_index(spacer[i])] = 1

  return np.reshape(sequence_pam_per_gene_grna, (len(sequence_pam_per_gene_grna), -1))


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


def homology_matrix_finder(spacer_list):
    homology_matrix = np.zeros((4, len(spacer_list)))
    for counter,spacer in enumerate(spacer_list):
        nuc_count = 0
        for nuc in ['A', 'C', 'G', 'T']:
            homology_matrix[nuc_count,counter] = int(longest_substring_passing_cutsite(spacer[16-3:16+3], nuc))
            nuc_count+=1
    return homology_matrix

def flatness_finder(indel_count_matrix):
    num_indels, num_sites = np.shape(indel_count_matrix)
    indel_fraction_mutant_matrix = indel_count_matrix / np.reshape(np.sum(indel_count_matrix, axis=0), (1, -1))
    max_grad = []
    entrop = []
    for col in range(num_sites):
        vec = np.copy(indel_fraction_mutant_matrix[:, col])
        vec = np.sort(vec)[::-1]
        max_grad.append(max(abs(np.gradient(vec))))
        entrop.append(entropy(vec))

    return np.asarray(max_grad),np.asarray(entrop)

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

  help_vec = np.sum(insertion_only_fraction_matrix, axis=0)
  help_vec[help_vec==0] = 'nan'

  insertion_only_fraction_matrix = insertion_only_fraction_matrix / np.reshape(help_vec, (1, -1))
  deletion_only_fraction_matrix = deletion_only_fraction_matrix / np.reshape(np.sum(deletion_only_fraction_matrix, axis=0), (1, -1))


  for site_index in range(site_num):
    exp_insertion_length[site_index] = np.inner(length_indel_insertion,insertion_only_fraction_matrix[:,site_index])
    exp_deletion_length[site_index] = np.inner(length_indel_deletion, deletion_only_fraction_matrix[:, site_index])

  # some sites do not get any insertions. this cuase nan. we make those entries zero.
  for i in range(np.size(exp_insertion_length)):
    if np.isnan(exp_insertion_length[i]):
      exp_insertion_length[i] = 0

  return exp_insertion_length,exp_deletion_length


def preprocess_other_cells():
  all_indel_names = []
  all_file_name = []
  #def preprocess_indel_files(data_folder):
  data_folder =  "/Users/amirali/Projects/OtherCellTypes/othercells/"
  #count_data_folder = data_folder + "counts-small/"
  count_data_folder = data_folder + "counts/"
  number_of_sites = 0
  for file_name in listdir(count_data_folder):
    file = open(count_data_folder+file_name,'r')
    for line_counter,line in enumerate(file):
      line = line.replace("\n","")
      line = line.replace('"', '')
      if line_counter==0:
        all_file_name.append(line)
      if 'I' in line or 'D' in line:  # this is an indel line
        indel_name_list = line.split(',')
        indel_name = []
        for partial_indel_name in indel_name_list[0:-1]:
          indel_name +=partial_indel_name
        indel_name = ''.join(indel_name)
        all_indel_names.append(indel_name)
    number_of_sites+=1

  all_indel_names = list(set(all_indel_names))
  all_indel_names.sort()
  all_file_name.sort()

  indel_count_matrix = np.zeros((len(all_indel_names), len(all_file_name)))
  no_variant_vector = np.zeros(len(all_file_name))
  other_vector = np.zeros(len(all_file_name))
  length_indel_insertion = np.zeros(len(all_indel_names), dtype=int)
  length_indel_deletion = np.zeros(len(all_indel_names), dtype=int)

  for file_name in listdir(count_data_folder):
    file = open(count_data_folder+file_name,'r')
    for line_counter,line in enumerate(file):
      line = line.replace("\n","")
      line = line.replace('"', '')
      if line_counter == 0:
        col = all_file_name.index(line)
      if 'I' in line or 'D' in line:  # this is an indel line
        indel_name_list = line.split(',')
        indel_name = []
        for partial_indel_name in indel_name_list[0:-1]:
          indel_name +=partial_indel_name
        indel_name = ''.join(indel_name)
        row = all_indel_names.index(indel_name)
        indel_count_matrix[row,col] = int(indel_name_list[-1])
      if 'no variant' in line:
        no_variant_vector[col] =int(line.split(',')[1])
      if 'Other' in line:
        other_vector[col] =int(line.split(',')[1])


  all_file_name_trimed = []
  for filename in all_file_name:
    all_file_name_trimed.append(filename.split('_')[1]+'_'+filename.split('_')[3])

  u, groupings = np.unique(all_file_name_trimed, return_inverse=True)

  selected_index = []
  for group in range(0,max(groupings)):
    triplets = np.where(groupings == group)[0]
    selected_index.append(triplets[np.argmax(np.sum(indel_count_matrix[:,triplets],axis=0))])


  indel_count_matrix = indel_count_matrix[:,selected_index]
  no_variant_vector = no_variant_vector[selected_index]
  other_vector = other_vector[selected_index]
  all_file_name = list(np.asarray(all_file_name)[selected_index])

  # get rid of zero entries
  zero_index = np.where(np.sum(indel_count_matrix, axis=0) + other_vector + no_variant_vector == 0)[0]
  indel_count_matrix = np.delete(indel_count_matrix, zero_index, 1)
  other_vector = np.delete(other_vector, zero_index, None)
  no_variant_vector = np.delete(no_variant_vector, zero_index, None)
  all_file_name = np.delete(all_file_name, zero_index, None)


  # here we are done with the count matrices
  # we do the insertion deletion length thing


  length_indel_insertion = np.zeros(len(all_indel_names))
  length_indel_deletion = np.zeros(len(all_indel_names))
  for indel_counter,indel in enumerate(all_indel_names):
    indel_locations = re.split('I|D',indel)[:-1]
    indel_types = ''.join(c for c in indel if (c == 'I' or c == 'D'))
    for i in range(len(indel_types)):
      if indel_types[i] == 'D':
        start, size = indel_locations[i].split(':')
        length_indel_deletion[indel_counter] += int(size)
      if indel_types[i] == 'I':
        start, size = indel_locations[i].split(':')
        length_indel_insertion[indel_counter] += int(size)



  return indel_count_matrix,no_variant_vector,other_vector,all_file_name,all_indel_names,length_indel_deletion,length_indel_insertion


def insertion_matrix_finder():

  insertion_file_names = []
  insertion_matrix = np.zeros((4,96*3+95*3+93+96+96))
  folder = '/Users/amirali/Projects/ins-othercells/'
  file_counter = 0
  for cell in listdir(folder):
    if cell != ".DS_Store":
      for file in listdir(folder+cell+'/'):

        insertion_file_names.append(file[11:-4].split('-')[0]+'-'+file[11:-4].split('-')[1])

        lines = open(folder+cell+'/'+ file, 'r')
        for line_counter, line in enumerate(lines):
          if line_counter>0:
              line = line[:-1].replace('"','')
              line_splited = line.split(',')
              counter = float(line_splited[-1])
              seq = line_splited[1]
              cigar = line_splited[2:-4]
              if len(cigar)==1 and seq in ['A','T','C','G'] and isinstance(counter, numbers.Number):
                insertion_matrix[one_hot_index(seq),file_counter ] += counter

        file_counter += 1

  return insertion_file_names,insertion_matrix


def insertion_matrix_finder_ryan():
  insertion_file_names = []
  insertion_matrix = np.zeros((4,96*3+95*3+93+96+96))
  folder = '/Users/amirali/Projects/ins-othercells/'
  file_counter = 0
  for cell in listdir(folder):
    if cell != ".DS_Store":
      for file in listdir(folder+cell+'/'):

        insertion_file_names.append(file[11:-4].split('-')[0]+'-'+file[11:-4].split('-')[1])

        lines = open(folder+cell+'/'+ file, 'r')
        for line_counter, line in enumerate(lines):
          if line_counter>0:
              line = line[:-1].replace('"','')
              line_splited = line.split(',')
              counter = float(line_splited[-1])
              seq = line_splited[1]
              cigar = line_splited[2:-4]
              if len(cigar)==1 and cigar[0]=='-3:1I' and  seq in ['A','T','C','G'] and isinstance(counter, numbers.Number):
                insertion_matrix[one_hot_index(seq),file_counter ] += counter

        file_counter += 1

  return insertion_file_names,insertion_matrix



def rare_insertion():
    insertion_file_names = []
    global_insertion_list = []
    folder = '/Users/amirali/Projects/ins-othercells/'
    file_counter = 0
    for cell in listdir(folder):
        if cell != ".DS_Store":
            for file in listdir(folder + cell + '/'):
                local_insertion_list = []
                insertion_file_names.append(file[11:-4].split('-')[0] + '-' + file[11:-4].split('-')[1])


                lines = open(folder + cell + '/' + file, 'r')
                for line_counter, line in enumerate(lines):
                    if line_counter > 0:
                        line = line[:-1].replace('"', '')
                        line_splited = line.split(',')
                        counter = float(line_splited[-1])
                        seq = line_splited[1]
                        cigar = line_splited[2:-4]
                        if len(cigar) == 1:
                            size = int(cigar[0].split(":")[1][0:-1])
                            if size > 10 and isinstance(counter, numbers.Number):
                                # print seq
                                # print line
                                # print counter
                                #local_insertion_list[samples.index(sample)].append(seq)
                                local_insertion_list.append(seq)


                file_counter += 1
                global_insertion_list.append(local_insertion_list)


    return global_insertion_list


# indel_count_matrix,no_variant_vector,other_vector,all_file_name,all_indel_names,length_indel_deletion,length_indel_insertion = preprocess_other_cells()
#
# # output is ready
# # find eff_vec
# total_vec = np.sum(indel_count_matrix,axis=0) + no_variant_vector
# eff_vec = np.sum(indel_count_matrix,axis=0) / total_vec
#
# # idel prop matrix
# indel_prop_matrix = np.zeros(np.shape(indel_count_matrix))
# indel_prop_matrix = indel_count_matrix / np.reshape(total_vec, (1, -1))
#
# # fraction of Indels
# fraction_insertions, fraction_deletions = fraction_of_deletion_insertion(indel_count_matrix,length_indel_insertion,length_indel_deletion)
#
# # fraction of indels (total)
# fraction_insertions_all, fraction_deletions_all = fraction_of_deletion_insertion_porportion(indel_prop_matrix,length_indel_insertion,length_indel_deletion)
#
#
# # flatness
# max_grad,entrop = flatness_finder(indel_count_matrix)
#
# # expected number of indels
# exp_insertion_length, exp_deletion_length = expected_deletion_insertion_length(indel_count_matrix,length_indel_insertion,length_indel_deletion)
#
# # input
# spcer_dict={}
# with open('/Users/amirali/Projects/OtherCellTypes/hg19seq.csv', 'rb') as csvfile:
#   spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#   row_counter = 0
#   for row in spamreader:
#     #print row[0].split(',')[6]
#     spcer_dict[row[0].split(',')[6]] = row[0].split(',')[5]
#
# spacer_list = []
# for site in all_file_name:
#   site_list = site.split('_')
#   site = ''.join(site_list[0]+'_'+site_list[1]+'_'+ site_list[2]+'_K562_'+site_list[4]+ '_R1.bam')
#   spacer_list.append(spcer_dict[site])
#
#
# # homopolymer matrix
# homology_matrix = homology_matrix_finder(spacer_list)
#
# # one-hot encoded
# spacer_pam_per_site_one_hot = load_gene_sequence(spacer_list)
#
# # cell-type code
# cell_type_vector = []
# for name in all_file_name:
#   cell_type_vector.append(name.split('_')[3])
# cell_type_vector = np.asarray(cell_type_vector)
#
#
# #insertion matrix
# insertion_matrix = np.zeros((4,len(all_file_name)))
# insertion_file_names,insertion_matrix_all = insertion_matrix_finder_ryan()
# for file_count,file in enumerate(all_file_name):
#   insertion_matrix[:,file_count] = insertion_matrix_all[:,insertion_file_names.index(file)]
#   # if np.sum(insertion_matrix_all[:,insertion_file_names.index(file)]) == 0:
#   #   insertion_matrix[:, file_count] = np.ones(4)


global_insertion_list = rare_insertion()
pickle.dump(global_insertion_list, open('storage_other_cell/rare_insertions.p', 'wb'))


#
# pickle.dump(spacer_list, open('storage_other_cell/spacer_list.p', 'wb'))
# pickle.dump(eff_vec, open('storage_other_cell/eff_vec.p', 'wb'))
# pickle.dump(fraction_insertions, open('storage_other_cell/fraction_insertions.p', 'wb'))
# pickle.dump(fraction_deletions, open('storage_other_cell/fraction_deletions.p', 'wb'))
# pickle.dump(fraction_insertions_all, open('storage_other_cell/fraction_insertions_over_total.p', 'wb'))
# pickle.dump(fraction_deletions_all, open('storage_other_cell/fraction_deletions_over_total.p', 'wb'))
# pickle.dump(exp_insertion_length, open('storage_other_cell/exp_insertion_length.p', 'wb'))
# pickle.dump(exp_deletion_length, open('storage_other_cell/exp_deletion_length.p', 'wb'))
# pickle.dump(max_grad, open('storage_other_cell/max_grad.p', 'wb'))
# pickle.dump(entrop, open('storage_other_cell/entrop.p', 'wb'))
# pickle.dump(spacer_list, open('storage_other_cell/spacer_list.p', 'wb'))
# pickle.dump(homology_matrix, open('storage_other_cell/homology_matrix.p', 'wb'))
# pickle.dump(spacer_pam_per_site_one_hot, open('storage_other_cell/spacer_pam_per_site_one_hot.p', 'wb'))
# pickle.dump(cell_type_vector, open('storage_other_cell/cell_type_vector.p', 'wb'))
#
#
# pickle.dump(indel_count_matrix, open('storage_other_cell/indel_count_matrix.p', 'wb'))
# pickle.dump(indel_prop_matrix, open('storage_other_cell/indel_prop_matrix.p', 'wb'))
# pickle.dump(no_variant_vector, open('storage_other_cell/no_variant_vector.p', 'wb'))
# pickle.dump(other_vector, open('storage_other_cell/other_vector.p', 'wb'))
# pickle.dump(all_file_name, open('storage_other_cell/all_file_name.p', 'wb'))
# pickle.dump(all_indel_names, open('storage_other_cell/all_indel_names.p', 'wb'))
# pickle.dump(length_indel_deletion, open('storage_other_cell/length_indel_deletion.p', 'wb'))
# pickle.dump(length_indel_insertion, open('storage_other_cell/length_indel_insertion.p', 'wb'))
# pickle.dump(insertion_matrix, open('storage_other_cell/insertion_matrix.p', 'wb'))
#
#
# insertion_matrix = insertion_matrix / np.reshape(np.sum(insertion_matrix, axis=0), (1, -1))
# insertion_matrix = np.nanmean(insertion_matrix,axis=1)
# plt.stem(insertion_matrix)
# plt.savefig('storage_other_cell/insertion_hist.pdf')
# print insertion_matrix
