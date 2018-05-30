import numpy as np
import glob
import pickle
import csv

def preprocess_indel_files(count_data_folder):
  #count_data_folder = data_folder + "sample_counts/"
  #prop_data_folder = data_folder + "sample_props/"
  #count_data_folder = data_folder + "counts/"
  # First process the files to glean the names of the genes and the different indels
  name_genes = []
  name_genes_grna = []
  name_indel_type = []
  for each_file in glob.glob(count_data_folder + "counts-*.txt"):
    with open(each_file) as f:
      i = 0
      process_file = False
      add_file = False
      for line in f:
        line = line.replace('\n', '')
        line = line.replace('_', '-')
        if i == 0:
          line = line.replace('"', '')
          l = line.split(',')
          process_file = True
          curr_gene_name = each_file[len(count_data_folder) + 7:-4].split('-')[0]
          curr_gene_grna_name = []
          for patient in range(np.size(l)):
            curr_gene_grna_name.append("%s-%s-%s" %(curr_gene_name,l[patient].split('-')[1],l[patient].split('-')[2] ))
        if i > 0 and process_file:
          l_indel = line.split('"')[1].split(',')
          l = line.split('"')[2].split(',')[1:]
          indel_type = ''
          # Some positions are of the form: "-23:-21D,-19:-15D", which get split by the process when we call split()
          # We try to account for such things in this space
          for j in range(np.size(l_indel)):
            indel_type += l_indel[j]
          # We only consider I or D
          if line.find('I') != -1 or line.find('D') != -1:
            name_indel_type.append(indel_type)
            if not add_file:
              name_genes.append(curr_gene_name)
              for patient in range(np.size(curr_gene_grna_name)):
                name_genes_grna.append(curr_gene_grna_name[patient])
              add_file = True
        i += 1

  # Take the unique values, in sorted order
  name_genes_unique = list(set(name_genes))
  name_genes_grna_unique = list(set(name_genes_grna))
  name_indel_type_unique = list(set(name_indel_type))
  name_genes_unique.sort()
  name_genes_grna_unique.sort()
  name_indel_type_unique.sort()

  ##
  # Then process the files again to get the actual counts from only the desired files, and from the desired rows and columns
  indel_count_matrix = np.zeros((len(name_indel_type_unique), len(name_genes_grna_unique)))
  length_indel_insertion = np.zeros(len(name_indel_type_unique), dtype = int)
  length_indel_deletion = np.zeros(len(name_indel_type_unique), dtype=int)
  no_variant_vec = np.zeros(len(name_genes_grna_unique))
  other_vec = np.zeros(len(name_genes_grna_unique))
  snv_vec = np.zeros(len(name_genes_grna_unique))
  hdr_vec = np.zeros(len(name_genes_grna_unique))

  for each_file in glob.glob(count_data_folder + "counts-*.txt"):
    print each_file
    with open(each_file) as f:
      i = 0
      process_file = False
      for line in f:
        line = line.replace('\n', '')
        line = line.replace('_', '-')
        if i == 0:
          line = line.replace('"', '')
          l = line.split(',')
          curr_gene_name = each_file[len(count_data_folder) + 7:-4].split('-')[0]
          col_index = []
          if "%s-%s-%s" %(curr_gene_name,l[0].split('-')[1],l[0].split('-')[2] )  in name_genes_grna_unique:
            process_file = True
            for patient in range(np.size(l)):
              col_index.append(name_genes_grna_unique.index("%s-%s-%s" %(curr_gene_name,l[patient].split('-')[1],l[patient].split('-')[2])))
        if i > 0 and process_file:
          l_indel = line.split('"')[1].split(',')
          l = line.split('"')[2].split(',')[1:]
          indel_type = ''
          len_indel_insertion = 0
          len_indel_deletion = 0
          # Some positions are of the form: "-23:-21D,-19:-15D", which get split by the process when we call split()
          # We try to account for such things in this space
          for j in range(0, np.size(l_indel)):
            indel_type += l_indel[j]
            if l_indel[j].find('I') != -1:
              begn_size = l_indel[j].replace("I", "")
              begn_size = begn_size.split(':')
              len_indel_insertion += int(begn_size[1])
            if l_indel[j].find('D') != -1:
              begn_size = l_indel[j].replace("D", "")
              begn_size = begn_size.split(':')
              len_indel_deletion += int(begn_size[1])
          # We ignore SNV, others, and no variants
          if line.find('I') != -1 or line.find('D') != -1:
            row_index = name_indel_type_unique.index(indel_type)
            length_indel_insertion[row_index] = len_indel_insertion
            length_indel_deletion[row_index] = len_indel_deletion
            for j in range(np.size(l)):
              if l[j] != 'NA':
                indel_count_matrix[row_index,col_index[j]] = float(l[j])
                #print row_index,col_index[j]
          if 'variant' in line:
            for j in range(np.size(l)):
              if l[j] != 'NA':
                no_variant_vec[col_index[j]] += float(l[j])

          if 'Other' in line:
            for j in range(np.size(l)):
              if l[j] != 'NA':
                other_vec[col_index[j]] += float(l[j])

          if 'SNV' in line:
            for j in range(np.size(l)):
              if l[j] != 'NA':
                snv_vec[col_index[j]] += float(l[j])

          if 'SNV:4,5,6' in line:
            for j in range(np.size(l)):
              if l[j] != 'NA':
                hdr_vec[col_index[j]] += float(l[j])



        i += 1

  # # finding the index for the indels with frequency of mutatnt reads < 0.01
  # rare_indel_index = []
  # indel_frac_mutant_read_matrix = indel_count_matrix / np.reshape(np.sum(indel_count_matrix, axis=0), (1, -1))
  # for row_index in range(np.shape(indel_frac_mutant_read_matrix)[0]):
  #   if max(indel_frac_mutant_read_matrix[row_index]) < 0.01:
  #     rare_indel_index.append(row_index)


  # ##
  # # Process the proportions file to get the proportions data
  # indel_prop_matrix = np.zeros((len(name_indel_type_unique), len(name_genes_grna_unique)))
  # for each_file in glob.glob(prop_data_folder + "proportions-*.txt"):
  #   #print each_file
  #   with open(each_file) as f:
  #     i = 0
  #     process_file = False
  #     for line in f:
  #       line = line.replace('\n', '')
  #       line = line.replace('_', '-')
  #       if i == 0:
  #         line = line.replace('"', '')
  #         l = line.split(',')
  #         curr_gene_name = each_file[len(prop_data_folder) + 12:-4].split('-')[0]
  #         col_index = []
  #         if "%s-%s-%s" % (curr_gene_name, l[0].split('-')[1], l[0].split('-')[2]) in name_genes_grna_unique:
  #           process_file = True
  #           for patient in range(np.size(l)):
  #             col_index.append(name_genes_grna_unique.index(
  #               "%s-%s-%s" % (curr_gene_name, l[patient].split('-')[1], l[patient].split('-')[2])))
  #       if i > 0 and process_file:
  #         l_indel = line.split('"')[1].split(',')
  #         l = line.split('"')[2].split(',')[1:]
  #         indel_type = ''
  #         len_indel = 0
  #         # Some positions are of the form: "-23:-21D,-19:-15D", which get split by the process when we call split()
  #         # We try to account for such things in this space
  #         for j in range(0, np.size(l_indel)):
  #           indel_type += l_indel[j]
  #         # We ignore SNV, others, and no variants
  #         if line.find('I') != -1 or line.find('D') != -1:
  #           row_index = name_indel_type_unique.index(indel_type)
  #           for j in range(np.size(l)):
  #             if l[j] != 'NA':
  #               indel_prop_matrix[row_index, col_index[j]] = float(l[j])
  #               #print row_index, col_index[j]
  #
  #       i += 1


  # ######
  # ###### here we filter out all indels with mutant read frequency less than 0.01
  # ######
  # name_indel_type_unique = np.delete(name_indel_type_unique, rare_indel_index).tolist()
  # indel_count_matrix = np.delete(indel_count_matrix, rare_indel_index, 0)
  # indel_prop_matrix = np.delete(indel_prop_matrix, rare_indel_index, 0)
  # length_indel_insertion = np.delete(length_indel_insertion, rare_indel_index, 0)
  # length_indel_deletion = np.delete(length_indel_deletion, rare_indel_index, 0)
  # ######


  ######
  ###### here we filter out all outcomes with very small read counts
  ######
  low_read_index = []
  low_read_patients = ['HSPH1-00018-J05', 'HAT1-00022-O21', 'ATP5D-00029-A17' , 'XPO5-00029-E01' , 'PON2-00019-K20']
  for crispr in range(np.shape(indel_count_matrix)[1]):
    if sum(indel_count_matrix[:,crispr]) < 2000 or (name_genes_grna_unique[crispr] in low_read_patients):
      low_read_index.append(crispr)

  indel_count_matrix = np.delete(indel_count_matrix, low_read_index, 1)
  #indel_prop_matrix = np.delete(indel_prop_matrix, low_read_index, 1)
  name_genes_grna_unique = list(np.delete(name_genes_grna_unique, low_read_index, 0))

  no_variant_vec = np.delete(no_variant_vec, low_read_index)
  other_vec = np.delete(other_vec, low_read_index)
  snv_vec = np.delete(snv_vec, low_read_index)
  hdr_vec = np.delete(hdr_vec, low_read_index)

  ######





  return name_genes_unique, name_genes_grna_unique, name_indel_type_unique, indel_count_matrix, no_variant_vec, other_vec, snv_vec,hdr_vec,  length_indel_insertion, length_indel_deletion
  #return name_genes_unique, name_genes_grna_unique, name_indel_type_unique , indel_count_matrix

#name_genes_unique, name_genes_grna_unique, name_indel_type_unique, indel_count_matrix, no_variant_vec, other_vec, snv_vec, hdr_vec,  length_indel_insertion, length_indel_deletion =  preprocess_indel_files('/Users/amirali/Projects/HDR/HDR/')


# # do this
# spacer_dict = {}
# with open('sequence_pam_gene_grna_big_file_donor_genomic_context_hdr.csv', 'rb') as csvfile:
#   spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#   row_counter = 0
#   for row in spamreader:
#     spacer_dict[row[0].split(',')[0]] = row[0].split(',')[2]+row[0].split(',')[3]
#
# spacer_pam_list_ALL = []
# for site_name in name_genes_grna_unique:
#   site_name_list = site_name.split('-')
#   spacer = spacer_dict[site_name_list[1] + '-' + site_name_list[2]]
#   print spacer
#   spacer_pam_list_ALL.append(spacer)
#
# pickle.dump(spacer_pam_list_ALL, open('HDR/spacer_pam_list_ALL.p', 'wb'))
# pickle.dump(name_genes_grna_unique, open('HDR/name_genes_grna_ALL.p', 'wb'))
# pickle.dump(name_indel_type_unique, open('HDR/name_indel_type_ALL.p', 'wb'))
# pickle.dump(indel_count_matrix, open('HDR/indel_count_matrix_ALL.p', 'wb'))
# pickle.dump(no_variant_vec, open('HDR/no_variant_vec_ALL.p', 'wb'))
# pickle.dump(other_vec, open('HDR/other_vec_ALL.p', 'wb'))
# pickle.dump(snv_vec, open('HDR/snv_vec_ALL.p', 'wb'))
# pickle.dump(hdr_vec, open('HDR/hdr_vec_ALL.p', 'wb'))
# pickle.dump(length_indel_insertion, open('HDR/length_indel_insertion_ALL.p', 'wb'))
# pickle.dump(length_indel_deletion, open('HDR/length_indel_deletion_ALL.p', 'wb'))


# ## then this
# name_genes_grna_unique_ALL = pickle.load(open('HDR/name_genes_grna_ALL.p', 'rb'))
# name_indel_type_unique_ALL = pickle.load(open('HDR/name_indel_type_ALL.p', 'rb'))
# indel_count_matrix_ALL = pickle.load(open('HDR/indel_count_matrix_ALL.p', 'rb'))
# no_variant_vec_ALL = pickle.load(open('HDR/no_variant_vec_ALL.p', 'rb'))
# other_vec_ALL = pickle.load(open('HDR/other_vec_ALL.p', 'rb'))
# snv_vec_ALL = pickle.load(open('HDR/snv_vec_ALL.p', 'rb'))
# hdr_vec_ALL = pickle.load(open('HDR/hdr_vec_ALL.p', 'rb'))
# length_indel_insertion_ALL = pickle.load(open('HDR/length_indel_insertion_ALL.p', 'rb'))
# length_indel_deletion_ALL = pickle.load(open('HDR/length_indel_deletion_ALL.p', 'rb'))
# spacer_pam_list_ALL = pickle.load(open('HDR/spacer_pam_list_ALL.p', 'rb'))
#
#
# #
# name_genes_grna_unique = pickle.load(open('Tcell-files/name_genes_grna_UNIQUE.p', 'rb'))
# spacers = pickle.load(open('Tcell-files/spacer_pam_list_UNIQUE.p', 'rb'))
#
# ccc = 0
# to_delete = []
# for counter1,spacer1 in enumerate(spacer_pam_list_ALL):
#   flag = 0
#   for counter2,spacer2 in enumerate(spacers):
#     if spacer1 == spacer2:
#       flag = 1
#   if flag==0:
#     to_delete.append(counter1)
#
# indel_count_matrix_ALL = np.delete(indel_count_matrix_ALL, to_delete, 1)
# name_genes_grna_unique_ALL = list(np.delete(name_genes_grna_unique_ALL, to_delete, 0))
# no_variant_vec_ALL = np.delete(no_variant_vec_ALL, to_delete)
# other_vec_ALL = np.delete(other_vec_ALL, to_delete)
# snv_vec_ALL = np.delete(snv_vec_ALL, to_delete)
# hdr_vec_ALL = np.delete(hdr_vec_ALL, to_delete)
# spacer_pam_list_ALL = list(np.delete(spacer_pam_list_ALL, to_delete, 0))
#
# pickle.dump(name_genes_grna_unique_ALL, open('HDR/name_genes_grna_ALL_matched.p', 'wb'))
# pickle.dump(indel_count_matrix_ALL, open('HDR/indel_count_matrix_ALL_matched.p', 'wb'))
# pickle.dump(no_variant_vec_ALL, open('HDR/no_variant_vec_ALL_matched.p', 'wb'))
# pickle.dump(other_vec_ALL, open('HDR/other_vec_ALL_matched.p', 'wb'))
# pickle.dump(snv_vec_ALL, open('HDR/snv_vec_ALL_matched.p', 'wb'))
# pickle.dump(hdr_vec_ALL, open('HDR/hdr_vec_ALL_matched.p', 'wb'))
# pickle.dump(spacer_pam_list_ALL, open('HDR/spacer_pam_list_ALL_matched.p', 'wb'))

## and finally this
name_genes_grna_unique = pickle.load(open('HDR/name_genes_grna_ALL_matched.p', 'rb'))
name_indel_type_unique = pickle.load(open('HDR/name_indel_type_ALL.p', 'rb'))
indel_count_matrix = pickle.load(open('HDR/indel_count_matrix_ALL_matched.p', 'rb'))
no_variant_vec = pickle.load(open('HDR/no_variant_vec_ALL_matched.p', 'rb'))
other_vec = pickle.load(open('HDR/other_vec_ALL_matched.p', 'rb'))
snv_vec = pickle.load(open('HDR/snv_vec_ALL_matched.p', 'rb'))
hdr_vec = pickle.load(open('HDR/snv_vec_ALL_matched.p', 'rb'))
length_indel_insertion = pickle.load(open('HDR/length_indel_insertion_ALL.p', 'rb'))
length_indel_deletion = pickle.load(open('HDR/length_indel_deletion_ALL.p', 'rb'))
spacer_pam_list = pickle.load(open('HDR/spacer_pam_list_ALL_matched.p', 'rb'))

spacers = pickle.load(open('Tcell-files/spacer_pam_list_UNIQUE.p', 'rb'))


print np.shape(name_genes_grna_unique)
print np.shape(name_indel_type_unique)
print np.shape(indel_count_matrix)
print np.shape(no_variant_vec)
print np.shape(other_vec)
print np.shape(snv_vec)
print np.shape(hdr_vec)
print np.shape(spacer_pam_list)

HDR_TCELL_matching_vector = []
for spacer1 in spacer_pam_list:
  for counter,spacer2 in enumerate(spacers):
    if spacer1==spacer2:
      HDR_TCELL_matching_vector.append(counter)



print HDR_TCELL_matching_vector
print len(HDR_TCELL_matching_vector)
print len(set(HDR_TCELL_matching_vector))

pickle.dump(HDR_TCELL_matching_vector, open('HDR/HDR_TCELL_matching_vector.p', 'wb'))