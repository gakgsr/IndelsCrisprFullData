from preprocess_indel_files import preprocess_indel_files
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import pickle

def load_gene_sequence(sequence_file_name, name_genes_grna_unique):
  # Create numpy matrix of size len(name_genes_grna_unique) * 23 * 4, to store the sequence as one-hot encoded
  sequence_pam_per_gene_grna = np.zeros((len(name_genes_grna_unique), 23, 4), dtype = bool)
  # Obtain the grna and PAM sequence corresponding to name_genes_grna_unique
  pam_per_gene_grna = {}
  sequence_per_gene_grna = {}
  # Obtain the PAM and grna sequence corresponding to each input file
  with open(sequence_file_name) as f:
    for line in f:
      line = line.replace('"', '')
      line = line.replace(' ', '')
      line = line.replace('\n', '')
      l = line.split(',')
      if l[1] + '-' + l[0] in name_genes_grna_unique:
        index_in_name_genes_grna_unique = name_genes_grna_unique.index(l[1] + '-' + l[0])
        sequence_per_gene_grna[index_in_name_genes_grna_unique] = l[2]
        pam_per_gene_grna[index_in_name_genes_grna_unique] = l[3]
  # Convert the above dictionaries to lists, for ease of working downstream
  pam_per_gene_grna_list = []
  sequence_per_gene_grna_list = []
  for i in range(len(name_genes_grna_unique)):
    pam_per_gene_grna_list.append(pam_per_gene_grna[i])
    sequence_per_gene_grna_list.append(sequence_per_gene_grna[i])
  return sequence_per_gene_grna_list, pam_per_gene_grna_list


#data_folder = "../IndelsData/"
#sequence_file_name = "sequence_pam_gene_grna.csv"
sequence_file_name = "sequence_pam_gene_grna_big_file.csv"
#data_folder = "/Users/amirali/Projects/CRISPR-data/R data/AM_TechMerg_Summary/"
data_folder = "/Users/amirali/Projects/CRISPR-data-Feb18/20nt_counts_only/"
#name_genes_unique, name_genes_grna_unique, name_indel_type_unique, indel_count_matrix, indel_prop_matrix, length_indel = preprocess_indel_files(data_folder)
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
print np.shape(indel_count_matrix)

sequence_per_gene_grna, pam_per_gene_grna = load_gene_sequence(sequence_file_name, name_genes_grna_unique)
print "load gene sequence is done!"


# Compute the unique PAM and cleavage words
unique_pam = list(set(pam_per_gene_grna))
unique_sequence = []
for i in range(len(sequence_per_gene_grna)):
  unique_sequence.append(sequence_per_gene_grna[i][16] + sequence_per_gene_grna[i][17])
unique_sequence = list(set(unique_sequence))

print unique_sequence

# Compute the proportions of insertions and deletions in each file
prop_insertions_gene_grna = np.zeros(len(name_genes_grna_unique), dtype = float)
prop_deletions_gene_grna = np.zeros(len(name_genes_grna_unique), dtype = float)
for i in range(len(name_genes_grna_unique)):
  for j in range(indel_prop_matrix.shape[0]):
    # across repeats
    if name_indel_type_unique[j].find('I') != -1:
      # Computes the average insertion. Comment the multiplication to compute the average proportion
      prop_insertions_gene_grna[i] += np.mean(indel_prop_matrix[j][i], dtype = float)*length_indel[j]
    if name_indel_type_unique[j].find('D') != -1:
      # Computes the average deletion. Comment the multiplication to compute the average proportion
      prop_deletions_gene_grna[i] += np.mean(indel_prop_matrix[j][i], dtype = float)*length_indel[j]

print "computing the proportion of insertions and deletion in each file is done!"

# Compute the sums for each unique pam and sequence
unique_pam_ins = [[] for _ in range(len(unique_pam))]
unique_pam_del = [[] for _ in range(len(unique_pam))]
unique_seq_ins = [[] for _ in range(len(unique_sequence))]
unique_seq_del = [[] for _ in range(len(unique_sequence))]


for i in range(len(name_genes_grna_unique)):
  pam_index = unique_pam.index(pam_per_gene_grna[i])
  sequence_index = unique_sequence.index(sequence_per_gene_grna[i][16] + sequence_per_gene_grna[i][17])

  unique_pam_ins[pam_index].append(prop_insertions_gene_grna[i])
  unique_pam_del[pam_index].append(prop_deletions_gene_grna[i])
  unique_seq_ins[sequence_index].append(prop_insertions_gene_grna[i])
  unique_seq_del[sequence_index].append(prop_deletions_gene_grna[i])


# Print the averages
for i in range(len(unique_pam)):
  print unique_pam[i]
  print np.mean(unique_pam_ins[i])
  print np.std(unique_pam_ins[i])
  print np.mean(unique_pam_del[i])
  print np.std(unique_pam_del[i])
  print "number of outcomes ", np.size(unique_pam_ins[i])
  print "--"

for i in range(len(unique_sequence)):
  print unique_sequence[i]
  print np.mean(unique_seq_ins[i])
  print np.std(unique_seq_ins[i])
  print np.mean(unique_seq_del[i])
  print np.std(unique_seq_del[i])
  print "number of outcomes ", np.size(unique_seq_ins[i])
  print "--"