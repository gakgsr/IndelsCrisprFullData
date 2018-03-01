import numpy as np
import glob
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Processes the indel count files
# Returns a matrix with columns being each column in a distinct file, rows being distinct indels
# Matrix contains the count of corresponding indels
def process_count_files(data_folder):
  # Names of the rows (the different indels)
  name_row = []
  # Names of the columns (the different file names appended with the column headers)
  name_col = []
  ##
  name_genes = []
  name_files = []
  ##
  # Access all the indel files
  # Read them to get a list of unique indels and the list of column names
  # We don't yet look into the numbers
  for out_file in glob.glob(data_folder + "counts-*.txt"):
    with open(out_file) as f:
      i = 0
      num_col_i = 0
      name_files.append(out_file[len(data_folder) + 7:-4])
      for line in f:
        total_count_i = 0
        line = line.replace('"', '')
        line = line.replace('\n', '')
        l = line.split(',')
        if i == 0:
          num_col_i = len(l)
          for j in range(len(l)):
            name_col_str = out_file[len(data_folder) + 7:-4] + '-' + l[j]
            name_col.append(name_col_str)
            name_gene_str = name_col_str.split('-')[0]
            name_genes.append(name_gene_str)
        else:
          row_name = ''
          # Some positions are of the form: "-23:-21D,-19:-15D", which get split by the process when we call split()
          # We try to account for such things in this space
          for j in range(0, len(l) - num_col_i):
            row_name += l[j]
          name_row.append(row_name)
        i += 1

  # Find unique indels and column headers
  unique_name_row = list(set(name_row))
  unique_name_col = list(set(name_col))
  unique_name_genes = list(set(name_genes))
  unique_name_row.sort()
  unique_name_col.sort()
  unique_name_genes.sort()
  name_files.sort()

  # Create a numpy matrix of size unique_name_row x unique_name_col of integers
  indel_count_matrix = np.zeros((len(unique_name_row), len(unique_name_col)), dtype = int)
  # Create a numpy vector of size unique_name_col to store number of insertions
  insertion_present = np.zeros(len(unique_name_col), dtype = int)
  # Create a numpy vector of size unique_name_col to store number of deletions
  deletion_present = np.zeros(len(unique_name_col), dtype = int)
  # Create a numpy vector of size unique_name_col to store number of SNV
  SNV_present = np.zeros(len(unique_name_col), dtype = int)
  # Create a numpy vector of size name_files to store number of insertions
  insertion_present_file = np.zeros(len(name_files), dtype = int)
  # Create a numpy vector of size name_files to store number of deletions
  deletion_present_file = np.zeros(len(name_files), dtype = int)
  # Create a numpy vector of size name_files to store number of SNV
  SNV_present_file = np.zeros(len(name_files), dtype = int)

  # Store the corresponding values from the files into indel_count_matrix
  # To do this, again access all the indel files, but also read the the indel count values this time
  for out_file in glob.glob(data_folder + "counts-*.txt"):
    with open(out_file) as f:
      i = 0
      file_col_name = []
      num_col_i = 0
      for line in f:
        line = line.replace('"', '')
        line = line.replace('\n', '')
        l = line.split(',')
        if i == 0:
          num_col_i = len(l)
          for j in range(len(l)):
            file_col_name.append(out_file[len(data_folder) + 7:-4] + '-' + l[j])
        else:
          row_name = ''
          # Some positions are of the form: "-23:-21D,-19:-15D", which get split by the process
          # We try to account for such things in this space
          for j in range(0, len(l) - num_col_i):
            row_name += l[j]
          for j in range(len(l) - num_col_i, len(l)):
            row_index = unique_name_row.index(row_name)
            col_index = unique_name_col.index(file_col_name[j - len(l) + num_col_i])
            # Extract count value
            if l[j] == 'NA':
              # Currently handling NA/nan values as -1. Can't find a numpy integer representation for NAN
              indel_count_matrix[row_index, col_index] = -1
            else:
              indel_count_matrix[row_index, col_index] = int(l[j])
        i += 1

  # Compute, by file and file + cell type, the number of types of indels above a threshold
  # To do this, again access all the indel files, but also read the the indel count values this time
  threshold = 5.0
  for out_file in glob.glob(data_folder + "proportions-*.txt"):
    with open(out_file) as f:
      i = 0
      file_col_name = []
      num_col_i = 0
      file_name_index = name_files.index(out_file[len(data_folder) + 12:-4])
      for line in f:
        line = line.replace('"', '')
        line = line.replace('\n', '')
        total_count_i = 0.0
        l = line.split(',')
        if i == 0:
          num_col_i = len(l)
          for j in range(len(l)):
            file_col_name.append(out_file[len(data_folder) + 12:-4] + '-' + l[j])
        else:
          row_name = ''
          # Some positions are of the form: "-23:-21D,-19:-15D", which get split by the process
          # We try to account for such things in this space
          for j in range(0, len(l) - num_col_i):
            row_name += l[j]
          for j in range(len(l) - num_col_i, len(l)):
            row_index = unique_name_row.index(row_name)
            col_index = unique_name_col.index(file_col_name[j - len(l) + num_col_i])
            # Extract count value
            if l[j] != 'NA' and float(l[j]) > threshold:
              # See if file has insertion, deletion, or SNV
              if row_name.find('SNV') != -1:
                SNV_present[col_index] += 1
              if row_name.find('I') != -1:
                insertion_present[col_index] += 1
              if row_name.find('D') != -1:
                deletion_present[col_index] += 1
            if l[j] != 'NA':
              total_count_i += float(l[j])
          if total_count_i/num_col_i > threshold:
            if row_name.find('SNV') != -1:
              SNV_present_file[file_name_index] += 1
            if row_name.find('I') != -1:
              insertion_present_file[file_name_index] += 1
            if row_name.find('D') != -1:
              deletion_present_file[file_name_index] += 1
        i += 1

  # Save the indel counts and row, column name information
  row_index_file = open('row_index.txt', 'w')
  for row_name_val in unique_name_row:
    row_index_file.write("%s\n" % row_name_val)
  col_index_file = open('column_index.txt', 'w')
  for col_name_val in unique_name_col:
    col_index_file.write("%s\n" % col_name_val)
  np.savetxt("indel_count_matrix.txt", indel_count_matrix)
  indel_type_file = open('indel_type_by_file_col.txt', 'w')
  indel_type_file.write("File_col_Name,Insertion_Count,Deletion_Count,SNV_Count\n")
  for i in range(len(unique_name_col)):
    indel_type_file.write("%s,%d,%d,%d\n" % (unique_name_col[i], insertion_present[i], deletion_present[i], SNV_present[i]))
  indel_type_file_only = open('indel_type_by_file.txt', 'w')
  indel_type_file_only.write("File_Name,Insertion_Count,Deletion_Count,SNV_Count\n")
  for i in range(len(name_files)):
    indel_type_file_only.write("%s,%d,%d,%d\n" % (name_files[i], insertion_present_file[i], deletion_present_file[i], SNV_present_file[i]))

  # Process to compute gene-wise count of insertions, deletions, and SNV and save
  insertion_present_gene = np.zeros(len(unique_name_genes), dtype = int)
  deletion_present_gene = np.zeros(len(unique_name_genes), dtype = int)
  SNV_present_gene = np.zeros(len(unique_name_genes), dtype = int)
  for i in range(len(unique_name_col)):
    gene_name = unique_name_col[i].split('-')[0]
    gene_name_index = unique_name_genes.index(gene_name)
    insertion_present_gene[gene_name_index] += insertion_present[i]
    deletion_present_gene[gene_name_index] += deletion_present[i]
    SNV_present_gene[gene_name_index] += SNV_present[i]
  indel_type_gene = open('indel_type_by_gene.txt', 'w')
  indel_type_gene.write("Gene_Name,Insertion_Count,Deletion_Count,SNV_Count\n")
  for i in range(len(unique_name_genes)):
    indel_type_gene.write("%s,%d,%d,%d\n" % (unique_name_genes[i], insertion_present_gene[i], deletion_present_gene[i], SNV_present_gene[i]))

  return indel_count_matrix, unique_name_row, unique_name_col


def analysis_count_files(indel_count_matrix, row_index, unique_name_col):
  [numberofindels,numberoffiles] = np.shape(indel_count_matrix)

  number_of_files_per_indel = []
  number_of_indels_per_file = []
  for i in range(numberofindels):
      number_of_files_per_indel.append(np.count_nonzero(indel_count_matrix[i]))

  for i in range(numberoffiles):
      number_of_indels_per_file.append(np.count_nonzero( np.transpose(indel_count_matrix)[i] ) )


  print "Number of Indel Types = ", np.size(number_of_files_per_indel)
  print "Number of Files = ", np.shape(indel_count_matrix)[1]

  print "Max Indel", row_index[np.argmax(number_of_files_per_indel)]
  print "Max Indel Occurance = ", max(number_of_files_per_indel)

  print "Min Indel", row_index[np.argmin(number_of_files_per_indel)]
  print "Min Indel Occurance = ", min(number_of_files_per_indel)

  for i in range(20):
    print row_index[np.argsort(number_of_files_per_indel)[::-1][i]]

  indel_count_matrix_small = np.copy(indel_count_matrix)
  indel_count_matrix_small = indel_count_matrix_small
  indel_count_matrix_small = indel_count_matrix_small[np.argsort(number_of_files_per_indel)[::-1][0:200]]
  print np.shape(indel_count_matrix_small)

  X = TSNE(n_components=2, random_state=0).fit_transform(np.transpose(indel_count_matrix_small))
  print np.shape(X)

  # Plot TSNE coloured by gene name (can be modified for colour)
  # Find the gene names and file names corresponding to each column name of indel_count_matrix
  gene_name = []
  file_name = []
  for i in range(len(unique_name_col)):
    parts_col_name = unique_name_col[i].split('-')
    gene_name.append(parts_col_name[0])
    if(len(parts_col_name[1]) == 2):
      file_name.append(parts_col_name[0] + '-' + parts_col_name[1])
    else:
      file_name.append(parts_col_name[0])
  gene_name_index = np.zeros(len(unique_name_col), dtype = int)
  file_name_index = np.zeros(len(unique_name_col), dtype = int)
  for i in range(len(unique_name_col)):
    parts_col_name = unique_name_col[i].split('-')
    gene_name_index[i] = gene_name.index(parts_col_name[0])
    if(len(parts_col_name[1]) == 2):
      file_name_index[i] = file_name.index(parts_col_name[0] + '-' + parts_col_name[1])
    else:
      file_name_index[i] = file_name.index(parts_col_name[0])

  gene_name_uniq = list(set(gene_name_index))
  file_name_uniq = list(set(file_name_index))
  gene_name_uniq.sort()
  file_name_uniq.sort()
  file_gene_map = np.zeros((len(gene_name_uniq), 5), dtype = int) - 1
  for i in range(len(gene_name_uniq)):
    k = 0
    for j in range(len(file_name_uniq)):
      if file_name[file_name_uniq[j]].split('-')[0] == gene_name[gene_name_uniq[i]]:
        file_gene_map[i, k] = file_name_uniq[j]
        k += 1


  for i in range(len(gene_name_uniq)):
    fig, ax = plt.subplots()
    ax.plot(X[:, 0], X[:, 1], marker = 'o', linestyle = '', label = 'All Genes')
    for j in range(5):
      if file_gene_map[i, j] != -1:
        ax.plot(X[file_name_index == file_gene_map[i, j], 0], X[file_name_index == file_gene_map[i, j], 1], marker = 'o', linestyle = '', label = file_name[file_gene_map[i, j]])
    ax.legend(loc='upper left')
    plt.savefig('TSNE-results/' + gene_name[gene_name_uniq[i]] + '_TSNE.pdf')
    plt.clf()

  plt.scatter(X[:, 0], X[:, 1])
  plt.savefig('all-genes.pdf')
  plt.clf()

  heat_map_inner_prod = np.matmul(np.transpose(indel_count_matrix_small), indel_count_matrix_small)
  # The next 2 lines change the plot from inner product to cosine distance
  #  For just the inner product, comment these two lines
  col_wise_norms = np.expand_dims(np.linalg.norm(indel_count_matrix_small, axis = 0), axis = 1)
  heat_map_inner_prod = np.divide(heat_map_inner_prod, np.matmul(col_wise_norms, np.transpose(col_wise_norms)))
  fig, axis = plt.subplots()
  heat_map = axis.pcolor(heat_map_inner_prod, cmap = plt.cm.Blues)
  axis.set_yticks(np.arange(heat_map_inner_prod.shape[0])+0.5, minor=False)
  axis.set_xticks(np.arange(heat_map_inner_prod.shape[1])+0.5, minor=False)
  axis.invert_yaxis()
  axis.set_yticklabels(unique_name_col, minor=False)
  axis.set_xticklabels(unique_name_col, minor=False)
  plt.xticks(fontsize=2.5, rotation=90)
  plt.yticks(fontsize=2.5)
  plt.colorbar(heat_map)
  plt.savefig('inner_product_heat_map.pdf')
  plt.clf

# Folder containing all the indel files
data_folder_name = "../IndelsData/"
#data_folder_name = "/Users/amirali/Projects/CRISPR-data/R data/AM_TechMerg_Summary/"
indel_count_matrix, unique_name_row, unique_name_col = process_count_files(data_folder_name)
analysis_count_files(indel_count_matrix, unique_name_row, unique_name_col)
