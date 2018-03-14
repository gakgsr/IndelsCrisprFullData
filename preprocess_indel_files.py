import numpy as np
import glob

def preprocess_indel_files(data_folder):
  #count_data_folder = data_folder + "sample_counts/"
  #prop_data_folder = data_folder + "sample_props/"
  count_data_folder = data_folder + "counts/"
  prop_data_folder = data_folder + "props/"
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
  length_indel = np.zeros(len(name_indel_type_unique), dtype = int)
  for each_file in glob.glob(count_data_folder + "counts-*.txt"):
    #print each_file
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
          len_indel = 0
          # Some positions are of the form: "-23:-21D,-19:-15D", which get split by the process when we call split()
          # We try to account for such things in this space
          for j in range(0, np.size(l_indel)):
            indel_type += l_indel[j]
            if l_indel[j].find('I') != -1:
              begn_end = l_indel[j].replace("I", "")
              begn_end = begn_end.split(':')
              if int(begn_end[1])>=int(begn_end[0]):
                len_indel += int(begn_end[1]) - int(begn_end[0]) + 1
            if l_indel[j].find('D') != -1:
              begn_end = l_indel[j].replace("D", "")
              begn_end = begn_end.split(':')
              if int(begn_end[1]) >= int(begn_end[0]):
                len_indel += int(begn_end[1]) - int(begn_end[0]) + 1
          # We ignore SNV, others, and no variants
          if line.find('I') != -1 or line.find('D') != -1:
            row_index = name_indel_type_unique.index(indel_type)
            length_indel[row_index] = len_indel
            for j in range(np.size(l)):
              if l[j] != 'NA':
                indel_count_matrix[row_index,col_index[j]] = float(l[j])
                #print row_index,col_index[j]

        i += 1


  # finding the index for the indels with frequency of mutatnt reads < 0.01
  rare_indel_index = []
  indel_frac_mutant_read_matrix = indel_count_matrix / np.reshape(np.sum(indel_count_matrix, axis=0), (1, -1))
  print np.shape(indel_frac_mutant_read_matrix)
  for row_index in range(np.shape(indel_frac_mutant_read_matrix)[0]):
    if max(indel_frac_mutant_read_matrix[row_index]) < 0.01:
      rare_indel_index.append(row_index)


  ##
  # Process the proportions file to get the proportions data
  indel_prop_matrix = np.zeros((len(name_indel_type_unique), len(name_genes_grna_unique)))
  for each_file in glob.glob(prop_data_folder + "proportions-*.txt"):
    #print each_file
    with open(each_file) as f:
      i = 0
      process_file = False
      for line in f:
        line = line.replace('\n', '')
        line = line.replace('_', '-')
        if i == 0:
          line = line.replace('"', '')
          l = line.split(',')
          curr_gene_name = each_file[len(prop_data_folder) + 12:-4].split('-')[0]
          col_index = []
          if "%s-%s-%s" % (curr_gene_name, l[0].split('-')[1], l[0].split('-')[2]) in name_genes_grna_unique:
            process_file = True
            for patient in range(np.size(l)):
              col_index.append(name_genes_grna_unique.index(
                "%s-%s-%s" % (curr_gene_name, l[patient].split('-')[1], l[patient].split('-')[2])))
        if i > 0 and process_file:
          l_indel = line.split('"')[1].split(',')
          l = line.split('"')[2].split(',')[1:]
          indel_type = ''
          len_indel = 0
          # Some positions are of the form: "-23:-21D,-19:-15D", which get split by the process when we call split()
          # We try to account for such things in this space
          for j in range(0, np.size(l_indel)):
            indel_type += l_indel[j]
          # We ignore SNV, others, and no variants
          if line.find('I') != -1 or line.find('D') != -1:
            row_index = name_indel_type_unique.index(indel_type)
            for j in range(np.size(l)):
              if l[j] != 'NA':
                indel_prop_matrix[row_index, col_index[j]] = float(l[j])
                #print row_index, col_index[j]

        i += 1


  ######
  ###### here we filter out all indels with mutant read frequency less than 0.01
  ######
  name_indel_type_unique = np.delete(name_indel_type_unique, rare_indel_index).tolist()
  indel_count_matrix = np.delete(indel_count_matrix, rare_indel_index, 0)
  indel_prop_matrix = np.delete(indel_prop_matrix, rare_indel_index, 0)
  length_indel = np.delete(length_indel, rare_indel_index, 0)
  ######


  # Save the indel counts, indel type, and gene-grna name information
  indel_type_file = open('indel_type.txt', 'w')
  for indel_type in name_indel_type_unique:
    indel_type_file.write("%s\n" % indel_type)
  genes_grna_file = open('genes_grna.txt', 'w')
  for genes_grna in name_genes_grna_unique:
    genes_grna_file.write("%s\n" % genes_grna)
  #np.savetxt("indel_count_matrix.txt", indel_count_matrix)
  #np.savetxt("indel_prop_matrix.txt", indel_prop_matrix)

  return name_genes_unique, name_genes_grna_unique, name_indel_type_unique, indel_count_matrix, indel_prop_matrix, length_indel
  #return name_genes_unique, name_genes_grna_unique, name_indel_type_unique , indel_count_matrix

