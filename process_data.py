import numpy as np
import glob

# Names of the rows (the different indels)
name_row = []
# Names of the columns (the different file names appended with the column headers)
name_col = []

# Folder containing all the indel files
data_folder = "../IndelsData/"

# Access all the indel files
# Read them to get a list of unique indels and the list of column names
# We don't yet look into the numbers
for out_file in glob.glob(data_folder + "counts-*.txt"):
  with open(out_file) as f:
    i = 0
    num_col_i = 0
    print "Reading" + out_file + " to take in row and column names"
    for line in f:
      line = line.replace('"', '')
      line = line.replace('\n', '')
      l = line.split(',')
      if i == 0:
        num_col_i = len(l)
        for j in range(len(l)):
          name_col.append(out_file[21:-4] + l[j])
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

# Create a numpy matrix of size unique_name_row x unique_name_col of integers
indel_count_matrix = np.zeros((len(unique_name_row), len(unique_name_col)), dtype = int)

# Store the corresponding values from the files into indel_count_matrix
# To do this, again access all the indel files, but also read the the indel count values this time
for out_file in glob.glob(data_folder + "counts-*.txt"):
  with open(out_file) as f:
    i = 0
    file_col_name = []
    num_col_i = 0
    print "Reading " + out_file + " for indel counts"
    for line in f:
      line = line.replace('"', '')
      line = line.replace('\n', '')
      l = line.split(',')
      if i == 0:
        num_col_i = len(l)
        for j in range(len(l)):
          file_col_name.append(out_file[21:-4] + l[j])
      else:
        row_name = ''
        # Some positions are of the form: "-23:-21D,-19:-15D", which get split by the process
        # We try to account for such things in this space
        for j in range(0, len(l) - num_col_i):
          row_name += l[j]
        for j in range(len(l) - num_col_i, len(l)):
          row_index = unique_name_row.index(row_name)
          col_index = unique_name_col.index(file_col_name[j - len(l) + num_col_i])
          if l[j] == 'NA':
            # Currently handling NA/nan values as -1. Can't find a numpy integer representation for NAN
            indel_count_matrix[row_index, col_index] = -1
          else:
            indel_count_matrix[row_index, col_index] = int(l[j])

      i += 1

# Save the indel counts and row, column name information
row_index_file = open('row_index.txt', 'w')
for row_name_val in unique_name_row:
  row_index_file.write("%s\n" % row_name_val)
col_index_file = open('column_index.txt', 'w')
for col_name_val in unique_name_col:
  col_index_file.write("%s\n" % col_name_val)
np.savetxt("indel_count_matrix.txt", indel_count_matrix)
print "Saved data, exiting."