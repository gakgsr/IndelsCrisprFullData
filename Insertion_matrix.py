import warnings
warnings.filterwarnings("ignore")
from os import listdir
import pickle
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import numbers


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


insertion_file_names = []
insertion_matrix = np.zeros((4,0))
target_indel = '-1:1I'
folder = '/Users/amirali/Projects/Ins_tbl/'
for file in listdir(folder):
    samples = []
    lines = open(folder+file,'r')
    for line_counter, line in enumerate(lines):
        if line_counter>0:
            line_splited = line.split(',')
            index = int(line_splited[-2])
            if index not in samples:
                samples.append(index)

    samples.sort()
    for sample in samples:

        insertion_file_names.append("%s-%d"  %(file[11:],sample))

    local_result_matrix = np.zeros((4,len(samples)))
    lines = open(folder + file, 'r')
    for line_counter, line in enumerate(lines):
        if line_counter>0:
            line = line[:-1].replace('"','')
            line_splited = line.split(',')
            counter = float(line_splited[-1])
            seq = line_splited[1]
            cigar = line_splited[2:-4]
            sample = int(line_splited[-2])
            if len(cigar)==1 and seq in ['A','T','C','G'] and isinstance(counter, numbers.Number):
                local_result_matrix[one_hot_index(seq), samples.index(sample)] += counter

    #print local_result_matrix
    insertion_matrix = np.concatenate((insertion_matrix, local_result_matrix), axis=1)


print np.shape(insertion_matrix)
insertion_matrix = insertion_matrix / np.reshape(np.sum(insertion_matrix, axis=0), (1, -1))

insertion_matrix_mean = np.nanmean(insertion_matrix,axis=1)
insertion_matrix_std = np.nanstd(insertion_matrix,axis=1)

print insertion_matrix_mean
print np.sum(insertion_matrix_mean)
fig, axis = plt.subplots()
plt.stem(insertion_matrix_mean)
axis.set_xticks(np.arange(4), minor=False)
axis.set_xticklabels(['A','C','G','T'], minor=False)
plt.savefig('test.pdf')
plt.clf()

print insertion_file_names
print np.shape(insertion_file_names)

insertion_file_names_standard = []
for file in insertion_file_names:
    #print "/Users/amirali/Projects/muteff/muteff-"+file[:-2]
    lines = open("/Users/amirali/Projects/muteff/muteff-"+file[:-2]).readlines()
    #print (lines[int(file[-1])].split(',')[0]).replace('"','')[6:]
    bbb =  (lines[int(file[-1])].split(',')[0]).replace('"','')[6:]
    bbb = bbb.replace('_','-')
    insertion_file_names_standard.append("%s-%s" %(file.split("-")[0], bbb )  )

print insertion_file_names_standard


name_genes_grna_unique = pickle.load(open('Tcell-files/name_genes_grna_UNIQUE.p', 'rb'))
insertion_matrix_unique = np.zeros((4,len(name_genes_grna_unique)))
for counter,name in enumerate(name_genes_grna_unique):
    if name in insertion_file_names_standard:
        insertion_matrix_unique[:,counter] = insertion_matrix[:,insertion_file_names_standard.index(name)]
    else:
        insertion_matrix_unique[:, counter] = insertion_matrix_mean


for col in range(len(name_genes_grna_unique)):
    if np.isnan(np.mean(insertion_matrix_unique[:,col])):
        insertion_matrix_unique[:,col] = insertion_matrix_mean

print np.mean(insertion_matrix_unique)
pickle.dump(insertion_matrix_unique, open('Tcell-files/insertion_matrix_UNIQUE.p', 'wb'))