import warnings
warnings.filterwarnings("ignore")
from os import listdir
import pickle
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import numbers
from scipy.stats import entropy

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
            #print cigar
            #if len(cigar)==1 and seq in ['A','T','C','G'] and isinstance(counter, numbers.Number):
            if len(cigar)==1 and cigar[0]=='-1:1I' and seq in ['A','T','C','G'] and isinstance(counter, numbers.Number):
                local_result_matrix[one_hot_index(seq), samples.index(sample)] += counter

    #print local_result_matrix
    insertion_matrix = np.concatenate((insertion_matrix, local_result_matrix), axis=1)


# print np.shape(insertion_matrix)
# print insertion_matrix
insertion_matrix = insertion_matrix / np.reshape(np.sum(insertion_matrix, axis=0), (1, -1))
# print insertion_matrix


insertion_matrix_mean = np.nanmean(insertion_matrix,axis=1)
insertion_matrix_std = np.nanstd(insertion_matrix,axis=1)

# print insertion_matrix_mean
# print np.sum(insertion_matrix_mean)
# fig, axis = plt.subplots()
# plt.stem(insertion_matrix_mean)
# axis.set_xticks(np.arange(4), minor=False)
# axis.set_xticklabels(['A','C','G','T'], minor=False)
# plt.savefig('test.pdf')
# plt.clf()
#
# print insertion_file_names
# print np.shape(insertion_file_names)

insertion_file_names_standard = []
for file in insertion_file_names:
    #print "/Users/amirali/Projects/muteff/muteff-"+file[:-2]
    lines = open("/Users/amirali/Projects/muteff/muteff-"+file[:-2]).readlines()
    #print (lines[int(file[-1])].split(',')[0]).replace('"','')[6:]
    bbb =  (lines[int(file[-1])].split(',')[0]).replace('"','')[6:]
    bbb = bbb.replace('_','-')
    insertion_file_names_standard.append("%s-%s" %(file.split("-")[0], bbb )  )

#print insertion_file_names_standard
#print np.shape(insertion_file_names_standard)


name_genes_grna_unique = pickle.load(open('Tcell-files/name_genes_grna_UNIQUE.p', 'rb'))
spacers = pickle.load(open('Tcell-files/spacer_pam_list_UNIQUE.p', 'rb'))
insertion_matrix_unique = np.zeros((4,len(name_genes_grna_unique)))
for counter,name in enumerate(name_genes_grna_unique):
    if name in insertion_file_names_standard:
        insertion_matrix_unique[:,counter] = insertion_matrix[:,insertion_file_names_standard.index(name)]
    else:
        insertion_matrix_unique[:, counter] = 0


#insertion_matrix_unique = insertion_matrix_unique / np.reshape(np.sum(insertion_matrix_unique, axis=0), (1, -1))
# for col in range(len(name_genes_grna_unique)):
#     if np.isnan(np.mean(insertion_matrix_unique[:,col])):
#         insertion_matrix_unique[:,col] = insertion_matrix_mean

#plt.stem(np.nanmean(insertion_matrix_unique,axis=1))
#plt.savefig('plots/insertion_hist.pdf')


entropy_vec = []
for col in range(len(name_genes_grna_unique)):
    if not np.isnan(np.mean(insertion_matrix_unique[:,col])) and np.mean(insertion_matrix_unique[:,col])>0:
        entropy_vec.append(entropy(insertion_matrix_unique[:,col]))
    # else:
    #     entropy_vec.append(np.nan)

print np.nanmean(entropy_vec)
print np.nanstd(entropy_vec)
print np.shape(entropy_vec)
print np.sort(entropy_vec)[::-1]
plt.hist(entropy_vec,bins=20)

insertion_matrix_unique = pickle.load(open('storage_other_cell/insertion_matrix(1:1I).p', 'rb'))


entropy_vec = []
for col in range(np.shape(insertion_matrix_unique)[1]):
    if not np.isnan(np.mean(insertion_matrix_unique[:,col])) and  np.mean(insertion_matrix_unique[:,col])>0:
        entropy_vec.append(entropy(insertion_matrix_unique[:,col]))

#entropy_vec = entropy_vec[np.logical_not(np.isnan(entropy_vec))]
#np.delete(entropy_vec, np.argwhere(np.isnan(entropy_vec)), None)
#plt.hist(entropy_vec,bins=20)
#
#
# plt.legend(['T Cell','Other Cell Types'])
# plt.title('2:1I')
# plt.xlabel('entropy')
# plt.ylabel('count')
# plt.savefig('insertion-ryan/hist(2:1I).pdf')


# file = open('insertion-ryan/-1:1I.txt','w')
# file.write("-1:1I\n")
# #A
# file.write("A\n")
# index_sorted  = np.argsort(insertion_matrix_unique[0,:])[::-1]
# for item in index_sorted[0:10]:
#     file.write("%s\n" %name_genes_grna_unique[item])
#     file.write("%s\n" %spacers[item])
#     file.write("fraction = %f\n" %insertion_matrix_unique[0,item])
#
# file.write("C\n")
# index_sorted  = np.argsort(insertion_matrix_unique[1,:])[::-1]
# for item in index_sorted[0:10]:
#     file.write("%s\n" %name_genes_grna_unique[item])
#     file.write("%s\n" %spacers[item])
#     file.write("fraction = %f\n" % insertion_matrix_unique[1, item])
#
# file.write("G\n")
# index_sorted  = np.argsort(insertion_matrix_unique[2,:])[::-1]
# print insertion_matrix_unique[2,index_sorted[0:10]]
# for item in index_sorted[0:10]:
#     file.write("%s\n" %name_genes_grna_unique[item])
#     file.write("%s\n" %spacers[item])
#     file.write("fraction = %f\n" % insertion_matrix_unique[2, item])
#
# file.write("T\n")
# index_sorted  = np.argsort(insertion_matrix_unique[3,:])[::-1]
# print insertion_matrix_unique[3,index_sorted[0:10]]
# for item in index_sorted[0:10]:
#     file.write("%s\n" %name_genes_grna_unique[item])
#     file.write("%s\n" %spacers[item])
#     file.write("fraction = %f\n" % insertion_matrix_unique[3, item])
