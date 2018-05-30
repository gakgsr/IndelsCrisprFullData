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
import csv
import os

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


def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def find_long_insertions(min_insertion_length):

    insertion_file_names = []
    global_insertion_list = []
    global_insertion_count = []

    long_seq_counter = 0
    insertion_matrix = np.zeros((4,0))
    folder = '/Users/amirali/Projects/ins-othercells/'
    for celltype in os.listdir(folder):
        if celltype != '.DS_Store':
            subfolder = folder+celltype+'/'
            for file in listdir(subfolder):
                insertion_file_names.append(file)
                local_insertion_list = []
                local_insertion_count = []
                lines = open(subfolder + file, 'r')
                for line_counter, line in enumerate(lines):
                    if line_counter > 0:
                        line = line[:-1].replace('"','')
                        line_splited = line.split(',')
                        counter = float(line_splited[-1])
                        seq = line_splited[1]
                        cigar = line_splited[2:-4]
                        sample = int(line_splited[-2])

                        if len(cigar)==1:
                            size = int(cigar[0].split(":")[1][0:-1])
                            if size > min_insertion_length and isinstance(counter, numbers.Number):
                                #print line
                                #print counter
                                local_insertion_list.append(seq)
                                local_insertion_count.append(counter)

                global_insertion_list.append(local_insertion_list)
                global_insertion_count.append(local_insertion_count)

    return global_insertion_list,global_insertion_count,insertion_file_names


# #################################################
# final_insertion_list,final_insertion_count,insertion_file_names = find_long_insertions(50)
#
# print np.shape(final_insertion_list)
# print np.shape(final_insertion_count)
# print np.shape(insertion_file_names)
#
# pickle.dump(insertion_file_names, open('long-insertions/insertion_file_names_other_cells.p', 'wb'))
# pickle.dump(final_insertion_list, open('long-insertions/long_insertion_list_other_cells_50.p', 'wb'))
# pickle.dump(final_insertion_count, open('long-insertions/long_insertion_count_other_cells_50.p', 'wb'))
#################################################

# # #################################################
long_insertion_list = pickle.load(open('long-insertions/long_insertion_list_other_cells_50.p', 'rb'))
long_insertion_count = pickle.load(open('long-insertions/long_insertion_count_other_cells_50.p', 'rb'))
insertion_file_names = pickle.load(open('long-insertions/insertion_file_names_other_cells.p', 'rb'))

# long_insertion_list_flat = []
# for list in long_insertion_list:
#     for item in list:
#         long_insertion_list_flat.append(item)
#
# print "total # of insertions = ",  len(long_insertion_list_flat)
#
# counter = 0
# total_coutner = 0
# outputs = []
# outstarts = []
# outstops = []
# outpercs = []
# for id_i, l in enumerate(long_insertion_list):
#     print("spacer: " + str(id_i))
#     outs = []
#     starts = []
#     stops = []
#     percs = []
#     for id_j, seq in enumerate(l):
#         print total_coutner
#         print seq
#         with open('/Users/amirali/Software/test2/temp.txt', 'w') as f:
#             label = '>spacer' + str(id_i) + '_insertion' + str(id_j)
#             f.write(label + '\n'+seq+'\n')
#
#         os.system('/usr/local/ncbi/blast/bin/blastn -query /Users/amirali/Software/test2/temp.txt -db ' + \
#              '/Users/amirali/Software/refgenome/blast/hg38 -outfmt 6 -out /Users/amirali/Software/test2/temp.out')
#
#
#         with open('/Users/amirali/Software/test2/temp.out', 'r') as f:
#             line = f.readline().split()
#             print line
#             if len(line)>0:
#                 chromosome_list = line[1].split('_')
#                 if len(chromosome_list)==1:
#                     chromosome = line[1].split('_')[0][3:]
#                     if chromosome == 'Y' or chromosome == 'X':
#                         chromosome = 23
#                     if chromosome == 'M': # Mitochondrial DNA
#                         chromosome = 24
#                     else:
#                         chromosome = int(chromosome)
#                 else:
#                     chromosome=25
#
#                 start = int(line[8])
#                 stop = int(line[9])
#                 perc = float(line[2])
#         #print chromosome
#         if len(line)>0:
#             outs.append(chromosome)
#             starts.append(start)
#             stops.append(stop)
#             percs.append(perc)
#
#         else:
#             outs.append(0)
#             starts.append(0)
#             stops.append(0)
#             percs.append(0)
#
#
#         print
#         #print(outs[-1])
#         #print outs
#         total_coutner+=1
#         #if len(outs[-1])>1:
#         #    counter+=1
#         #if outs[-1]==0:
#         #   counter+=1
#
#         #print 'ratio = ', counter/float(total_coutner)
#
#     outputs.append(outs)
#     outstarts.append(starts)
#     outstops.append(stops)
#     outpercs.append(percs)
#
#
# pickle.dump(outputs, open('long-insertions/blast_chr_outputs_other_cells_50.p', 'wb'))
# pickle.dump(outstarts, open('long-insertions/blast_start_outputs_other_cells_50.p', 'wb'))
# pickle.dump(outstops, open('long-insertions/blast_stop_outputs_other_cells_50.p', 'wb'))
# pickle.dump(outpercs, open('long-insertions/blast_percentage_outputs_other_cells_50.p', 'wb'))
#
# # #################################################
# location_dict = {}
# name_genes_grna_unique = pickle.load(open('Tcell-files/name_genes_grna_UNIQUE.p', 'rb'))
# with open('sequence_pam_gene_grna_big_file_donor_genomic_context.csv', 'rb') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#     row_counter = 0
#     for row in spamreader:
#         location_dict[row[0].split(',')[0]] = row[0].split(',')[4]
#
# location_vec = []
# for site_name in name_genes_grna_unique:
#     site_name_list = site_name.split('-')
#     location = location_dict[site_name_list[1] + '-' + site_name_list[2]]
#     location_vec.append(location)

location_vec = []
for name in insertion_file_names:
    location_vec.append('+'+name.split('_')[1])

print location_vec


# # #################################################
chr_length = np.asarray([248956422, 242193529, 198295559, 190214555, 181538259, 170805979, 159345973, 145138636, 138394717, 133797422, 135086622, 133275309, 114364328, 107043718, 101991189, 90338345, 83257441, 80373285, 58617616, 64444167, 46709983, 50818468, 213268310 ])
chr_length = chr_length/ float(np.sum(chr_length))

long_insertion_list_flat = []
for list in long_insertion_list:
    for item in list:
        long_insertion_list_flat.append(item)

print "total # of insertions = ",  len(long_insertion_list_flat)

chr_list = pickle.load(open('long-insertions/blast_chr_outputs_other_cells_50.p', 'rb'))
start_list = pickle.load(open('long-insertions/blast_start_outputs_other_cells_50.p', 'rb'))
stop_list = pickle.load(open('long-insertions/blast_stop_outputs_other_cells_50.p', 'rb'))
percentage_list = pickle.load(open('long-insertions/blast_percentage_outputs_other_cells_50.p', 'rb'))

chr_list_flat = []
chr_vec = np.zeros(25)
for list_counter,list in enumerate(chr_list):
    for item_counter,item in enumerate(list):
        chr_list_flat.append(item)
        if item != 0:  # chr0 means that there is no alignment
            chr_vec[item-1] += long_insertion_count[list_counter][item_counter]

#print len(chr_list_flat)
#print chr_vec[0:23]/chr_length
print chr_vec

total_ratio = 0
total_ratio_weighted_read = 0
count_ratio = 0
dist_vec = []
bound_vec = []
for list_counter,list in enumerate(chr_list):
    distance = 0
    distance_counter = 0
    bound = 0
    print location_vec[list_counter]
    print list
    chr = location_vec[list_counter].split(':')[0][4:]
    chr_start = int(location_vec[list_counter].split(':')[1].split('-')[0])
    if chr=='Y' or chr=='X':
        chr = 23
    else:
        chr = int(chr)
    if len(list) - list.count(0)>0:
        total_ratio += list.count(chr) / float(len(list) - list.count(0))
        list1 = np.copy(np.asarray(list))
        list1[list1 != chr] = 0
        total_ratio_weighted_read += np.dot(list1,np.asarray(long_insertion_count[list_counter]) ) / np.dot(np.asarray(list), np.asarray(long_insertion_count[list_counter]) )
        count_ratio+=1

    for item_counter, item in enumerate(list):
        if item == chr:
            distance += abs(chr_start - start_list[list_counter][item_counter])
            distance_counter += 1

    if distance_counter>0:
        distance = distance / float(distance_counter)
        dist_vec.append(distance)
    else:
        dist_vec.append(0)

    if len(list)>1:
        bound = 1- list.count(0)/float(len(list))

    bound_vec.append(bound)



    print np.asarray(start_list[list_counter])
    #print np.asarray(stop_list[list_counter])

dist_vec = np.asarray(dist_vec)

print chr_vec
print "Percentage of insertion seq matching the cut site chromosome", total_ratio / float(count_ratio)
print "Percentage of insertion seq matching the cut site chromosome (read number)", total_ratio_weighted_read / float(count_ratio)
print "Average alignment ratio", np.mean(bound_vec)

dist_vec = dist_vec[dist_vec>0]
dist_vec = dist_vec[dist_vec<1e7]
plt.hist(dist_vec,bins=100)
plt.savefig('long-insertions-plots/distance_hist_other_cells_50_zoom.pdf')