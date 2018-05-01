import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import pickle
import numpy as np
import csv
import math
import glob
import math
import random
from scipy import cluster
from sklearn.cluster import KMeans
from scipy.spatial.distance import hamming
import re
import ot
import time
from scipy.stats import kendalltau




print "loading name_genes_grna_unique ..."
name_genes_grna_unique = pickle.load(open('storage/name_genes_grna_unique_one_patient_per_site.p', 'rb'))
#name_genes_grna_unique = pickle.load(open('storage/name_genes_grna_unique.p', 'rb'))
print "loading name_indel_type_unique ..."
name_indel_type_unique = pickle.load(open('storage/name_indel_type_unique.p', 'rb'))
print "loading indel_count_matrix ..."
indel_count_matrix = pickle.load(open('storage/indel_count_matrix_one_patient_per_site.p', 'rb'))
#indel_count_matrix = pickle.load(open('storage/indel_count_matrix.p', 'rb'))
print "loading indel_prop_matrix ..."
indel_prop_matrix = pickle.load(open('storage/indel_prop_matrix_one_patient_per_site.p', 'rb'))
#indel_prop_matrix = pickle.load(open('storage/indel_prop_matrix.p', 'rb'))
print "loading length_indel ..."
length_indel_insertion = pickle.load(open('storage/length_indel_insertion.p', 'rb'))
length_indel_deletion = pickle.load(open('storage/length_indel_deletion.p', 'rb'))
print "loading homopolymer matrix"
homopolymer_matrix = pickle.load(open('storage/homopolymer_matrix_w-3:3.p', 'rb'))



def jaccard_distance(set1,set2):
  return 1 - float(len(list(set(set1) & set(set2)))) / len(list(set(set1) | set(set2)))


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



def indel_distance_finder(name_indel_type_unique):
    C = np.zeros((len(name_indel_type_unique),len(name_indel_type_unique)))
    for indel1_index,indel1 in enumerate(name_indel_type_unique):

        indel_locations = re.split('I|D',indel1)[:-1]
        indel_types = ''.join(c for c in indel1 if (c=='I' or c=='D'))

        deleted = np.asarray([])
        inserted = np.asarray([])

        for i in range(len(indel_types)):

            start, size = indel_locations[i].split(':')
            start = int(start)
            size = int(size)
            if start > 0:
                start = start - 1

            if indel_types[i]=='D':
                deleted = np.concatenate((deleted,  np.asarray(range(start,start+size)) ))

            if indel_types[i]=='I':
                inserted = np.concatenate((inserted, np.asarray(range(start, start + size))))

        deleted_indel1 = set(list(deleted))
        inserted_indel1 = set(list(inserted))

        for indel2_index, indel2 in enumerate(name_indel_type_unique):

            indel_locations = re.split('I|D', indel2)[:-1]
            indel_types = ''.join(c for c in indel2 if (c == 'I' or c == 'D'))

            deleted = np.asarray([])
            inserted = np.asarray([])

            for i in range(len(indel_types)):

                start, size = indel_locations[i].split(':')
                start = int(start)
                size = int(size)
                if start > 0:
                    start = start - 1

                if indel_types[i] == 'D':
                    deleted = np.concatenate((deleted, np.asarray(range(start, start + size))))

                if indel_types[i] == 'I':
                    inserted = np.concatenate((inserted, np.asarray(range(start, start + size))))

            deleted_indel2 = set(list(deleted))
            inserted_indel2 = set(list(inserted))

            C[indel1_index,indel2_index] = len(inserted_indel1 | inserted_indel2) - len(inserted_indel1 & inserted_indel2) + len(deleted_indel1 | deleted_indel2) - len(deleted_indel1 & deleted_indel2)


    return C


[number_of_indels,number_of_sites] = np.shape(indel_count_matrix)

# #C = indel_distance_finder(name_indel_type_unique)
# #pickle.dump(C, open('optimal_transport/cost_matrix_simple_start.p', 'wb'))
# C = pickle.load(open('optimal_transport/cost_matrix_simple_start.p', 'rb'))

# C = np.ones((number_of_indels,number_of_indels))
# W = np.zeros((number_of_sites,number_of_sites))
#
# indel_fraction_mutant_matrix = indel_count_matrix / np.reshape(np.sum(indel_count_matrix, axis=0), (1, -1))
# start_time = time.time()
# for i in range(number_of_sites):
#     print "i = ", i
#     elapsed_time = time.time() - start_time
#     print 'elapsed time =', elapsed_time
#     for j in range(i,number_of_sites):
#         veci = indel_fraction_mutant_matrix[:,i]
#         vecj = indel_fraction_mutant_matrix[:,j]
#
#         veci_index_sort = np.argsort(veci)[::-1]
#         vecj_index_sort = np.argsort(vecj)[::-1]
#
#         W[i,j]=ot.emd2(veci[veci_index_sort[0:20]]/np.sum(veci[veci_index_sort[0:20]]), vecj[vecj_index_sort[0:20]]/np.sum(vecj[vecj_index_sort[0:20]]), C[veci_index_sort[0:20][:, None] ,vecj_index_sort[0:20]]) # exact linear program
#
# WW = np.copy(W)
# for i in range(number_of_sites):
#     for j in range(0,i):
#         WW[i,j] = W[j,i]
# pickle.dump(WW, open('optimal_transport/w_matrix_top_20_symetric_control_all1_C.p', 'wb'))
# #pickle.dump(W, open('optimal_transport/w_matrix_simple_start.p', 'wb'))


# spacer_dict = {}
# with open('sequence_pam_gene_grna_big_file_donor_genomic_context.csv', 'rb') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#     row_counter = 0
#     for row in spamreader:
#         spacer_dict[row[0].split(',')[0]] = row[0].split(',')[4]
#
# spacer_list = []
# for site_name in name_genes_grna_unique:
#     site_name_list = site_name.split('-')
#     spacer_list.append(spacer_dict[site_name_list[1] + '-' + site_name_list[2]])
#
# D_spacer_edit = np.zeros((number_of_sites,number_of_sites))
# for i in range(number_of_sites):
#     print i
#     for j in range(number_of_sites):
#         D_spacer_edit[i,j] = levenshteinDistance(spacer_list[i],spacer_list[j]) #CHECK HERE!
#
# print D_spacer_edit
# pickle.dump(D_spacer_edit, open('optimal_transport/D_spacer_edit_context.p', 'wb'))

#
#C = pickle.load(open('optimal_transport/cost_matrix_simple_start.p', 'rb'))

# W = pickle.load(open('optimal_transport/w_matrix_simple_start.p', 'rb'))
#
# WW = np.copy(W)
# for i in range(number_of_sites):
#     for j in range(0,i):
#         WW[i,j] = W[j,i]
#
# print WW[0:6,0:6]
#
# pickle.dump(WW, open('optimal_transport/w_matrix_simple_start_symetric.p', 'wb'))


#
# J = np.zeros((number_of_sites,number_of_sites))
# topk = 15
# indel_set_matrix = np.zeros((topk, number_of_sites))
# for i in range(number_of_sites):
#     indel_set_matrix[:, i] = np.argsort(indel_count_matrix[:, i])[-topk:]
#
# for i in range(number_of_sites):
#     print i
#     for j in range(number_of_sites):
#         J[i, j] = jaccard_distance(indel_set_matrix[:, i], indel_set_matrix[:, j])
#
#
# print J[0:5,0:5]
# pickle.dump(J, open('optimal_transport/J_matrix_top15.p', 'wb'))


# #########
C = pickle.load(open('optimal_transport/cost_matrix_simple_start.p', 'rb'))
WW = pickle.load(open('optimal_transport/w_matrix_top_20_symetric_control_all1_C.p', 'rb'))
D_spacer_edit = pickle.load(open('optimal_transport/D_spacer_edit.p', 'rb'))
J = pickle.load(open('optimal_transport/J_matrix_top15.p', 'rb'))

number_of_sites = np.shape(D_spacer_edit)[0]

WW_mean = []
WW_sem = []
J_mean = []
J_sem = []
R_mean = []
R_sem = []

#top_what_vec = np.linspace(2,20,20,dtype=int)
top_what_vec = range(1,20)

for top_what in top_what_vec:

    WW_tau_list = []
    J_tau_list = []
    R_tau_list = []

    for i in range(number_of_sites):

        vec = np.copy(D_spacer_edit[i])
        vec = np.delete(vec,i)
        index_dist_edit =  np.argsort(vec)[0:top_what]

        vec = np.copy(WW[i])
        vec = np.delete(vec,i)
        index_dist_WW = np.argsort(vec)[0:top_what]

        vec = np.copy(J[i])
        vec = np.delete(vec,i)
        index_dist_J = np.argsort(vec)[0:top_what]

        temp = np.asarray(range(number_of_sites))
        np.random.shuffle(temp)
        index_dist_random = temp[0:top_what]

        tau, p_value = kendalltau(index_dist_edit,index_dist_WW)
        WW_tau_list.append(tau)

        tau, p_value = kendalltau(index_dist_edit,index_dist_J)
        J_tau_list.append(tau)

        tau, p_value = kendalltau(index_dist_edit,index_dist_random)
        R_tau_list.append(tau)


    WW_mean.append(np.mean(WW_tau_list))
    WW_sem.append(np.std(WW_tau_list) / np.sqrt(number_of_sites))

    J_mean.append(np.mean(J_tau_list))
    J_sem.append(np.std(J_tau_list) / np.sqrt(number_of_sites))

    R_mean.append(np.mean(R_tau_list))
    R_sem.append(np.std(R_tau_list) / np.sqrt(number_of_sites))

plt.figure()
plt.errorbar(top_what_vec, WW_mean, yerr=WW_sem)
plt.errorbar(top_what_vec,  J_mean, yerr=J_sem)
plt.errorbar(top_what_vec,  R_mean, yerr=R_sem)
plt.legend(['Optimal Transport' , 'Jaccard' , 'Random'])
plt.xlabel('Top-k NNs')
plt.ylabel('Kendalls tau')
plt.savefig('optimal_transport/compare_J_top20W_control_call1_C.pdf')
plt.clf