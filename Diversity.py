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
from scipy.stats import entropy
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

num_indels,num_sites = np.shape(indel_count_matrix)
indel_fraction_mutant_matrix = indel_count_matrix / np.reshape(np.sum(indel_count_matrix, axis=0), (1, -1))

max_grad = []
for col in range(num_sites):
    vec = np.copy(indel_fraction_mutant_matrix[:,col])
    vec = np.sort(vec)[::-1]
    max_grad.append(max(abs(np.gradient(vec))))

for col in np.argsort(max_grad)[0:20]:
    vec = np.copy(indel_fraction_mutant_matrix[:,col])
    vec = np.sort(vec)[::-1]
    plt.plot(vec[0:30])

plt.legend(['20 Sample Sites'])
plt.title('Heavy Tail')
plt.ylabel('Frequency')
plt.xlabel('Frequency Sorted Indel Index')
plt.savefig('Sharpness/heavy_tail.pdf')
plt.clf()

for col in np.argsort(max_grad)[-20:]:
    vec = np.copy(indel_fraction_mutant_matrix[:,col])
    vec = np.sort(vec)[::-1]
    plt.plot(vec[0:30])

plt.legend(['20 Sample Sites'])
plt.title('Light Tail')
plt.ylabel('Frequency')
plt.xlabel('Frequency Sorted Indel Index')
plt.savefig('Sharpness/light_tail.pdf')
plt.clf()

plt.hist(max_grad)
plt.ylabel('Count')
plt.xlabel('Max(abs(grad))')
plt.savefig('Sharpness/hist_max_grad.pdf')
plt.clf()

##########################################

entropy_vec = []
for col in range(num_sites):
    vec = np.copy(indel_fraction_mutant_matrix[:,col])
    vec = np.sort(vec)[::-1]
    vec = vec/np.sum(vec)
    entropy_vec.append(entropy(vec))
print np.size(vec)

for col in np.argsort(entropy_vec)[0:20]:
    vec = np.copy(indel_fraction_mutant_matrix[:,col])
    vec = np.sort(vec)[::-1]
    plt.plot(vec[0:30])

plt.legend(['20 Sample Sites'])
plt.title('Light Tail')
plt.ylabel('Frequency')
plt.xlabel('Frequency Sorted Indel Index')
plt.savefig('Sharpness/light_tail_entropy.pdf')
plt.clf()

for col in np.argsort(entropy_vec)[-20:]:
    vec = np.copy(indel_fraction_mutant_matrix[:,col])
    vec = np.sort(vec)[::-1]
    plt.plot(vec[0:30])

plt.legend(['20 Sample Sites'])
plt.ylabel('Frequency')
plt.title('Heavy Tail')
plt.xlabel('Frequency Sorted Indel Index')
plt.savefig('Sharpness/heavy_tail_entropy.pdf')
plt.clf()


print "entropy mean", np.mean(entropy_vec)
print "entropy median", np.median(entropy_vec)


plt.hist(entropy_vec)
plt.ylabel('Count')
plt.xlabel('Entropy')
plt.savefig('Sharpness/hist_entropy.pdf')
plt.clf()
